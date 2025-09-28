import os
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MASTER_DIR = BASE_DIR / "backend" / "data" / "processed" / "latest"
MASTER_WX_CSV = MASTER_DIR / "weather_hourly_master.csv"
RAW_WX_CSV    = BASE_DIR / "backend" / "data" / "processed" / "nasa" / "T2M,PRECTOT,WS2M,ALLSKY_SFC_SW_DWN,RH2M_20190101_20250920.csv"

OUT_DIR  = BASE_DIR / "backend" / "models" / "weather"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARS = ["temp_c", "rh", "wind_mps", "precip_mm", "ghi_kwhm2"]
HORIZONS = list(range(1, 13))  # 1..12h


def _load_weather_frame() -> pd.DataFrame:
    """Prefer the hourly master dataset, falling back to the raw export if needed."""

    if MASTER_WX_CSV.exists():
        df_master = _read_weather_normalized(MASTER_WX_CSV)
        if not df_master.empty:
            return df_master
        print("Weather master CSV is empty; falling back to raw export for training.")
    else:
        print("Weather master CSV missing; falling back to raw export for training.")

    return _read_weather_normalized(RAW_WX_CSV)


def _read_weather_normalized(csv_path: Path) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
      ['timestamp','temp_c','rh','wind_mps','precip_mm','ghi_kwhm2']
    Builds 'timestamp' from YEAR/MO/DY/HR or 'period' if needed.
    Converts -999 sentinels to NaN. Timestamps are naive UTC.
    """
    if not csv_path.exists():
        return pd.DataFrame(columns=["timestamp","temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"])

    df = pd.read_csv(csv_path)

    # Build timestamp if missing
    if "timestamp" not in df.columns:
        if all(c in df.columns for c in ["YEAR","MO","DY","HR"]):
            base = pd.to_datetime(dict(year=df["YEAR"], month=df["MO"], day=df["DY"]),
                                  errors="coerce", utc=True)
            hr = pd.to_numeric(df["HR"], errors="coerce").fillna(0).astype(int)
            df["timestamp"] = (base + pd.to_timedelta(hr, unit="h")).dt.tz_convert(None)
        elif "period" in df.columns:
            df["timestamp"] = pd.to_datetime(df["period"], errors="coerce", utc=True).dt.tz_convert(None)
        else:
            # Can't construct timestamp → return empty
            return pd.DataFrame(columns=["timestamp","temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"])

    # Standardize column names
    rename = {
        "T2M": "temp_c",
        "RH2M": "rh",
        "WS2M": "wind_mps",
        "ALLSKY_SFC_SW_DWN": "ghi_kwhm2",
        "PRECTOTCORR": "precip_mm",
        "PRECTOT": "precip_mm",
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # Ensure expected columns exist
    for col in ["temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"]:
        if col not in df.columns:
            df[col] = np.nan

    # Clean types / sentinels
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    for col in ["temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([-999, -999.0], np.nan)

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp","temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"]]

def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def add_time_features(df):
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    df["hour"] = ts.dt.hour
    df["dow"]  = ts.dt.dayofweek
    # cyclic encodings
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

def make_supervised(df, var, horizon):
    """
    Build supervised rows for one variable and one horizon (h hours ahead).
    Creates lags, rollups, and label var@t+h.
    """
    use = df[["timestamp", var]].copy()
    use = use.sort_values("timestamp").reset_index(drop=True)

    # lags
    for k in [1,2,3,6,12,24]:
        use[f"L{var}_{k}"] = use[var].shift(k)

    # deltas & rollups (small set to keep it fast)
    use[f"D{var}_1"] = use[var] - use[var].shift(1)
    for w in [3,6,12,24]:
        use[f"R{var}_mean_{w}"] = use[var].rolling(w).mean()
        use[f"R{var}_std_{w}"]  = use[var].rolling(w).std()

    # label at t+h
    use[f"Y{var}_h{horizon}"] = use[var].shift(-horizon)

    # time features (aligned at t)
    use = use.join(add_time_features(df)[["hour_sin","hour_cos","dow_sin","dow_cos","is_weekend"]])

    # drop incomplete
    use = use.dropna().reset_index(drop=True)

    y = use[f"Y{var}_h{horizon}"].astype(float)
    X = use.drop(columns=[f"Y{var}_h{horizon}", "timestamp", var])

    # simple temporal split: last ~30 days as validation (if present)
    if len(use) < 24*40:
        tr_idx = np.arange(len(use))
        va_idx = np.array([], dtype=int)
    else:
        cutoff = use.index.max() - 24*30  # ~30 days
        tr_idx = np.where(use.index <= cutoff)[0]
        va_idx = np.where(use.index >  cutoff)[0]

    return X, y, tr_idx, va_idx

def train_one(var):
    print(f"\n=== Training {var} ===")
    df = _load_weather_frame()
    if df.empty:
        raise RuntimeError("No usable rows in weather datasets after normalization (missing timestamp/values).")

    scores = {}
    for h in HORIZONS:
        X, y, tr, va = make_supervised(df, var, h)

        if len(tr) < 1000:
            print(f"[{var} h+{h}] too little data ({len(tr)} rows). Skipping.")
            continue

        m = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
        )
        m.fit(X.iloc[tr], y.iloc[tr])

        # evaluate
        if len(va) > 0:
            yhat = m.predict(X.iloc[va])
            score = rmse(y.iloc[va], yhat)
            scores[h] = score
            print(f"[{var} h+{h}] RMSE={score:,.3f}  (n_tr={len(tr)}, n_va={len(va)})")
        else:
            print(f"[{var} h+{h}] trained (no val split)  (n_tr={len(tr)})")

        # save
        path = OUT_DIR / f"{var}_h{h}.joblib"
        joblib.dump({"model": m, "features": list(X.columns)}, path)

    return scores

def main():
    all_scores = {}
    for v in VARS:
        all_scores[v] = train_one(v)
    # quick summary
    print("\n=== Validation RMSE by horizon ===")
    for v, d in all_scores.items():
        if not d: 
            print(f"{v}: (no scores)")
            continue
        line = ", ".join([f"h+{h}:{rmse:.1f}" for h, rmse in sorted(d.items())])
        print(f"{v}: {line}")

if __name__ == "__main__":
    main()