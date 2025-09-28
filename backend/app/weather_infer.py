import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MASTER_DIR = BASE_DIR / "data" / "processed" / "latest"
WX_MASTER_CSV = MASTER_DIR / "weather_hourly_master.csv"
WX_RAW_CSV    = BASE_DIR / "data" / "processed" / "nasa" / "T2M,PRECTOT,WS2M,ALLSKY_SFC_SW_DWN,RH2M_20190101_20250920.csv"
MODEL_DIR = BASE_DIR / "models" / "weather"

def _read_weather_frame(min_rows: int = 48) -> pd.DataFrame:
    """Prefer the curated master dataset; fall back to the raw export when needed."""

    if WX_MASTER_CSV.exists():
        df_master = pd.read_csv(WX_MASTER_CSV)
        if not df_master.empty and len(df_master) >= min_rows:
            return df_master
        print(
            f"Weather master CSV has {len(df_master)} rows; falling back to raw export for history."
        )
    else:
        print("Weather master CSV missing; using raw export instead.")

    if WX_RAW_CSV.exists():
        return pd.read_csv(WX_RAW_CSV)

    return pd.DataFrame()


def _infer_hourly_index_from_name(csv_path: Path, rows: int) -> pd.DatetimeIndex | None:
    if rows <= 0:
        return None

    stem = csv_path.stem
    parts = stem.split('_')
    if len(parts) < 2:
        return None

    start_token, end_token = parts[-2], parts[-1]
    try:
        start_ts = pd.to_datetime(start_token, format="%Y%m%d", utc=True)
    except ValueError:
        return None

    try:
        idx = pd.date_range(start=start_ts, periods=rows, freq="h")
    except Exception:
        return None

    try:
        end_ts = pd.to_datetime(end_token, format="%Y%m%d", utc=True)
        expected_end = idx[-1]
        if abs((expected_end.normalize() - end_ts).total_seconds()) > 86400:
            return None
    except Exception:
        pass

    return idx.tz_convert(None) if getattr(idx, 'tz', None) is not None else idx


VARS = ["temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"]
H    = list(range(1,13))

def _time_feats(ts):
    ts = pd.to_datetime(ts)
    hour = ts.hour; dow = ts.dayofweek
    return {
        "hour_sin": np.sin(2*np.pi*hour/24),
        "hour_cos": np.cos(2*np.pi*hour/24),
        "dow_sin":  np.sin(2*np.pi*dow/7),
        "dow_cos":  np.cos(2*np.pi*dow/7),
        "is_weekend": int(dow>=5),
    }

def _latest_frame():
    df = _read_weather_frame()
    if df.empty:
        raise RuntimeError("No weather data available. Run ingest to populate the master CSV.")

    if "timestamp" not in df.columns:
        source_path = WX_RAW_CSV if WX_RAW_CSV.exists() else WX_MASTER_CSV
        inferred_idx = _infer_hourly_index_from_name(source_path, len(df))
        if inferred_idx is None:
            raise RuntimeError("Weather CSV lacks timestamp column and cannot infer hourly index.")
        df["timestamp"] = inferred_idx
        print(f"Inferred hourly timestamps for {source_path.name} based on filename.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    rename = {
        "T2M": "temp_c",
        "RH2M": "rh",
        "WS2M": "wind_mps",
        "ALLSKY_SFC_SW_DWN": "ghi_kwhm2",
        "PRECTOTCORR": "precip_mm",
        "PRECTOT": "precip_mm",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    expected_cols = ["temp_c", "rh", "wind_mps", "precip_mm", "ghi_kwhm2"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[["timestamp"] + expected_cols]

    numeric_cols = [col for col in ["temp_c", "rh", "wind_mps", "precip_mm", "ghi_kwhm2"] if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df[numeric_cols] = df[numeric_cols].replace({-999: np.nan, -999.0: np.nan})

    if len(df) < 48:
        raise RuntimeError(
            "Weather history has fewer than 48 rows; run the hourly ingest or backfill history first."
        )

    history_hours = max(48, 24 * 14)
    if len(df) > history_hours:
        df = df.tail(history_hours)

    if "rh" in df.columns:
        df["rh"] = df["rh"].clip(0, 100)
    for c in ["wind_mps", "precip_mm", "ghi_kwhm2"]:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)

    return df.reset_index(drop=True)


def _feat_row(df, var, now_ts):
    # build the same features as in training for a single timestamp
    sub = df[["timestamp", var]].copy().sort_values("timestamp")
    use = sub.set_index("timestamp")[var].astype(float)
    now_ts = pd.Timestamp(now_ts)

    if now_ts not in use.index:
        raise RuntimeError(f"Timestamp {now_ts} not found in dataset for {var}.")

    feats: dict[str, float] = {}
    for k in [1, 2, 3, 6, 12, 24]:
        val = use.shift(k).get(now_ts, np.nan)
        feats[f"L{var}_{k}"] = float(val) if pd.notna(val) else np.nan

    diff_val = use.diff().get(now_ts, np.nan)
    feats[f"D{var}_1"] = float(diff_val) if pd.notna(diff_val) else np.nan

    for w in [3, 6, 12, 24]:
        mean_val = use.rolling(w).mean().get(now_ts, np.nan)
        std_val = use.rolling(w).std().get(now_ts, np.nan)
        feats[f"R{var}_mean_{w}"] = float(mean_val) if pd.notna(mean_val) else np.nan
        feats[f"R{var}_std_{w}"] = float(std_val) if pd.notna(std_val) else np.nan

    feats.update(_time_feats(now_ts))
    return {k: (float(v) if pd.notna(v) else np.nan) for k, v in feats.items()}
def forecast_weather(hours=12):
    df = _latest_frame()
    now_ts = df["timestamp"].max()  # the last ingested hour
    out = {v: [] for v in VARS}
    horizon_list = list(range(1, hours+1))

    # rolling-forward: for h>1, we do not feed our own preds back (keeps it simple & stable for demo)
    for h in horizon_list:
        ts_h = now_ts + pd.Timedelta(hours=h)
        for v in VARS:
            model_path = MODEL_DIR / f"{v}_h{h}.joblib"
            if not model_path.exists():
                out[v].append(None); continue
            bundle = joblib.load(model_path)
            model = bundle["model"]; feats = bundle["features"]

            xrow = _feat_row(df, v, now_ts)  # features at time 'now'
            X = pd.DataFrame([xrow])[feats].ffill(axis=1).fillna(0.0)
            pred = float(model.predict(X)[0])
            # clip sane bounds
            if v == "rh": pred = float(np.clip(pred, 0, 100))
            if v in ("wind_mps","precip_mm","ghi_kwhm2"): pred = float(max(pred, 0.0))
            out[v].append(pred)
    return {"horizon_hours": horizon_list, "pred": out}
