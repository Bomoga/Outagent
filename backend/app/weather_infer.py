import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
WX_CSV   = BASE_DIR / "data" / "processed" / "nasa" / "weather_master.csv"
MODEL_DIR= BASE_DIR.parents[0] / "models" / "weather"  # ../models/weather

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
    df = pd.read_csv(WX_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").dropna(subset=["timestamp"])
    # basic clipping to be safe
    df["rh"] = df["rh"].clip(0,100)
    for c in ["wind_mps","precip_mm","ghi_kwhm2"]:
        if c in df.columns: df[c] = df[c].clip(lower=0)
    return df

def _feat_row(df, var, now_ts):
    # build the same features as in training for a single timestamp
    sub = df[["timestamp", var]].copy().sort_values("timestamp")
    tmax = sub["timestamp"].max()
    assert pd.Timestamp(now_ts) <= tmax, "now_ts beyond data"
    use = sub.set_index("timestamp")[var].astype(float)

    feats = {}
    # lags
    for k in [1,2,3,6,12,24]:
        feats[f"L{var}_{k}"] = float(use.loc[now_ts] - use.shift(k).reindex(use.index).loc[now_ts] + use.shift(k).reindex(use.index).loc[now_ts]) if now_ts in use.index else float("nan")
        # the expression above ensures KeyError-free; simpler:
        try:
            feats[f"L{var}_{k}"] = float(use.shift(k).loc[now_ts])
        except Exception:
            feats[f"L{var}_{k}"] = np.nan
    # delta
    try:
        feats[f"D{var}_1"] = float(use.loc[now_ts] - use.shift(1).loc[now_ts])
    except Exception:
        feats[f"D{var}_1"] = np.nan
    # rollups
    for w in [3,6,12,24]:
        feats[f"R{var}_mean_{w}"] = float(use.rolling(w).mean().reindex(use.index).loc[now_ts]) if now_ts in use.index else np.nan
        feats[f"R{var}_std_{w}"]  = float(use.rolling(w).std().reindex(use.index).loc[now_ts]) if now_ts in use.index else np.nan

    feats.update(_time_feats(now_ts))
    return feats

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
            X = pd.DataFrame([xrow])[feats].fillna(method="ffill", axis=1).fillna(0.0)
            pred = float(model.predict(X)[0])
            # clip sane bounds
            if v == "rh": pred = float(np.clip(pred, 0, 100))
            if v in ("wind_mps","precip_mm","ghi_kwhm2"): pred = float(max(pred, 0.0))
            out[v].append(pred)
    return {"horizon_hours": horizon_list, "pred": out}