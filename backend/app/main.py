import asyncio
import contextlib
import logging
from fastapi import FastAPI, Query, HTTPException
from pathlib import Path
from joblib import load
import pandas as pd
from typing import List, Dict
from pydantic import BaseModel
from fastapi.responses import ORJSONResponse
from fastapi import APIRouter
from .weather_infer import forecast_weather
from backend.app import ingest_hourly
from backend.settings import get_settings


CACHE = Path("backend/data/processed/latest/latest_features.parquet")
DATA = Path("backend/data/processed/merged/merged_hourly.parquet")
MODELS_DIR = Path("backend/models/demand")
HORIZONS = list(range(1, 13))  # trained horizons

class Observation(BaseModel):
    timestamp: str
    load_mw: float
    temp_c: float
    rh: float
    wind_mps: float
    precip_mm: float
    ghi_kwhm2: float

router = APIRouter()
app = FastAPI(default_response_class=ORJSONResponse)

LOGGER = logging.getLogger(__name__)
settings = get_settings()
INGEST_INTERVAL_SECONDS = settings.ingest_interval_seconds
_INGEST_TASK: asyncio.Task | None = None

# Feature engineering (must match training)
def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

def add_load_feats(df: pd.DataFrame) -> pd.DataFrame:
    y = "load_mw"
    df["load_t1"]   = df[y].shift(1)
    df["load_t24"]  = df[y].shift(24)
    df["load_t168"] = df[y].shift(168)
    df["roll_mean_24"] = df[y].rolling(24).mean()
    df["roll_max_24"]  = df[y].rolling(24).max()
    df["roll_std_24"]  = df[y].rolling(24).std()
    return df

FEATURES = [
    "load_t1","load_t24","load_t168",
    "roll_mean_24","roll_max_24","roll_std_24",
    "temp_c","rh","wind_mps","precip_mm","ghi_kwhm2",
    "hour","dow","month","is_weekend"
]


MODELS = {}  # {h: {"model": lgb, "features": FEATURES}}
def load_models(horizons: List[int] = HORIZONS):
    global MODELS
    if MODELS:
        return
    for h in horizons:
        bundle = load(MODELS_DIR / f"load_plus{h}h.joblib")
        MODELS[h] = bundle
    # sanity: all share the same features list
    missing = [h for h in horizons if h not in MODELS]
    if missing:
        raise RuntimeError(f"Missing model files for horizons: {missing}")

load_models()


def read_latest_feature_row() -> pd.Series:
    if CACHE.exists():
        return pd.read_parquet(CACHE).iloc[-1]
    # Fallback (first run): compute from full data once
    df = pd.read_parquet(DATA).sort_values("timestamp").reset_index(drop=True)
    df = add_time_feats(add_load_feats(df))
    last = df.dropna(subset=FEATURES)
    if last.empty:
        raise HTTPException(500, "Not enough history to compute features.")
    latest = last.iloc[ -1 : ]  # DataFrame
    latest.to_parquet(CACHE, index=False)
    return latest.iloc[-1]

# Forecast logic (semi-autoregressive roll-forward) 
def predict_next_hours(latest_row: pd.Series, hours: int) -> List[float]:
    # Build future rows with rolled calendar fields
    rows = []
    base_hour = int(latest_row["hour"])
    base_dow = int(latest_row["dow"])
    for h in range(1, hours + 1):
        r = latest_row.copy()
        r["hour"] = (base_hour + h) % 24
        r["dow"] = (base_dow + (base_hour + h) // 24) % 7
        r["is_weekend"] = 1 if r["dow"] >= 5 else 0
        rows.append(r)
    Xf = pd.DataFrame(rows)

    preds: List[float] = []
    for h in range(1, hours + 1):
        bundle = MODELS.get(h)
        if bundle is None:
            # If you only trained up to 12h, cap there
            bundle = MODELS[max(MODELS.keys())]
        m, feats = bundle["model"], bundle["features"]
        yhat = float(m.predict(Xf.iloc[[h-1]][feats])[0])
        preds.append(yhat)
        # feed-forward so later horizons can depend on earlier predictions
        Xf.loc[h-1:, "load_t1"] = yhat
        # quick approximate update for rolling mean (optional)
        if "roll_mean_24" in feats:
            Xf.loc[h-1:, "roll_mean_24"] = (
                Xf.loc[h-1:, "roll_mean_24"] * 23/24 + yhat/24
            )
    return preds

# Background ingest helpers
async def _run_ingest_once_async() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ingest_hourly.ingest_once)


async def _ingest_loop() -> None:
    while True:
        try:
            await _run_ingest_once_async()
        except Exception:
            LOGGER.exception("Hourly ingest failed")
        await asyncio.sleep(INGEST_INTERVAL_SECONDS)


@app.on_event("startup")
async def _start_ingest_loop() -> None:
    global _INGEST_TASK
    if _INGEST_TASK is None:
        _INGEST_TASK = asyncio.create_task(_ingest_loop())


@app.on_event("shutdown")
async def _stop_ingest_loop() -> None:
    global _INGEST_TASK
    if _INGEST_TASK is not None:
        _INGEST_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _INGEST_TASK
        _INGEST_TASK = None

# Public endpoints 
@app.get("/health")
def health():
    return {"ok": True, "models_loaded": sorted(MODELS.keys())}

@app.get("/forecast/load")
def forecast_load(hours: int = Query(12, ge=1, le=24)):
    latest = read_latest_feature_row()
    preds = predict_next_hours(latest, hours)
    return {
        "horizon_hours": list(range(1, hours + 1)),
        "prediction_mw": preds
    }

# Ingestion to append new hourly observation 
    timestamp: str            # ISO, e.g. "2025-09-27T16:00:00"
    load_mw: float
    temp_c: float
    rh: float
    wind_mps: float
    precip_mm: float
    ghi_kwhm2: float

@app.post("/ingest/hour")
def ingest_hour(obs: Observation):
    # load/append as before...
    if DATA.exists():
        df = pd.read_parquet(DATA)
    else:
        df = pd.DataFrame(columns=[
            "timestamp","load_mw","temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"
        ])
    try:
        ts = pd.to_datetime(obs.timestamp, utc=True, errors="raise")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="timestamp must be an ISO 8601 datetime string (e.g. 2025-09-28T00:00:00Z)",
        ) from exc

    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)

    row = {
        "timestamp": ts,
        "load_mw": obs.load_mw,
        "temp_c": obs.temp_c,
        "rh": obs.rh,
        "wind_mps": obs.wind_mps,
        "precip_mm": obs.precip_mm,
        "ghi_kwhm2": obs.ghi_kwhm2,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(DATA, index=False)

    # Recompute ONLY the last ~200 hours to get fresh lags/rolls
    tail = df.tail(200).copy()
    tail = add_time_feats(add_load_feats(tail))
    latest = tail.dropna(subset=FEATURES).iloc[-1:]
    latest.to_parquet(CACHE, index=False)

    return {"ok": True, "rows": int(len(df))}

@router.get("/forecast/weather")
def forecast_weather_api(hours: int = 12):
    hours = max(1, min(24, hours))
    return forecast_weather(hours=hours)

app.include_router(router)