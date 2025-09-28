import asyncio
import contextlib
import logging
from pathlib import Path
from typing import List

import pandas as pd
from joblib import load
from pydantic import BaseModel
from fastapi import FastAPI, Query, HTTPException, APIRouter
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .weather_infer import forecast_weather
from backend.app import ingest_hourly
from backend.settings import get_settings

# ---------------- PATHS & CONFIG ----------------
CACHE = Path("backend/data/processed/latest/latest_features.parquet")
DATA = Path("backend/data/processed/merged/merged_hourly.parquet")
MODELS_DIR = Path("backend/models/demand")
HORIZONS = list(range(1, 13))  # Predict 12 hours ahead

# ---------------- FASTAPI APP ----------------
router = APIRouter()
app = FastAPI(default_response_class=ORJSONResponse)

# Allow frontend access (can restrict to local dev origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOGGER = logging.getLogger(__name__)
settings = get_settings()
INGEST_INTERVAL_SECONDS = settings.ingest_interval_seconds
_INGEST_TASK: asyncio.Task | None = None

# ---------------- SCHEMAS ----------------
class Observation(BaseModel):
    timestamp: str  # ISO 8601
    load_mw: float
    temp_c: float
    rh: float
    wind_mps: float
    precip_mm: float
    ghi_kwhm2: float

# ---------------- FEATURE ENGINEERING ----------------
def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

def add_load_feats(df: pd.DataFrame) -> pd.DataFrame:
    y = "load_mw"
    df["load_t1"] = df[y].shift(1)
    df["load_t24"] = df[y].shift(24)
    df["load_t168"] = df[y].shift(168)
    df["roll_mean_24"] = df[y].rolling(24).mean()
    df["roll_max_24"] = df[y].rolling(24).max()
    df["roll_std_24"] = df[y].rolling(24).std()
    return df

FEATURES = [
    "load_t1", "load_t24", "load_t168",
    "roll_mean_24", "roll_max_24", "roll_std_24",
    "temp_c", "rh", "wind_mps", "precip_mm", "ghi_kwhm2",
    "hour", "dow", "month", "is_weekend"
]

# ---------------- MODEL LOADING ----------------
MODELS: dict[int, dict] = {}
def load_models(horizons: List[int] = HORIZONS):
    """Load all horizon-specific demand forecast models."""
    global MODELS
    if MODELS:
        return  # Already loaded
    for h in horizons:
        bundle = load(MODELS_DIR / f"load_plus{h}h.joblib")
        MODELS[h] = bundle

    missing = [h for h in horizons if h not in MODELS]
    if missing:
        raise RuntimeError(f"Missing model files for horizons: {missing}")

load_models()

# ---------------- DATA HELPERS ----------------
def read_latest_feature_row() -> pd.Series:
    """Return the most recent feature row, recompute if cache missing."""
    if CACHE.exists():
        return pd.read_parquet(CACHE).iloc[-1]

    df = pd.read_parquet(DATA).sort_values("timestamp").reset_index(drop=True)
    df = add_time_feats(add_load_feats(df))
    last = df.dropna(subset=FEATURES)
    if last.empty:
        raise HTTPException(500, "Not enough history to compute features.")
    latest = last.iloc[-1:]
    latest.to_parquet(CACHE, index=False)
    return latest.iloc[-1]

def predict_next_hours(latest_row: pd.Series, hours: int) -> List[float]:
    """Predict load for the next N hours using preloaded models."""
    rows = []
    base_hour = int(latest_row["hour"])
    base_dow = int(latest_row["dow"])

    # Construct future feature rows
    for h in range(1, hours + 1):
        r = latest_row.copy()
        r["hour"] = (base_hour + h) % 24
        r["dow"] = (base_dow + (base_hour + h) // 24) % 7
        r["is_weekend"] = 1 if r["dow"] >= 5 else 0
        rows.append(r)
    Xf = pd.DataFrame(rows)

    preds: List[float] = []
    for h in range(1, hours + 1):
        bundle = MODELS.get(h, MODELS[max(MODELS.keys())])
        m, feats = bundle["model"], bundle["features"]
        yhat = float(m.predict(Xf.iloc[[h - 1]][feats])[0])
        preds.append(yhat)
        Xf.loc[h - 1 :, "load_t1"] = yhat
        if "roll_mean_24" in feats:
            Xf.loc[h - 1 :, "roll_mean_24"] = (
                Xf.loc[h - 1 :, "roll_mean_24"] * 23 / 24 + yhat / 24
            )
    return preds

# ---------------- RISK ASSESSMENT ----------------
def compute_risk_score(predictions: List[float], weather_df: pd.DataFrame) -> List[float]:
    """Compute a composite risk score [0, 1] based on demand + weather."""
    MAX_CAPACITY = 5000  # MW
    MAX_WIND = 25        # m/s threshold for severe risk
    MAX_RAIN = 50        # mm/hr flash flood threshold

    scores = []
    for yhat, (_, row) in zip(predictions, weather_df.iterrows()):
        demand_factor = min(yhat / MAX_CAPACITY, 1.0)
        wind_factor = min(row.get("wind_mps", 0) / MAX_WIND, 1.0)
        rain_factor = min(row.get("precip_mm", 0) / MAX_RAIN, 1.0)

        score = 0.5 * demand_factor + 0.3 * wind_factor + 0.2 * rain_factor

        if demand_factor > 0.7 and (wind_factor > 0.5 or rain_factor > 0.5):
            score = min(score * 1.2, 1.0)

        scores.append(round(score, 3))
    return scores

# ---------------- BACKGROUND INGEST ----------------
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

# ---------------- ROUTES ----------------
@app.get("/health")
def health():
    return {"ok": True, "models_loaded": sorted(MODELS.keys())}

@app.get("/forecast/load")
def forecast_load(hours: int = Query(12, ge=1, le=24)):
    latest = read_latest_feature_row()
    preds = predict_next_hours(latest, hours)
    return {"horizon_hours": list(range(1, hours + 1)), "prediction_mw": preds}

@app.get("/forecast/risk")
def forecast_risk(hours: int = Query(12, ge=1, le=24)):
    """Combine forecasted load + weather to produce a risk score."""
    latest = read_latest_feature_row()
    preds = predict_next_hours(latest, hours)
    weather_df = pd.read_parquet(DATA).sort_values("timestamp").tail(hours)
    scores = compute_risk_score(preds, weather_df)
    return {"horizon_hours": list(range(1, hours + 1)), "risk_score": scores}

@app.get("/history/load")
def history_load(hours: int = Query(24, ge=1, le=168)):
    if not DATA.exists():
        raise HTTPException(status_code=404, detail="No historical data found.")
    df = pd.read_parquet(DATA).sort_values("timestamp").tail(hours)
    return {
        "timestamps": df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "load_mw": df["load_mw"].tolist(),
    }

@app.get("/history/weather")
def history_weather(hours: int = Query(24, ge=1, le=168)):
    if not DATA.exists():
        raise HTTPException(status_code=404, detail="No historical data found.")
    df = pd.read_parquet(DATA).sort_values("timestamp").tail(hours)

    numeric_cols = ["temp_c", "rh", "wind_mps", "precip_mm", "ghi_kwhm2"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df[numeric_cols] = df[numeric_cols].replace({-999: pd.NA, -999.0: pd.NA})

    if df[numeric_cols].select_dtypes(include="number").shape[1] > 0:
        df[numeric_cols] = df[numeric_cols].interpolate(
            method="linear", limit_direction="both"
        )

    df["wet_bulb"] = df["temp_c"] * (df["rh"] / 100) ** 0.125
    return {
        "timestamps": df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "ghi_kwhm2": df["ghi_kwhm2"].tolist(),
        "wind_mps": df["wind_mps"].tolist(),
        "precip_mm": df["precip_mm"].tolist(),
        "wet_bulb": df["wet_bulb"].tolist(),
    }

@app.post("/ingest/hour")
def ingest_hour(obs: Observation):
    if DATA.exists():
        df = pd.read_parquet(DATA)
    else:
        df = pd.DataFrame(columns=[
            "timestamp", "load_mw", "temp_c", "rh", "wind_mps", "precip_mm", "ghi_kwhm2"
        ])
    ts = pd.to_datetime(obs.timestamp, utc=True, errors="raise")
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
