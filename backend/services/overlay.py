import math
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
import pandas as pd

from backend.services import risk_scoring, weather_fetcher

DEFAULT_REGION = {
    "min_lat": 24.5,
    "max_lat": 31.0,
    "min_lng": -87.7,
    "max_lng": -80.0,
    "step": 0.5,
}


@dataclass(frozen=True)
class GridPoint:
    lat: float
    lng: float
    flood_exposure: float


def _frange(start: float, stop: float, step: float) -> Iterator[float]:
    value = start
    while value <= stop + 1e-9:
        yield round(value, 4)
        value += step


def _estimate_flood_exposure(lat: float, lng: float) -> float:
    east = math.exp(-((lng + 80.0) / 0.6) ** 2)
    west = math.exp(-((lng + 82.8) / 0.8) ** 2)
    south = math.exp(-((lat - 24.5) / 0.7) ** 2)
    inland_wetland = math.exp(-((lat - 28.0) / 1.2) ** 2) * 0.6
    exposure = max(east, west, south, inland_wetland)
    return float(np.clip(exposure, 0.0, 1.0))


def _build_grid(region: dict | None = None) -> List[GridPoint]:
    cfg = region or DEFAULT_REGION
    points: List[GridPoint] = []
    for lat in _frange(cfg["min_lat"], cfg["max_lat"], cfg["step"]):
        for lng in _frange(cfg["min_lng"], cfg["max_lng"], cfg["step"]):
            exposure = _estimate_flood_exposure(lat, lng)
            points.append(GridPoint(lat, lng, exposure))
    return points


def build_flood_overlay(region: dict | None = None) -> dict:
    grid = _build_grid(region)
    features = []
    for point in grid:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "flood_exposure": point.flood_exposure,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [point.lng, point.lat],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_risk_overlay(
    *,
    horizon_hours: int = 12,
    region: dict | None = None,
) -> dict:
    grid = _build_grid(region)
    start = pd.Timestamp.utcnow().floor("h")
    index = pd.date_range(start=start, periods=horizon_hours, freq="h", tz="UTC")

    # Synthetic demand series with daily pattern
    hours = np.arange(len(index))
    base_load = 42000 + 6000 * np.sin((hours + 6) / 24 * 2 * np.pi)
    demand = pd.Series(base_load, index=index)
    demand_diff = demand.diff().fillna(0.0)

    weather = weather_fetcher.load_weather_features(
        start.to_pydatetime(),
        (index[-1]).to_pydatetime(),
    )
    weather = weather.reindex(index).ffill().bfill()
    weather["rain_mm_1h"] = 3.0 * np.maximum(0, np.sin((hours - 4) / 12 * np.pi))
    weather["rain_mm_6h"] = weather["rain_mm_1h"].rolling(6, min_periods=1).sum()

    base_features = pd.DataFrame(
        {
            "value": demand,
            "value_diff": demand_diff,
        },
        index=index,
    ).join(weather, how="left").ffill().bfill()

    features = []
    for timestamp, row in base_features.iterrows():
        row_df = pd.DataFrame([row])
        for point in grid:
            row_df_gp = row_df.copy()
            row_df_gp["flood_exposure"] = point.flood_exposure
            scored = risk_scoring.compute_risk(row_df_gp)
            risk_value = float(scored["risk"].iloc[0])
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "timestamp": timestamp.isoformat(),
                        "risk": risk_value,
                        "flood_exposure": point.flood_exposure,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [point.lng, point.lat],
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}
