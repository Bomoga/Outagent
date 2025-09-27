import numpy as np
import pandas as pd


def _clip01(values):
    return np.clip(values, 0.0, 1.0)


def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    # Wind component
    wind = features.get("wind_speed_mps", pd.Series(0.0, index=features.index)).astype(float)
    gust = 1.4 * wind
    features["wind_component"] = _clip01((gust - 12.0) / (25.0 - 12.0))

    # Rain components
    rain_1h = features.get("rain_mm_1h", pd.Series(0.0, index=features.index)).astype(float)
    rain_6h = features.get("rain_mm_6h", pd.Series(0.0, index=features.index)).astype(float)
    features["rain_component"] = _clip01(rain_1h / 20.0)

    flood_exposure = features.get("flood_exposure", pd.Series(0.0, index=features.index)).astype(float)
    features["flood_component"] = flood_exposure * _clip01(rain_6h / 60.0)

    demand = features.get("value", pd.Series(0.0, index=features.index)).astype(float)
    demand_diff = features.get("value_diff", pd.Series(0.0, index=features.index)).astype(float)

    if len(demand) > 1:
        demand_nonzero = demand.replace(0, np.nan)
        ramp_nonzero = demand_diff.abs().replace(0, np.nan)
        p95_demand = demand_nonzero.quantile(0.95)
        if not np.isfinite(p95_demand) or p95_demand <= 0:
            p95_demand = max(demand_nonzero.mean(), 1.0)
        p95_ramp = ramp_nonzero.quantile(0.95)
        if not np.isfinite(p95_ramp) or p95_ramp <= 0:
            p95_ramp = max(ramp_nonzero.mean(), 1.0)
    else:
        p95_demand = max(float(demand.iloc[0]), 1.0) if len(demand) else 1.0
        p95_ramp = max(float(abs(demand_diff.iloc[0])), 1.0) if len(demand_diff) else 1.0

    features["demand_level"] = _clip01(demand / p95_demand)
    features["demand_ramp"] = _clip01(demand_diff.abs() / p95_ramp)

    return features


def compute_risk(df: pd.DataFrame) -> pd.DataFrame:
    comp = compute_components(df)
    risk = (
        0.40 * comp["wind_component"]
        + 0.20 * comp["rain_component"]
        + 0.15 * comp["flood_component"]
        + 0.15 * comp["demand_level"]
        + 0.10 * comp["demand_ramp"]
    )
    comp["risk"] = _clip01(risk.astype(float))
    return comp
