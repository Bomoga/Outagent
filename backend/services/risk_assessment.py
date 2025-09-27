from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _normalize_distance_score(distance_m: Optional[float], cutoff_m: float = 1000.0) -> float:
    """Convert a distance in meters into a 0..1 proximity score.

    Closer => higher score. distance_m None => 0.0
    """
    if distance_m is None:
        return 0.0
    if distance_m <= 0:
        return 1.0
    score = max(0.0, 1.0 - (distance_m / cutoff_m))
    return min(max(score, 0.0), 1.0)


def assess_asset_risk(
    asset_id: str,
    weather_metrics: Dict[str, Any],
    flood_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assess risk for a single asset and return a dict with a risk_score and components.

    Inputs expected:
    - weather_metrics: {"storm_severity": float in 0..1, ...}
    - flood_info: {"in_flood_zone": bool, "flood_distance_m": float|None, "surge_zone_flag": bool}
    """
    storm_severity = float(weather_metrics.get("storm_severity", 0.0))
    flood_info = flood_info or {}
    in_zone = bool(flood_info.get("in_flood_zone", False))
    flood_distance = flood_info.get("flood_distance_m")
    surge_flag = bool(flood_info.get("surge_zone_flag", False))

    # Components
    storm_component = storm_severity  # already 0..1
    proximity_component = _normalize_distance_score(flood_distance, cutoff_m=5000.0)
    flood_component = 1.0 if in_zone else proximity_component
    surge_component = 1.0 if surge_flag else 0.0

    # Weighted aggregation
    # Weights chosen for PoC: storm 40%, flood 40%, surge 20%
    risk_score = (0.4 * storm_component) + (0.4 * flood_component) + (0.2 * surge_component)
    risk_score = round(min(max(risk_score, 0.0), 1.0), 3)

    return {
        "id": asset_id,
        "risk_score": risk_score,
        "components": {
            "storm": round(storm_component, 3),
            "flood": round(float(flood_component), 3),
            "surge": int(surge_component),
            "flood_distance_m": flood_distance,
        },
    }


def assess_assets(
    weather_context: Dict[str, Any],
    flood_context: Dict[str, Any],
    assets: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assess multiple assets. Returns a list of per-asset risk dicts.

    flood_context is expected to be {"assets": [ {id, in_flood_zone, flood_distance_m, surge_zone_flag}, ... ] }
    weather_context is expected to contain top-level metrics (storm_severity etc.) which will be used as a region-level driver.
    assets is an iterable of {"id":..., ...}
    """
    metrics = weather_context.get("metrics") if isinstance(weather_context, dict) else {}
    # Build a lookup for flood info by asset id
    flood_lookup: Dict[str, Dict[str, Any]] = {}
    fc_assets = flood_context.get("assets") if isinstance(flood_context, dict) else []
    for entry in fc_assets:
        if isinstance(entry, dict) and "id" in entry:
            flood_lookup[entry["id"]] = entry

    results: List[Dict[str, Any]] = []
    for a in assets:
        aid = a.get("id")
        flood_info = flood_lookup.get(aid) if aid is not None else None
        res = assess_asset_risk(str(aid), metrics or {}, flood_info)
        results.append(res)

    return results
