from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

try:
    from pydantic import ConfigDict, model_validator
except ImportError:  # pragma: no cover - pydantic v1
    ConfigDict = None  # type: ignore[assignment]
    model_validator = None  # type: ignore[assignment]

try:
    from pydantic.class_validators import root_validator  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from pydantic import root_validator  # type: ignore
    except ImportError:  # pragma: no cover
        root_validator = None  # type: ignore

_SERIES_RULES: dict[str, tuple[str, str]] = {
    "mw_actual": ("load_mw", "last"),
    "load_mw": ("load_mw", "last"),
    "mw_forecast": ("forecast_mw", "last"),
    "forecast_mw": ("forecast_mw", "last"),
    "windspeed": ("wind_mps", "max"),
    "wind_speed": ("wind_mps", "max"),
    "wind_mps": ("wind_mps", "max"),
    "precipitation": ("precip_mm", "max"),
    "precip_mm": ("precip_mm", "max"),
    "rainfall": ("precip_mm", "max"),
    "wet_bulb_2m": ("temp_c", "max"),
    "temperature": ("temp_c", "max"),
    "temp_c": ("temp_c", "max"),
    "flood_stage": ("flood_gauge_ft", "max"),
    "flood_stage_ft": ("flood_gauge_ft", "max"),
    "flood_gauge_ft": ("flood_gauge_ft", "max"),
    "outage_reports": ("outage_reports", "sum"),
    "outages": ("outage_reports", "sum"),
}


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_value_ts(item: Any) -> tuple[Optional[float], Optional[str]]:
    if isinstance(item, dict):
        ts = item.get("ts") or item.get("timestamp") or item.get("time")
        for key in ("value", "max", "min", "avg", "average", "mean", "latest"):
            if key in item:
                val = _coerce_float(item[key])
                if val is not None:
                    return val, ts
        if len(item) == 1:
            return _coerce_float(next(iter(item.values()))), ts
        return None, ts
    return _coerce_float(item), None


def _collapse_series(raw: Any, mode: str) -> tuple[Optional[float], Optional[str]]:
    val, ts = _extract_value_ts(raw)
    if val is not None:
        return val, ts
    if isinstance(raw, (list, tuple)):
        values: List[float] = []
        times: List[Optional[str]] = []
        for item in raw:
            v, t = _extract_value_ts(item)
            if v is not None:
                values.append(v)
                times.append(t)
        if not values:
            return None, None
        if mode == "max":
            idx = max(range(len(values)), key=values.__getitem__)
            return values[idx], times[idx] if idx < len(times) else None
        if mode == "min":
            idx = min(range(len(values)), key=values.__getitem__)
            return values[idx], times[idx] if idx < len(times) else None
        if mode == "sum":
            return sum(values), times[-1] if times else None
        if mode == "avg":
            return sum(values) / len(values), times[-1] if times else None
        return values[-1], times[-1] if times else None
    return None, None


def _preprocess(values: Any) -> Any:
    if not isinstance(values, dict):
        return values
    data = dict(values)
    normalized: dict[str, Any] = {}
    timestamps: List[str] = []
    for raw_key, (target, mode) in _SERIES_RULES.items():
        if raw_key not in data:
            continue
        val, ts = _collapse_series(data[raw_key], mode)
        if val is not None:
            normalized[target] = val
        if ts:
            timestamps.append(ts)
    if "timestamp" not in data and timestamps:
        normalized["timestamp"] = timestamps[-1]
    if "narrative" not in data:
        parts: List[str] = []
        region = data.get("region_label")
        if region:
            parts.append(f"Region: {region}")
        horizon = data.get("horizon_hours")
        if horizon not in (None, ""):
            parts.append(f"Horizon: {horizon}h")
        if parts:
            normalized["narrative"] = "; ".join(parts)
    if normalized:
        data.update({k: v for k, v in normalized.items() if v is not None})
    return data


def _model_dump(model: BaseModel, **kwargs):  # pragma: no cover
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)  # type: ignore[attr-defined]
    return model.dict(**kwargs)


class MetricsPayload(BaseModel):
    timestamp: Optional[str] = Field(default=None)
    load_mw: Optional[float] = Field(default=None, ge=0)
    forecast_mw: Optional[float] = Field(default=None, ge=0)
    temp_c: Optional[float] = None
    precip_mm: Optional[float] = Field(default=None, ge=0)
    wind_mps: Optional[float] = Field(default=None, ge=0)
    flood_gauge_ft: Optional[float] = None
    outage_reports: Optional[int] = Field(default=None, ge=0)
    narrative: Optional[str] = None

    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True, extra="ignore", str_strip_whitespace=True)
    else:
        class Config:
            allow_population_by_field_name = True
            anystr_strip_whitespace = True
            extra = "ignore"

    if model_validator is not None:
        @model_validator(mode="before")
        def _coerce_payload(cls, values):
            return _preprocess(values)
    elif root_validator is not None:
        @root_validator(pre=True)
        def _coerce_payload(cls, values):
            return _preprocess(values)


class RiskAdvice(BaseModel):
    score: float = Field(..., ge=0, le=100)
    level: Literal["low", "moderate", "high", "critical"]
    summary: str
    factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


def call_gemini(payload: MetricsPayload) -> RiskAdvice:
    ai_response = _call_gemini_if_available(payload)
    if ai_response is not None:
        return ai_response
    return _heuristic_advice(payload)


def _call_gemini_if_available(payload: MetricsPayload) -> Optional[RiskAdvice]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        LOGGER.warning("google.generativeai not installed; skipping Gemini call")
        return None

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    genai.configure(api_key=api_key)
    prompt = _build_prompt(payload)
    try:
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content([prompt], generation_config={"response_mime_type": "application/json"})
    except Exception as exc:  # pragma: no cover - network errors
        LOGGER.warning("Gemini request failed; using heuristic fallback: %s", exc, exc_info=True)
        return None

    text = getattr(response, "text", None)
    if not text:
        LOGGER.warning("Gemini response empty; using heuristic fallback")
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        LOGGER.warning("Gemini response not valid JSON; using fallback")
        return None
    try:
        if hasattr(RiskAdvice, "model_validate"):
            return RiskAdvice.model_validate(data)  # type: ignore[attr-defined]
        return RiskAdvice.parse_obj(data)
    except Exception as exc:
        LOGGER.warning("Gemini response invalid for schema; using fallback: %s", exc)
        return None


def _heuristic_advice(payload: MetricsPayload) -> RiskAdvice:
    total_weight = 0.0
    weighted_score = 0.0
    factors: List[str] = []
    recommendations: List[str] = []
    seen = set()

    def add_factor(text: str) -> None:
        if text and text not in seen:
            seen.add(text)
            factors.append(text)

    def add_reco(text: str) -> None:
        if text and text not in recommendations:
            recommendations.append(text)

    def fmt(value: float | int | None, unit: str = "") -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            if abs(value) >= 1000:
                return f"{value:,.0f}{unit}"
            return f"{value:,.1f}{unit}"
        return f"{value:,}{unit}"

    def apply_high(name: str, value: Optional[float], warn: float, critical: float, weight: float,
                   unit: str, warn_reco: str, crit_reco: str) -> None:
        nonlocal total_weight, weighted_score
        if value is None:
            return
        sev, level = _score_high(value, warn, critical)
        total_weight += weight
        weighted_score += sev * weight
        if level == "critical":
            add_factor(f"{name} critical at {fmt(value, unit)} (= {critical}{unit})")
            add_reco(crit_reco)
        elif level == "elevated":
            add_factor(f"{name} elevated at {fmt(value, unit)}")
            add_reco(warn_reco)

    apply_high(
        name="Real-time load",
        value=payload.load_mw,
        warn=18000,
        critical=22000,
        weight=0.18,
        unit=" MW",
        warn_reco="Request voluntary demand response to relieve load pressure.",
        crit_reco="Trigger emergency load-shedding protocol and alert system operators.",
    )

    apply_high(
        name="Forecast load",
        value=payload.forecast_mw,
        warn=20000,
        critical=24000,
        weight=0.14,
        unit=" MW",
        warn_reco="Secure fast-ramping generation ahead of peak forecast.",
        crit_reco="Lock in contingency generation and confirm reserve availability immediately.",
    )

    if payload.load_mw is not None and payload.forecast_mw:
        deviation = abs(payload.load_mw - payload.forecast_mw) / max(payload.forecast_mw, 1e-6)
        sev, level = _score_high(deviation, 0.1, 0.25)
        weight = 0.14
        total_weight += weight
        weighted_score += sev * weight
        if level == "critical":
            add_factor(f"Load/forecast imbalance of {deviation * 100:.1f}% is critical")
            add_reco("Balance generation stack immediately; investigate unit outages and demand spikes.")
        elif level == "elevated":
            add_factor(f"Load deviates from forecast by {deviation * 100:.1f}%")
            add_reco("Reconcile forecast assumptions and line up balancing resources for the next block.")

    if payload.temp_c is not None:
        temp_weight = 0.1
        hot_score, hot_level = _score_high(payload.temp_c, 32.0, 37.0)
        cold_score, cold_level = _score_low(payload.temp_c, 5.0, -5.0)
        sev, level = (hot_score, hot_level)
        description = "High heat"
        warn_reco = "Communicate conservation messaging to reduce cooling load spikes."
        crit_reco = "Deploy cooling centers and ensure grid redundancy to manage extreme heat."
        if cold_score > hot_score:
            sev, level = cold_score, cold_level
            description = "Extreme cold"
            warn_reco = "Line up gas supply and inspect heaters for increased cold-weather demand."
            crit_reco = "Activate cold-weather emergency plan and protect exposed infrastructure."
        total_weight += temp_weight
        weighted_score += sev * temp_weight
        if level == "critical":
            add_factor(f"{description} at {fmt(payload.temp_c, ' C')} is critical")
            add_reco(crit_reco)
        elif level == "elevated":
            add_factor(f"{description} at {fmt(payload.temp_c, ' C')} driving demand")
            add_reco(warn_reco)

    apply_high(
        name="Rainfall intensity",
        value=payload.precip_mm,
        warn=10.0,
        critical=25.0,
        weight=0.12,
        unit=" mm",
        warn_reco="Inspect drainage near substations and stage flood mitigation crews.",
        crit_reco="Deploy flood response teams and safeguard low-lying assets immediately.",
    )

    apply_high(
        name="Wind speed",
        value=payload.wind_mps,
        warn=10.0,
        critical=18.0,
        weight=0.12,
        unit=" m/s",
        warn_reco="Alert field crews about gusty conditions; prioritize vegetation patrols.",
        crit_reco="Suspend elevated work and prepare outage crews for wind-related faults.",
    )

    apply_high(
        name="Flood stage",
        value=payload.flood_gauge_ft,
        warn=12.0,
        critical=15.0,
        weight=0.1,
        unit=" ft",
        warn_reco="Monitor river gauges closely and safeguard substations in flood plains.",
        crit_reco="Execute flood protection plan and relocate vulnerable equipment now.",
    )

    apply_high(
        name="Outage reports",
        value=float(payload.outage_reports) if payload.outage_reports is not None else None,
        warn=25.0,
        critical=150.0,
        weight=0.12,
        unit="",
        warn_reco="Mobilize additional troubleshooters to contain growing outages.",
        crit_reco="Stand up incident command structure and broadcast customer outage alerts.",
    )

    if total_weight == 0:
        score_pct = 0.0
    else:
        score_pct = min(100.0, round((weighted_score / total_weight) * 100, 1))

    if score_pct >= 75:
        level = "critical"
        base_summary = "Critical operational risk detected; activate emergency procedures."
    elif score_pct >= 55:
        level = "high"
        base_summary = "High operational risk; take immediate mitigation steps."
    elif score_pct >= 35:
        level = "moderate"
        base_summary = "Moderate operational risk; monitor and prep contingency actions."
    else:
        level = "low"
        base_summary = "Risk is currently low based on the provided metrics."

    if factors:
        drivers = ", ".join(factors[:3])
        if len(factors) > 3:
            drivers += ", and additional factors"
        summary = f"{base_summary} Key drivers: {drivers}."
    else:
        summary = base_summary

    if not recommendations:
        recommendations.append("Continue routine monitoring and update metrics as conditions evolve.")

    return RiskAdvice(
        score=score_pct,
        level=level,
        summary=summary,
        factors=factors,
        recommendations=recommendations,
    )


def _score_high(value: float, warn: float, critical: float) -> Tuple[float, str]:
    if value >= critical:
        return 1.0, "critical"
    if value >= warn:
        span = max(critical - warn, 1e-6)
        frac = (value - warn) / span
        return 0.6 + 0.4 * max(0.0, min(1.0, frac)), "elevated"
    if warn <= 0:
        return 0.0, "nominal"
    base = max(0.0, value / warn)
    return min(0.3, 0.3 * base), "nominal"


def _score_low(value: float, warn: float, critical: float) -> Tuple[float, str]:
    if value <= critical:
        return 1.0, "critical"
    if value <= warn:
        span = max(warn - critical, 1e-6)
        frac = (warn - value) / span
        return 0.6 + 0.4 * max(0.0, min(1.0, frac)), "elevated"
    if warn <= 0:
        return 0.0, "nominal"
    base = max(0.0, warn / max(value, 1e-6))
    return min(0.3, 0.3 * base), "nominal"


def _build_prompt(payload: MetricsPayload) -> str:
    data = _model_dump(payload, exclude_none=True)
    return (
        "You are an energy-grid risk advisor for the Outagent platform. Respond strictly with JSON using the schema "
        "{'score': number 0-100, 'level': 'low'|'moderate'|'high'|'critical', 'summary': string, "
        "'factors': [string], 'recommendations': [string]} based on the metrics below.\n"
        f"Metrics: {json.dumps(data, indent=2, sort_keys=True)}\n"
    )
