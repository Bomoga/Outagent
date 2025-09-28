import os, math, json, google.generativeai as genai
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from backend.settings import get_settings

# Init Gemini client
settings = get_settings()
genai.configure(api_key=settings.gemini_api_key)
MODEL_NAME = "gemini-1.5-flash"  # fast + cheap, swap to pro if you prefer

# What the frontend will POST
class MetricPoint(BaseModel):
    ts: str
    value: float

class MetricsPayload(BaseModel):
    # Send only what you have. Keep names stable.
    insulation: List[MetricPoint] = []
    windspeed: List[MetricPoint] = []
    precipitation: List[MetricPoint] = []
    wet_bulb_2m: List[MetricPoint] = []
    # For power-side context (optional)
    mw_actual: List[MetricPoint] = []
    mw_forecast: List[MetricPoint] = []
    # Geography for wording
    region_label: str = "South Florida"
    horizon_hours: int = 12

class RiskAdvice(BaseModel):
    risk_level: Literal["LOW","MODERATE","HIGH","SEVERE"]
    risk_score: float = Field(ge=0, le=1)  # normalized 0..1
    summary: str
    actions: List[str]
    confidence: Literal["low","medium","high"]
    rationale_short: Optional[str] = None

def _mini_heu_score(m: MetricsPayload) -> float:
    """
    Quick local fallback: normalize a naive risk score 0..1 to combine with Gemini.
    This guards against empty or failed LLM responses.
    """
    def p95(arr):
        if not arr: return 0.0
        xs = sorted([a.value for a in arr])
        i = max(0, int(0.95 * (len(xs)-1)))
        return xs[i]
    wind = min(p95(m.windspeed) / 30.0, 1.0)           # 30 m/s ~ 67 mph
    rain = min(p95(m.precipitation) / 50.0, 1.0)       # 50 mm/hr heavy
    wetb = min(max(p95(m.wet_bulb_2m)-22, 0)/8.0, 1.0) # muggy/heat stress
    load = 0.0
    if m.mw_forecast and m.mw_actual:
        fmax = max([x.value for x in m.mw_forecast] + [1])
        amax = max([x.value for x in m.mw_actual] + [1])
        # high demand relative to recent actuals -> more grid strain
        load = min((fmax / max(amax, 1.0)) - 1.0, 1.0)
        load = max(load, 0.0)
    # weight
    score = 0.35*wind + 0.35*rain + 0.15*wetb + 0.15*load
    return max(0.0, min(score, 1.0))

SYSTEM_RULES = """
You are a grid reliability assistant for storm/outage preparedness.
Read short metric series (recent history + next 12h forecast) and return:
- A concise risk level
- 3–6 action items that are specific and practical for residents/utilities
Keep it under 120 words total. No fearmongering.
Only use information present in the metrics; avoid making up locations or times.
Always output strict JSON matching the schema provided.
"""

USER_TEMPLATE = """
Region: {region_label}
Horizon (hours): {h}

Metrics (ISO ts, value):
- Insulation (W/m^2 surrogate): {insulation}
- Windspeed (m/s): {windspeed}
- Precipitation (mm/hr): {precip}
- Wet-bulb @2m (°C): {wet}
- Power Actual (MW): {mw_act}
- Power Forecast (MW): {mw_fc}

First, qualitatively assess outage risk drivers (wind gusts >20 m/s, heavy rain >20 mm/hr,
extreme wet-bulb >26°C, sharp demand spikes). Prefer recent peaks and the next-12h outlook.

Return STRICT JSON with keys:
risk_level ∈ ["LOW","MODERATE","HIGH","SEVERE"],
risk_score ∈ [0,1],
summary (≤ 45 words),
actions (3–6 bullet strings, imperative, ≤ 14 words each),
confidence ∈ ["low","medium","high"],
rationale_short (≤ 30 words).
Do not add any other keys or prose.
"""

def call_gemini(payload: MetricsPayload) -> RiskAdvice:
    user_prompt = USER_TEMPLATE.format(
        region_label=payload.region_label,
        h=payload.horizon_hours,
        insulation=[(p.ts, round(p.value,3)) for p in payload.insulation[-24:]],
        windspeed=[(p.ts, round(p.value,3)) for p in payload.windspeed[-24:]],
        precip=[(p.ts, round(p.value,3)) for p in payload.precipitation[-24:]],
        wet=[(p.ts, round(p.value,3)) for p in payload.wet_bulb_2m[-24:]],
        mw_act=[(p.ts, round(p.value,3)) for p in payload.mw_actual[-24:]],
        mw_fc=[(p.ts, round(p.value,3)) for p in payload.mw_forecast[-24:]],
    )
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(
        [{"role":"system","parts":[SYSTEM_RULES]},
         {"role":"user","parts":[user_prompt]}],
        generation_config={"temperature":0.3, "max_output_tokens": 512}
    )
    text = resp.text.strip()
    try:
        data = json.loads(text)
        # soft sanity: blend with local heuristic so the score is never absurd
        blended = 0.6*float(data.get("risk_score",0.0)) + 0.4*_mini_heu_score(payload)
        data["risk_score"] = round(max(0.0, min(blended, 1.0)), 2)
        # normalize level from score if missing or inconsistent
        if data.get("risk_level") not in ["LOW","MODERATE","HIGH","SEVERE"]:
            s = data["risk_score"]
            level = "LOW" if s < 0.25 else "MODERATE" if s < 0.5 else "HIGH" if s < 0.75 else "SEVERE"
            data["risk_level"] = level
        return RiskAdvice(**data)
    except Exception:
        # hard fallback if parsing fails
        s = _mini_heu_score(payload)
        level = "LOW" if s < 0.25 else "MODERATE" if s < 0.5 else "HIGH" if s < 0.75 else "SEVERE"
        return RiskAdvice(
            risk_level=level,
            risk_score=round(s,2),
            summary="Automated heuristic fallback risk estimate based on wind, rain, heat, and load.",
            actions=[
                "Charge phones and backup batteries",
                "Secure loose outdoor items",
                "Check flashlights and first-aid kit",
                "Refuel vehicle and test generator outdoors",
            ],
            confidence="low",
            rationale_short="LLM response unavailable; heuristic used."
        )