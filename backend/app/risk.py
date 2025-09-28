from fastapi import APIRouter, HTTPException

from backend.services.risk_advisor import MetricsPayload, RiskAdvice, call_gemini

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.post("", response_model=RiskAdvice)
def generate_risk(payload: MetricsPayload) -> RiskAdvice:
    try:
        return call_gemini(payload)
    except Exception as exc:  # pragma: no cover - surfaced via HTTP
        raise HTTPException(status_code=500, detail=str(exc)) from exc
