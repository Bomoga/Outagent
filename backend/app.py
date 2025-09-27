from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.services.overlay import build_flood_overlay, build_risk_overlay

app = FastAPI(title="Outagent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/flood-overlay")
def flood_overlay() -> dict:
    return build_flood_overlay()


@app.get("/api/risk-overlay")
def risk_overlay(horizon_hours: int = 12) -> dict:
    return build_risk_overlay(horizon_hours=horizon_hours)


@app.get("/api/overlay")
def combined_overlay(horizon_hours: int = 12) -> dict:
    return {
        "flood": build_flood_overlay(),
        "risk": build_risk_overlay(horizon_hours=horizon_hours),
    }
