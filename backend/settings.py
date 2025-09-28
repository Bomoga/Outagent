from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(
            f"Missing environment variable '{name}'. Create a .env file (see .env.example) or export it in your shell."
        )
    return value


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable '{name}' must be a float, got: {raw}") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable '{name}' must be an integer, got: {raw}") from exc


@dataclass(frozen=True)
class Settings:
    api_base: str
    eia_api_key: str
    gemini_api_key: str
    wx_lat: float
    wx_lon: float
    ingest_interval_seconds: int
    fema_bbox: str
    fema_api_key: str


@lru_cache()
def get_settings() -> Settings:
    return Settings(
        api_base=os.getenv("OUTAGENT_API_BASE", "http://127.0.0.1:8000"),
        eia_api_key=_require_env("EIA_API_KEY"),
        gemini_api_key=_require_env("GEMINI_API_KEY"),
        wx_lat=_get_float("WX_LAT", 26.5225),
        wx_lon=_get_float("WX_LON", -81.1637),
        ingest_interval_seconds=_get_int("OUTAGENT_INGEST_INTERVAL", 3600),
        fema_bbox=os.getenv("FEMA_BBOX", "-81.8,25.0,-80.0,26.5"),
        fema_api_key=_require_env("FEMA_API_KEY"),
    )

