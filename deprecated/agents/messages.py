from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel


class EnergyHistoryReadyPayload(BaseModel):
    respondent: str
    cache_path: str
    rows: int
    start: datetime
    end: datetime
