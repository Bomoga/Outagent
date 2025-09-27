"""Synthetic weather feature generation for development and testing."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional

import numpy as np
import pandas as pd


def load_weather_features(
    start: datetime,
    end: datetime,
    *,
    freq: str = "1h",
) -> pd.DataFrame:
    """Generate placeholder weather features aligned to an hourly index.

    In production this function should call an actual weather API (e.g., NASA POWER
    or NOAA). For development we synthesise a deterministic signal based on time.
    """

    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    else:
        start = start.astimezone(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    else:
        end = end.astimezone(UTC)

    index = pd.date_range(start=start, end=end, freq=freq, tz=UTC)
    if index.empty:
        return pd.DataFrame(index=index)

    hours = index.hour.to_numpy()
    day_of_year = index.dayofyear.to_numpy()

    temperature = 24 + 6 * np.sin(2 * np.pi * hours / 24) + 2 * np.cos(2 * np.pi * day_of_year / 365)
    humidity = 60 + 15 * np.cos(2 * np.pi * hours / 24)
    wind_speed = 3 + 0.8 * np.sin(2 * np.pi * day_of_year / 365)

    return pd.DataFrame(
        {
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "wind_speed_mps": wind_speed,
        },
        index=index,
    )

