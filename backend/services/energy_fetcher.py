"""Utilities for retrieving and preparing EIA balancing-authority data."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx
import numpy as np
import pandas as pd

EIA_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

@dataclass(slots=True)
class EIAQuery:
    api_key: str
    frequency: str
    respondent: str
    start: str
    end: str
    metrics: Sequence[str]
    type_filter: Optional[str] = "D"
    page_size: int = 5000
    additional_facets: Optional[Dict[str, Sequence[str]]] = None

    def as_params(self, offset: int) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "api_key": self.api_key,
            "frequency": self.frequency,
            "start": self.start,
            "end": self.end,
            "offset": offset,
            "length": self.page_size,
        }
        for idx, metric in enumerate(self.metrics):
            params[f"data[{idx}]"] = metric
        if self.type_filter:
            params["facets[type][0]"] = self.type_filter
        params["facets[respondent][0]"] = self.respondent
        params["sort[0][column]"] = "period"
        params["sort[0][direction]"] = "asc"
        if self.additional_facets:
            for facet_name, values in self.additional_facets.items():
                for idx, value in enumerate(values):
                    params[f"facets[{facet_name}][{idx}]"] = value
        return params


async def fetch_eia_timeseries(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    frequency: str,
    respondent: str,
    start: datetime | str,
    end: datetime | str,
    metrics: Sequence[str] = ("value",),
    type_filter: Optional[str] = "D",
    page_size: int = 5000,
    max_pages: Optional[int] = None,
    delay_seconds: float = 0.0,
    additional_facets: Optional[Dict[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    
    query = EIAQuery(
        api_key=api_key,
        frequency=frequency,
        respondent=respondent,
        start=_ensure_timestamp_str(start),
        end=_ensure_timestamp_str(end),
        metrics=metrics,
        type_filter=type_filter,
        page_size=page_size,
        additional_facets=additional_facets,
    )

    frames: List[pd.DataFrame] = []
    offset = 0
    page = 0

    while True:
        params = query.as_params(offset)
        response = await client.get(EIA_BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        payload: Dict[str, Any] = response.json()
        resp = payload.get("response", {})
        rows = resp.get("data", [])
        total = int(resp.get("total") or 0)

        if not rows:
            break

        df = pd.DataFrame(rows)
        if df.empty:
            break

        for metric in metrics:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors="coerce")
        frames.append(df)

        offset += page_size
        page += 1
        if total and offset >= total:
            break
        if max_pages is not None and page >= max_pages:
            break
        if delay_seconds:
            await asyncio.sleep(delay_seconds)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def clean_eia_timeseries(
    df: pd.DataFrame,
    *,
    value_column: str = "value",
    period_column: str = "period",
    z_score_threshold: float = 4.0,
    interpolate_method: str = "time",
    fill_limit: Optional[int] = None,
) -> pd.DataFrame:

    if df.empty:
        return df

    cleaned = df.copy()

    if value_column in cleaned.columns:
        cleaned[value_column] = pd.to_numeric(cleaned[value_column], errors="coerce")
        cleaned.loc[cleaned[value_column] < 0, value_column] = np.nan

        mean = cleaned[value_column].mean()
        std = cleaned[value_column].std()
        if std and not np.isnan(std):
            z_scores = (cleaned[value_column] - mean) / std
            cleaned.loc[abs(z_scores) > z_score_threshold, value_column] = np.nan

    if period_column in cleaned.columns:
        cleaned[period_column] = pd.to_datetime(cleaned[period_column], errors="coerce")
        cleaned = cleaned.dropna(subset=[period_column])
        cleaned = cleaned.sort_values(period_column)
        cleaned = cleaned.drop_duplicates(subset=period_column, keep="last")
        cleaned = cleaned.set_index(period_column)

    if value_column in cleaned.columns and interpolate_method:
        cleaned[value_column] = cleaned[value_column].interpolate(
            method=interpolate_method, limit=fill_limit
        )
        cleaned[value_column] = cleaned[value_column].ffill().bfill()

    return cleaned


def engineer_energy_features(
    df: pd.DataFrame,
    *,
    value_column: str = "value",
    window_sizes: Sequence[int] = (3, 6, 12, 24),
) -> pd.DataFrame:

    if df.empty:
        return df

    features = df.copy()
    if value_column in features.columns:
        features[f"{value_column}_diff"] = features[value_column].diff()
        features[f"{value_column}_pct_change"] = features[value_column].pct_change()
        for window in window_sizes:
            rolling = features[value_column].rolling(window=window)
            features[f"{value_column}_roll_mean_{window}"] = rolling.mean()
            features[f"{value_column}_roll_std_{window}"] = rolling.std()
        daily = features.index.to_series().dt.hour
        features[f"{value_column}_by_hour"] = features.groupby(daily)[value_column].transform("mean")

    features = features.replace({np.inf: np.nan, -np.inf: np.nan})
    features = features.fillna(method="ffill").fillna(method="bfill")
    return features


def _ensure_timestamp_str(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H")
    return value
