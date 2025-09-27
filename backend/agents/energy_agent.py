from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState

from backend.agents.base_agent import AgentConfig, BaseAgent
from backend.services import energy_fetcher


@dataclass(slots=True)
class CachedFeatures:
    frame: pd.DataFrame
    value_column: str = "value"


class EnergyAgentConfig(AgentConfig):
    """Configuration specific to the energy agent."""

    eia_api_key: Optional[str] = None
    eia_api_key_env_var: str = "EIA_API_KEY"
    default_balancing_authorities: List[str] = Field(default_factory=lambda: ["PJM"])
    default_frequency: str = "hourly"
    default_type_filter: Optional[str] = "D"
    default_metrics: List[str] = Field(default_factory=lambda: ["value"])
    default_lookback_hours: int = 168
    default_page_size: int = 5000
    default_max_rows: int = 168


class FetchEnergyPayload(BaseModel):
    """Command payload for fetching EIA data."""

    respondent: Optional[str] = None
    respondents: Optional[List[str]] = None
    frequency: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Optional[List[str]] = None
    type_filter: Optional[str] = None
    page_size: Optional[int] = None
    include_cleaned: bool = True
    include_features: bool = True
    z_score_threshold: float = 4.0
    interpolate_method: str = "time"
    max_rows: Optional[int] = None
    additional_facets: Optional[Dict[str, List[str]]] = None


class ForecastEnergyPayload(BaseModel):
    """Command payload for running an outage forecast."""

    respondent: str
    horizon_hours: int = 24


class EnergyAgent(BaseAgent):
    """Agent that orchestrates EIA data ingestion and energy outage forecasting."""

    def __init__(self, config: EnergyAgentConfig) -> None:
        super().__init__(config)
        self._api_key: Optional[str] = None
        self._feature_cache: Dict[str, CachedFeatures] = {}

    @property
    def energy_config(self) -> EnergyAgentConfig:
        return self.config  # type: ignore[return-value]

    @classmethod
    def get_agent_card(cls) -> AgentCard:
        return AgentCard(
            name="energy-agent",
            version="0.1.0",
            description=(
                "Fetches EIA balancing-authority data, engineers stability features, "
                "and provides outage risk signals for downstream agents."
            ),
            url="https://outagent.local/a2a/energy",
            default_input_modes=["application/json"],
            default_output_modes=["application/json"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(
                    id="energy.fetch",
                    name="Fetch EIA demand/generation",
                    description=(
                        "Acquire and clean balancing-authority timeseries from the EIA API, "
                        "optionally deriving engineered features."
                    ),
                    tags=["energy", "data-ingestion", "eia"],
                ),
                AgentSkill(
                    id="energy.forecast",
                    name="Energy outage forecast",
                    description="Generate short-term outage risk indicators using recent energy features.",
                    tags=["energy", "forecast", "risk"],
                ),
            ],
        )

    async def bootstrap(self, ctx) -> None:
        self.logger.info("energy_agent_bootstrap")
        cfg = self.energy_config
        self._api_key = cfg.eia_api_key or os.getenv(cfg.eia_api_key_env_var)
        if not self._api_key:
            self.logger.warning(
                "missing_eia_api_key",
                env_var=cfg.eia_api_key_env_var,
            )
        self.logger.info("energy_agent_ready")

    async def shutdown(self, ctx) -> None:
        self.logger.info("energy_agent_shutdown")
        self._feature_cache.clear()

    @BaseAgent.message_handler("energy.fetch", payload_model=FetchEnergyPayload)
    async def handle_fetch_energy(self, ctx, event_queue, payload: FetchEnergyPayload, message) -> None:
        cfg = self.energy_config
        api_key = self._api_key or cfg.eia_api_key or os.getenv(cfg.eia_api_key_env_var)
        if not api_key:
            await self.send_error(event_queue, ctx, "EIA API key is not configured.")
            return

        respondents = self._resolve_respondents(payload)
        window_start, window_end = self._resolve_window(payload)
        metrics = payload.metrics or cfg.default_metrics
        value_column = metrics[0] if metrics else "value"
        page_size = payload.page_size or cfg.default_page_size
        type_filter = payload.type_filter or cfg.default_type_filter
        max_rows = payload.max_rows or cfg.default_max_rows

        await self.send_status_update(
            event_queue,
            ctx,
            state=TaskState.working,
            status_message=f"Fetching EIA data for {', '.join(respondents)}.",
        )

        for respondent in respondents:
            try:
                raw_df = await energy_fetcher.fetch_eia_timeseries(
                    client=self.http_client,
                    api_key=api_key,
                    frequency=payload.frequency or cfg.default_frequency,
                    respondent=respondent,
                    start=window_start,
                    end=window_end,
                    metrics=metrics,
                    type_filter=type_filter,
                    page_size=page_size,
                    additional_facets=payload.additional_facets,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("eia_fetch_failed", respondent=respondent, error=str(exc))
                await self.send_error(event_queue, ctx, f"Failed to fetch EIA data for {respondent}: {exc}")
                continue

            clean_df: Optional[pd.DataFrame] = None
            features_df: Optional[pd.DataFrame] = None
            features_preview: Optional[pd.DataFrame] = None

            if payload.include_cleaned and not raw_df.empty:
                clean_df = energy_fetcher.clean_eia_timeseries(
                    raw_df,
                    value_column=value_column,
                    z_score_threshold=payload.z_score_threshold,
                    interpolate_method=payload.interpolate_method,
                )

            if clean_df is not None and not clean_df.empty:
                features_df = energy_fetcher.engineer_energy_features(
                    clean_df,
                    value_column=value_column,
                )
                if payload.include_features:
                    features_preview = features_df
            else:
                features_df = None

            to_cache = None
            for candidate in (features_df, clean_df, raw_df):
                if candidate is not None and not candidate.empty:
                    to_cache = candidate
                    break

            if to_cache is not None:
                self._feature_cache[respondent] = CachedFeatures(to_cache, value_column=value_column)

            await self.send_data_response(
                event_queue,
                ctx,
                data={
                    "respondent": respondent,
                    "raw_rows": int(raw_df.shape[0]),
                    "clean_rows": int(clean_df.shape[0]) if clean_df is not None else 0,
                    "period_start": window_start.strftime("%Y-%m-%dT%H"),
                    "period_end": window_end.strftime("%Y-%m-%dT%H"),
                    "raw_preview": _df_to_records(raw_df, max_rows),
                    "clean_preview": _df_to_records(clean_df, max_rows),
                    "feature_preview": _df_to_records(features_preview, max_rows),
                },
                text=f"Fetched energy data for {respondent}.",
                final=False,
                state=TaskState.working,
            )

        await self.send_text_response(
            event_queue,
            ctx,
            text="Energy data ingestion complete.",
            state=TaskState.completed,
        )

    @BaseAgent.message_handler("energy.forecast", payload_model=ForecastEnergyPayload)
    async def handle_energy_forecast(self, ctx, event_queue, payload: ForecastEnergyPayload, message) -> None:
        cached = self._feature_cache.get(payload.respondent)
        if cached is None or cached.frame.empty:
            await self.send_error(
                event_queue,
                ctx,
                f"No cached dataset for {payload.respondent}; run energy.fetch first.",
            )
            return

        features = cached.frame
        value_column = cached.value_column
        latest_timestamp = features.index.max()
        if isinstance(latest_timestamp, pd.Timestamp):
            latest_ts_str = latest_timestamp.tz_localize(None).isoformat()
        else:
            latest_ts_str = str(latest_timestamp)

        risk_score = _calculate_simple_risk_score(features, value_column=value_column)
        forecast = [
            {
                "respondent": payload.respondent,
                "horizon_hour": hour + 1,
                "risk_score": risk_score,
                "generated_at": latest_ts_str,
            }
            for hour in range(payload.horizon_hours)
        ]

        await self.send_data_response(
            event_queue,
            ctx,
            data={"respondent": payload.respondent, "forecast": forecast},
            text=f"Generated {payload.horizon_hours}-hour forecast for {payload.respondent}.",
        )

    def _resolve_respondents(self, payload: FetchEnergyPayload) -> List[str]:
        if payload.respondents:
            return payload.respondents
        if payload.respondent:
            return [payload.respondent]
        return self.energy_config.default_balancing_authorities

    def _resolve_window(self, payload: FetchEnergyPayload) -> tuple[datetime, datetime]:
        end = payload.end_time or datetime.utcnow()
        start = payload.start_time or (end - timedelta(hours=self.energy_config.default_lookback_hours))
        return start, end


def _df_to_records(df: Optional[pd.DataFrame], limit: Optional[int]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    data = df.reset_index()
    if limit:
        data = data.head(limit)
    for column in data.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]):
        data[column] = data[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return data.to_dict(orient="records")


def _calculate_simple_risk_score(features: pd.DataFrame, value_column: str = "value") -> float:
    if features.empty:
        return 0.0
    diff_col = f"{value_column}_diff"
    latest_diff = float(abs(features[diff_col].iloc[-1])) if diff_col in features.columns else 0.0
    if value_column in features.columns:
        baseline = float(abs(features[value_column].iloc[-1])) or 1.0
    else:
        baseline = 1.0
    ratio = min(latest_diff / max(baseline, 1.0), 1.0)
    return round(ratio, 3)



