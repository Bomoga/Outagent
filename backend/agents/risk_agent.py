from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState

from backend.agents.base_agent import AgentConfig, BaseAgent
from backend.agents.messages import EnergyHistoryReadyPayload
from backend.services import risk_assessment, weather_fetcher


class RiskAgentConfig(AgentConfig):
    """Configuration for the risk assessment agent."""

    model_dir: Path = Path("models/risk")


class RiskAgent(BaseAgent):
    """Combines energy history with weather data to train consumption forecasts."""

    def __init__(self, config: RiskAgentConfig) -> None:
        super().__init__(config)
        self.model_dir = config.model_dir

    @classmethod
    def get_agent_card(cls) -> AgentCard:
        return AgentCard(
            name="risk-agent",
            version="0.1.0",
            description="Aggregates multi-source data to assess outage risk and demand forecasts.",
            url="https://outagent.local/a2a/risk",
            default_input_modes=["application/json"],
            default_output_modes=["application/json"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(
                    id="risk.energy.train",
                    name="Train energy demand model",
                    description="Combine energy history with weather features to train an hourly demand model.",
                    tags=["risk", "energy", "forecast"],
                ),
            ],
        )

    async def bootstrap(self, ctx) -> None:
        self.logger.info("risk_agent_bootstrap")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("risk_agent_ready")

    async def shutdown(self, ctx) -> None:
        self.logger.info("risk_agent_shutdown")

    @BaseAgent.message_handler("energy.history.ready", payload_model=EnergyHistoryReadyPayload)
    async def handle_energy_history_ready(self, ctx, event_queue, payload: EnergyHistoryReadyPayload, message) -> None:
        self.logger.info(
            "energy_history_received",
            respondent=payload.respondent,
            cache_path=payload.cache_path,
            rows=payload.rows,
        )

        await self.send_status_update(
            event_queue,
            ctx,
            state=TaskState.working,
            status_message=f"Loading energy history from {payload.cache_path}",
        )

        energy_df = pd.read_csv(payload.cache_path, parse_dates=["period"])  # type: ignore[arg-type]
        energy_df = energy_df.set_index("period").sort_index()
        if getattr(energy_df.index, "tz", None) is None:
            energy_df.index = energy_df.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
        else:
            energy_df.index = energy_df.index.tz_convert("UTC")
        energy_df = energy_df.ffill().bfill()

        start, end = energy_df.index.min(), energy_df.index.max()
        weather_df = weather_fetcher.load_weather_features(start, end)
        dataset = energy_df.join(weather_df, how="left")
        dataset["hour"] = dataset.index.hour
        dataset["day_of_week"] = dataset.index.dayofweek
        dataset["month"] = dataset.index.month
        dataset = dataset.ffill().bfill()

        model_path = self.model_dir / f"energy_demand_{payload.respondent.lower()}.joblib"

        try:
            metrics = risk_assessment.train_hourly_consumption_model(
                dataset,
                target_column="value",
                model_path=model_path,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("model_training_failed", error=str(exc))
            await self.send_error(event_queue, ctx, f"Model training failed: {exc}")
            return

        await self.send_data_response(
            event_queue,
            ctx,
            data={
                "respondent": payload.respondent,
                "model_path": str(model_path),
                "metrics": metrics,
            },
            text="Risk model trained successfully.",
            state=TaskState.completed,
        )


