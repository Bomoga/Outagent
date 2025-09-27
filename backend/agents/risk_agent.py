from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel

from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState

from backend.agents.base_agent import AgentConfig, BaseAgent
from backend.services import risk_assessment


class RiskRequestPayload(BaseModel):
    weather_context: Dict[str, Any]
    flood_context: Dict[str, Any]
    asset_points: List[Dict[str, Any]]


class RiskAgentConfig(AgentConfig):
    pass


class RiskAgent(BaseAgent):
    @classmethod
    def get_agent_card(cls) -> AgentCard:
        return AgentCard(
            name="risk-agent",
            version="0.1.0",
            description="Aggregate weather + flood signals into per-asset risk scores.",
            url="https://outagent.local/a2a/risk",
            default_input_modes=["application/json"],
            default_output_modes=["application/json"],
            capabilities=AgentCapabilities(streaming=False),
            skills=[
                AgentSkill(
                    id="risk.assess",
                    name="Assess risk",
                    description="Compute per-asset risk scores from WeatherContext and FloodContext",
                    tags=["risk", "aggregation"],
                )
            ],
        )

    async def bootstrap(self, ctx) -> None:
        self.logger.info("risk_agent_bootstrap")

    async def shutdown(self, ctx) -> None:
        self.logger.info("risk_agent_shutdown")

    @BaseAgent.message_handler("risk.assess", payload_model=RiskRequestPayload)
    async def handle_risk_assess(self, ctx, event_queue, payload: RiskRequestPayload, message) -> None:
        await self.send_status_update(
            event_queue,
            ctx,
            state=TaskState.working,
            status_message=f"Assessing risk for {len(payload.asset_points)} assets",
        )

        try:
            results = risk_assessment.assess_assets(
                weather_context=payload.weather_context,
                flood_context=payload.flood_context,
                assets=payload.asset_points,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("risk_assessment_failed", error=str(exc))
            await self.send_error(event_queue, ctx, f"Risk assessment failed: {exc}")
            return

        await self.send_data_response(
            event_queue,
            ctx,
            data={"risk_results": results},
            text=f"Computed risk for {len(results)} assets",
        )
