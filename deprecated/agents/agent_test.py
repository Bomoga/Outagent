from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass 
from datetime import UTC, datetime, timedelta

from a2a.types import DataPart, Message, Part, Role

from backend.agents.energy_agent import (
    EnergyAgent,
    EnergyAgentConfig,
    FetchEnergyPayload,
    ForecastEnergyPayload,
)

class StubContext:
    def __init__(self, message: Message, task_id: str, context_id: str) -> None:
        self.message = message
        self.task_id = task_id
        self.context_id = context_id

    @property
    def metadata(self) -> dict[str,str]:
        return {}

class StubEventQueue:
    """Collects events that the agent would normally stream back through A2A."""
    def __init__(self) -> None:
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)

    async def close(self, *_, **__):
        return

def build_message(command: str, payload: dict, *, task_id: str, context_id: str) -> Message:
    """Create an A2A Message with a single JSON data part."""
    part = Part(root=DataPart(data={"type": command, "payload": payload}))
    return Message(
        message_id=str(uuid.uuid4()),
        role=Role.user,
        parts=[part],
        task_id=task_id,
        context_id=context_id,
    )

async def main() -> None:
    api_key = os.environ.get("EIA_API_KEY", "mq7cQLfepEbZ674BT2NOHHvhMs0pzbglrXM3Gdfn")

    agent = EnergyAgent(
        EnergyAgentConfig(
            agent_id="energy-agent",
            eia_api_key=api_key,
            default_balancing_authorities=["FPL"],
        )
    )

    task_id = "task-" + uuid.uuid4().hex[:6]
    context_id = "ctx-" + uuid.uuid4().hex[:6]

    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=6)

    fetch_payload = FetchEnergyPayload(
        respondent="FPL",
        start_time=start,
        end_time=end,
        metrics=["value"],
        page_size=2000,
    )

    fetch_message = build_message("energy.fetch", fetch_payload.model_dump(exclude_none=True), task_id=task_id, context_id=context_id)
    fetch_ctx = StubContext(fetch_message, task_id=task_id, context_id=context_id)
    queue = StubEventQueue()

    await agent.bootstrap(fetch_ctx)
    await agent.handle_fetch_energy(fetch_ctx, queue, fetch_payload, fetch_message)

    print("=== Events from energy.fetch ===")
    for event in queue.events:
        print(type(event).__name__, getattr(event, "task_id", None), getattr(event, "status", None))

    queue.events.clear()

    forecast_payload = ForecastEnergyPayload(respondent="FPL", horizon_hours=12)
    forecast_message = build_message("energy.forecast", forecast_payload.model_dump(), task_id=task_id, context_id=context_id)
    forecast_ctx = StubContext(forecast_message, task_id=task_id, context_id=context_id)

    await agent.handle_energy_forecast(forecast_ctx, queue, forecast_payload, forecast_message)

    print("\n=== Events from energy.forecast ===")
    for event in queue.events:
        print(type(event).__name__, getattr(event, "task_id", None))
        if hasattr(event, "parts"):
            for part in event.parts:
                root = part.root
                if hasattr(root, "data"):
                    forecast = root.data.get("forecast")
                    if forecast:
                        print("Forecast payload:", forecast)
                        print("Risk scores: ", [entry["risk_score"] for entry in forecast])

    await agent.shutdown(forecast_ctx)


if __name__ == "__main__":
    asyncio.run(main())
