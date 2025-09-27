from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from a2a.types import DataPart, Message, Part, Role

from backend.agents.energy_agent import (
    EnergyAgent,
    EnergyAgentConfig,
    FetchEnergyPayload,
    FetchEnergyHistoryPayload,
    ForecastEnergyPayload,
)
from backend.agents.messages import EnergyHistoryReadyPayload
from backend.agents.risk_agent import RiskAgent, RiskAgentConfig


@dataclass
class StubContext:
    message: Message
    task_id: str
    context_id: str
    sent_messages: list = field(default_factory=list)

    @property
    def metadata(self) -> dict[str, str]:
        return {}

    def send(
        self,
        *,
        target_agent: str,
        message: Message,
        correlation_id: str | None = None,
    ) -> None:
        self.sent_messages.append((target_agent, message, correlation_id))


class StubEventQueue:
    """Collect events emitted by agents during tests."""

    def __init__(self) -> None:
        self.events: list = []

    async def enqueue_event(self, event):
        self.events.append(event)

    async def close(self, *_, **__):
        return


def build_message(command: str, payload: dict, *, task_id: str, context_id: str) -> Message:
    """Construct an A2A message carrying a JSON data payload."""
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

    energy_agent = EnergyAgent(
        EnergyAgentConfig(
            agent_id="energy-agent",
            eia_api_key=api_key,
            default_balancing_authorities=["FPL"],
        )
    )
    risk_agent = RiskAgent(
        RiskAgentConfig(
            agent_id="risk-agent",
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

    fetch_message = build_message(
        "energy.fetch",
        fetch_payload.model_dump(exclude_none=True),
        task_id=task_id,
        context_id=context_id,
    )
    fetch_ctx = StubContext(fetch_message, task_id=task_id, context_id=context_id)
    energy_queue = StubEventQueue()

    bootstrap_message = build_message("risk.bootstrap", {}, task_id=task_id, context_id=context_id)
    risk_boot_ctx = StubContext(bootstrap_message, task_id=task_id, context_id=context_id)
    risk_queue = StubEventQueue()

    await energy_agent.bootstrap(fetch_ctx)
    await risk_agent.bootstrap(risk_boot_ctx)

    await energy_agent.handle_fetch_energy(fetch_ctx, energy_queue, fetch_payload, fetch_message)

    print("=== Events from energy.fetch ===")
    for event in energy_queue.events:
        print(type(event).__name__, getattr(event, "task_id", None), getattr(event, "status", None))

    energy_queue.events.clear()

    forecast_payload = ForecastEnergyPayload(respondent="FPL", horizon_hours=12)
    forecast_message = build_message(
        "energy.forecast",
        forecast_payload.model_dump(exclude_none=True),
        task_id=task_id,
        context_id=context_id,
    )
    forecast_ctx = StubContext(forecast_message, task_id=task_id, context_id=context_id)

    await energy_agent.handle_energy_forecast(forecast_ctx, energy_queue, forecast_payload, forecast_message)

    print("\n=== Events from energy.forecast ===")
    for event in energy_queue.events:
        print(type(event).__name__, getattr(event, "task_id", None))
        if hasattr(event, "parts"):
            for part in event.parts:
                root = part.root
                if hasattr(root, "data"):
                    forecast = root.data.get("forecast")
                    if forecast:
                        print("Forecast payload:", forecast)
                        print("Risk scores:", [entry["risk_score"] for entry in forecast])

    energy_queue.events.clear()

    history_payload = FetchEnergyHistoryPayload(respondent="FPL")
    history_message = build_message(
        "energy.fetch_history",
        history_payload.model_dump(exclude_none=True),
        task_id=task_id,
        context_id=context_id,
    )
    history_ctx = StubContext(history_message, task_id=task_id, context_id=context_id)

    await energy_agent.handle_fetch_energy_history(history_ctx, energy_queue, history_payload, history_message)

    print("\n=== Events from energy.fetch_history ===")
    history_summary: EnergyHistoryReadyPayload | None = None
    for event in energy_queue.events:
        print(type(event).__name__, getattr(event, "task_id", None))
        if hasattr(event, "parts"):
            for part in event.parts:
                root = part.root
                if hasattr(root, "data"):
                    print("History payload:", root.data)
                    if isinstance(root.data, dict) and {"respondent", "cache_path"}.issubset(root.data.keys()):
                        history_summary = EnergyHistoryReadyPayload.model_validate(root.data)

    energy_queue.events.clear()

    # Process outgoing messages to the risk agent
    risk_train_ctx: StubContext | None = None
    for target_agent, message, _ in history_ctx.sent_messages:
        if target_agent != "risk-agent":
            continue
        payload_dict = None
        for part in message.parts:
            data = getattr(part.root, "data", None)
            if isinstance(data, dict) and data.get("type") == "energy.history.ready":
                payload_dict = data.get("payload", {})
                break
        if payload_dict is None:
            continue
        summary = EnergyHistoryReadyPayload.model_validate(payload_dict)
        risk_train_ctx = StubContext(message, task_id=message.task_id, context_id=message.context_id)
        await risk_agent.handle_energy_history_ready(risk_train_ctx, risk_queue, summary, message)

    if history_summary is None:
        print("No history summary produced; aborting risk agent test.")
    else:
        print("\n=== Events from risk.energy.train ===")
        for event in risk_queue.events:
            print(type(event).__name__, getattr(event, "task_id", None), getattr(event, "status", None))
            if hasattr(event, "parts"):
                for part in event.parts:
                    root = part.root
                    if hasattr(root, "data"):
                        print("Risk payload:", root.data)

    await energy_agent.shutdown(history_ctx)
    if risk_train_ctx is not None:
        await risk_agent.shutdown(risk_train_ctx)
    else:
        await risk_agent.shutdown(risk_boot_ctx)


if __name__ == "__main__":
    asyncio.run(main())

