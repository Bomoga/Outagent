from __future__ import annotations

import abc
import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, Iterator, Optional, Type

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    AgentCard,
    DataPart,
    Message,
    Part,
    Role,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

HandlerMethod = Callable[["BaseAgent", RequestContext, EventQueue, Any, Message], Awaitable[None]]


@dataclass(slots=True)
class _HandlerMetadata:
    func: HandlerMethod
    payload_model: Optional[Type[BaseModel]] = None


class AgentConfig(BaseModel):
    agent_id: str
    polling_interval_seconds: float = 30.0
    feature_store_path: str = "data/features"
    http_timeout_seconds: float = 20.0


class _MessageHandlerRegistry:
    def __init__(self) -> None:
        self._registry: Dict[Type["BaseAgent"], Dict[str, _HandlerMetadata]] = {}

    def register(self, owner: Type["BaseAgent"], command: str, metadata: _HandlerMetadata) -> None:
        self._registry.setdefault(owner, {})[command] = metadata

    def iter_handlers(self, owner: Type["BaseAgent"]) -> Iterable[tuple[str, _HandlerMetadata]]:
        for cls in owner.mro():
            if not issubclass(cls, BaseAgent):
                continue
            mapping = self._registry.get(cls)
            if mapping:
                yield from mapping.items()


_HANDLER_REGISTRY = _MessageHandlerRegistry()


@dataclass(slots=True)
class _BoundHandler:
    func: Callable[[RequestContext, EventQueue, Any, Message], Awaitable[None]]
    payload_model: Optional[Type[BaseModel]] = None


class BaseAgent(AgentExecutor, abc.ABC):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__()
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__, agent_id=config.agent_id)
        self._http_timeout = httpx.Timeout(config.http_timeout_seconds)
        self._http_client: Optional[httpx.AsyncClient] = None
        self._handlers: Dict[str, _BoundHandler] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._install_handlers()

    @property
    def agent_name(self) -> str:
        return self.get_agent_card().name

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._http_timeout)
        return self._http_client

    @classmethod
    @abc.abstractmethod
    def get_agent_card(cls) -> AgentCard:
        raise NotImplementedError

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            message = context.message
            if message is None:
                raise ValueError("Request did not include a message payload.")
            command, raw_payload = self._extract_command(message)
            handler = self._handlers.get(command)
            if handler is None:
                self.logger.warning("unhandled_command", command=command)
                await self.send_error(event_queue, context, f"Unsupported command '{command}'.")
                return
            payload = raw_payload
            if handler.payload_model is not None:
                try:
                    payload = handler.payload_model.model_validate(raw_payload)
                except ValidationError as exc:
                    self.logger.warning("invalid_payload", command=command, errors=exc.errors())
                    await self.send_error(event_queue, context, f"Invalid payload for '{command}'.")
                    return
            await handler.func(context, event_queue, payload, message)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("agent_execute_error", error=str(exc))
            await self.send_error(event_queue, context, "Execution failure.")
        finally:
            await self._cancel_background_tasks()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        self.logger.info("cancel_requested")
        await self._cancel_background_tasks()
        await self._send_status(
            event_queue,
            context,
            state=TaskState.canceled,
            final=True,
            status_message="Task cancelled by request.",
        )
        await self._close_http_client()

    def schedule_background_task(self, coro: Awaitable[Any]) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def bootstrap(self, ctx: RequestContext) -> None:
        """Override to perform agent-specific startup work."""

    async def shutdown(self, ctx: RequestContext) -> None:
        """Override to release agent-specific resources."""

    async def tick(self, ctx: RequestContext) -> None:
        """Override for periodic activities (optional)."""

    async def send_text_response(
        self,
        event_queue: EventQueue,
        context: RequestContext,
        text: str,
        *,
        final: bool = True,
        state: TaskState = TaskState.completed,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        message = self._build_message(context, text=text, metadata=metadata)
        await event_queue.enqueue_event(message)
        await self._send_status(event_queue, context, state=state, final=final)

    async def send_data_response(
        self,
        event_queue: EventQueue,
        context: RequestContext,
        data: dict[str, Any],
        *,
        text: Optional[str] = None,
        final: bool = True,
        state: TaskState = TaskState.completed,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        message = self._build_message(context, text=text, data=data, metadata=metadata)
        await event_queue.enqueue_event(message)
        await self._send_status(event_queue, context, state=state, final=final)

    async def send_status_update(
        self,
        event_queue: EventQueue,
        context: RequestContext,
        *,
        state: TaskState,
        final: bool = False,
        status_message: Optional[str] = None,
    ) -> None:
        await self._send_status(
            event_queue,
            context,
            state=state,
            final=final,
            status_message=status_message,
        )

    async def send_error(
        self,
        event_queue: EventQueue,
        context: RequestContext,
        detail: str,
    ) -> None:
        message = self._build_message(context, text=detail)
        await event_queue.enqueue_event(message)
        await self._send_status(
            event_queue,
            context,
            state=TaskState.failed,
            final=True,
            status_message=detail,
        )

    def _build_message(
        self,
        context: RequestContext,
        *,
        text: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        parts: list[Part] = []
        if text is not None:
            parts.append(Part(root=TextPart(text=text)))
        if data is not None:
            parts.append(Part(root=DataPart(data=data)))
        if not parts:
            raise ValueError("At least one of text or data must be provided.")
        return Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=parts,
            task_id=self._resolve_task_id(context),
            context_id=self._resolve_context_id(context),
            metadata=metadata,
        )

    async def _send_status(
        self,
        event_queue: EventQueue,
        context: RequestContext,
        *,
        state: TaskState,
        final: bool,
        status_message: Optional[str] = None,
    ) -> None:
        message_obj = None
        if status_message:
            message_obj = self._build_message(context, text=status_message)
        status = TaskStatus(state=state, message=message_obj)
        event = TaskStatusUpdateEvent(
            context_id=self._resolve_context_id(context),
            task_id=self._resolve_task_id(context),
            status=status,
            final=final,
        )
        await event_queue.enqueue_event(event)

    def _extract_command(self, message: Message) -> tuple[str, Any]:
        for part in message.parts:
            content = part.root
            if isinstance(content, DataPart):
                data = content.data
                if isinstance(data, dict):
                    command = data.get("type") or data.get("command")
                    payload = data.get("payload", {})
                    if command:
                        return command, payload
        raise ValueError("Message does not contain a command data part.")

    def _install_handlers(self) -> None:
        for command, metadata in _HANDLER_REGISTRY.iter_handlers(type(self)):
            bound = metadata.func.__get__(self, type(self))
            self._handlers[command] = _BoundHandler(bound, metadata.payload_model)

    async def _close_http_client(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def _cancel_background_tasks(self) -> None:
        if not self._background_tasks:
            return
        for task in list(self._background_tasks):
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

    def _resolve_task_id(self, context: RequestContext) -> str:
        message = context.message
        if context.task_id:
            return context.task_id
        if message and message.task_id:
            return message.task_id
        raise ValueError("Request context is missing a task_id.")

    def _resolve_context_id(self, context: RequestContext) -> str:
        message = context.message
        if context.context_id:
            return context.context_id
        if message and message.context_id:
            return message.context_id
        raise ValueError("Request context is missing a context_id.")

    @classmethod
    def message_handler(
        cls,
        command: str,
        *,
        payload_model: Optional[Type[BaseModel]] = None,
    ) -> Callable[[HandlerMethod], HandlerMethod]:
        def decorator(func: HandlerMethod) -> HandlerMethod:
            _HANDLER_REGISTRY.register(cls, command, _HandlerMetadata(func, payload_model))
            return func

        return decorator

    @contextlib.contextmanager
    def log_context(self, **kwargs: Any) -> Iterator[None]:
        structlog.contextvars.bind_contextvars(**kwargs)
        try:
            yield
        finally:
            structlog.contextvars.unbind_contextvars(*kwargs.keys())
