"""The sole application entry point for role-based model generation."""

import asyncio
from dataclasses import dataclass
from typing import overload

import httpx
import structlog
from langchain_core.messages import AIMessage, BaseMessage
from opentelemetry import trace
from pydantic import BaseModel

from app.core.config_models import LLMSettings
from app.core.deadline import Deadline
from app.core.errors import LlmConfigurationError, LlmResponseError, UpstreamUnavailableError
from app.core.llm_config import ModelRole
from app.core.observability import mark_degraded, model_role
from app.services.llm.contracts import CompletionRequest, GenerationBudget, ToolCall, ToolDefinition
from app.services.llm.messages import to_wire
from app.services.llm.registry import ModelRegistry, StructuredTier
from app.services.llm.structured import generate_structured
from app.services.llm.transport import LlmTransport

logger = structlog.get_logger(__name__)


@dataclass
class _FallbackState:
    """One call's forward-only model position; never stored on the shared service."""

    index: int = 0


class LlmService:
    """Application-owned HTTP resources with injectable offline dependencies."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        registry: ModelRegistry | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings.model_copy(deep=True)
        self._registry = registry
        self._client = client

    async def start(self) -> None:
        """Validate evidence before creating the client; no provider request is made."""
        if self._registry is None:
            self._registry = await asyncio.to_thread(ModelRegistry.load, self._settings)
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=str(self._settings.base_url).rstrip("/") + "/",
                headers={"Authorization": f"Bearer {self._settings.api_key.get_secret_value()}"},
                timeout=self._settings.timeout_s,
                follow_redirects=False,
            )

    async def aclose(self) -> None:
        """Release pooled HTTP connections, including when application startup fails."""
        if self._client is not None:
            await self._client.aclose()

    @overload
    async def call(
        self,
        role: ModelRole,
        messages: list[BaseMessage],
        *,
        deadline: Deadline,
        response_format: None = None,
    ) -> BaseMessage: ...

    @overload
    async def call[T: BaseModel](
        self,
        role: ModelRole,
        messages: list[BaseMessage],
        *,
        deadline: Deadline,
        response_format: type[T],
    ) -> T: ...

    async def call(
        self,
        role: ModelRole,
        messages: list[BaseMessage],
        *,
        deadline: Deadline,
        response_format: type[BaseModel] | None = None,
    ) -> BaseMessage | BaseModel:
        """Generate with independent fallback state and one immutable request deadline."""
        if self._registry is None or self._client is None:
            raise LlmConfigurationError()
        config = self._registry.role(role)
        if config.model is None or config.timeout_s is None:
            raise LlmConfigurationError()
        chain = [config.model, *config.fallback_models]
        state = _FallbackState()
        while True:
            deadline.check("llm_call")
            model = chain[state.index]
            request = CompletionRequest(
                model=model,
                messages=to_wire(messages),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            logger.info("llm_call_started", role=role, model=model, fallback_index=state.index)
            _record_role(role)
            try:
                with model_role(role.value):
                    return await self._generate(
                        request, response_format, deadline, config.timeout_s
                    )
            except UpstreamUnavailableError:
                deadline.check("llm_fallback")
                state.index += 1
                if state.index >= len(chain):
                    raise
                mark_degraded("llm")
                logger.info(
                    "llm_model_fallback", role=role, model=model, fallback_index=state.index
                )

    async def generate_structured[T: BaseModel](
        self,
        role: ModelRole,
        messages: list[BaseMessage],
        schema: type[T],
        *,
        deadline: Deadline,
    ) -> T:
        """Return the requested schema type through the common structured ladder."""
        return await self.call(role, messages, response_format=schema, deadline=deadline)

    async def call_with_tools(
        self,
        role: ModelRole,
        messages: list[BaseMessage],
        tools: list[ToolDefinition],
        *,
        deadline: Deadline,
    ) -> list[ToolCall]:
        """Request at most one native call; callers validate and dispatch it locally."""
        if self._registry is None or self._client is None or not tools:
            raise LlmConfigurationError()
        config = self._registry.role(role)
        if config.model is None or config.timeout_s is None:
            raise LlmConfigurationError()
        chain = [config.model, *config.fallback_models]
        for index, model in enumerate(chain):
            deadline.check("llm_native_tools")
            request = CompletionRequest(
                model=model,
                messages=to_wire(messages),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        },
                    }
                    for tool in tools
                ],
                tool_choice="auto",
                parallel_tool_calls=False,
            )
            logger.info("llm_native_tools_started", role=role, fallback_index=index)
            try:
                with model_role(role.value):
                    completion = await LlmTransport(self._client).complete(
                        request,
                        deadline=deadline,
                        timeout_s=config.timeout_s,
                        tier=StructuredTier.TOOL,
                    )
            except UpstreamUnavailableError:
                deadline.check("llm_native_tools_fallback")
                if index + 1 >= len(chain):
                    raise
                mark_degraded("llm")
                continue
            choice = completion.choices[0]
            calls = choice.message.tool_calls or []
            if choice.message.role != "assistant":
                raise LlmResponseError()
            if choice.finish_reason == "stop" and not calls:
                return []
            if (
                choice.finish_reason != "tool_calls"
                or len(calls) != 1
                or not calls[0].id
                or not calls[0].function.name
            ):
                raise LlmResponseError()
            return calls
        raise LlmConfigurationError()

    async def _generate(
        self,
        request: CompletionRequest,
        schema: type[BaseModel] | None,
        deadline: Deadline,
        timeout_s: float,
    ) -> BaseModel | BaseMessage:
        if self._registry is None or self._client is None:
            raise LlmConfigurationError()
        transport = LlmTransport(self._client)
        if schema is not None:
            return await generate_structured(
                transport,
                request,
                schema,
                budget=GenerationBudget(deadline=deadline, timeout_s=timeout_s),
                starting_tier=self._registry.tier(request.model),
            )
        completion = await transport.complete(
            request, deadline=deadline, timeout_s=timeout_s, tier=None
        )
        choice = completion.choices[0]
        if (
            choice.finish_reason != "stop"
            or choice.message.role != "assistant"
            or choice.message.content is None
            or choice.message.tool_calls
        ):
            raise LlmResponseError()
        return AIMessage(content=choice.message.content)


def _record_role(role: ModelRole) -> None:
    try:
        trace.get_current_span().set_attribute("llm.role", role.value)
    except Exception:
        logger.exception("llm_trace_unavailable", exc_info=False)
