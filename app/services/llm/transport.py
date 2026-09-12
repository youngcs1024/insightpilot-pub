"""One retry owner around each asynchronous provider HTTP operation."""

from http import HTTPStatus

import httpx
import structlog
from opentelemetry import trace
from pydantic import ValidationError

from app.core.deadline import Deadline
from app.core.errors import (
    LlmCapabilityError,
    LlmRateLimitError,
    LlmRequestError,
    LlmResponseError,
    LlmUnavailableError,
)
from app.core.observability import TraceMetadata, current_role, observe
from app.core.retry import run_operation
from app.services.llm.contracts import Completion, CompletionRequest, ErrorResponse
from app.services.llm.registry import StructuredTier

logger = structlog.get_logger(__name__)
_TEMPORARY = {500, 502, 503, 504}
_CAPABILITY_CODES = {"unsupported_parameter", "unsupported_value", "unsupported_response_format"}


class LlmTransport:
    """The service owns the injected client and closes it with the application."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def complete(
        self,
        request: CompletionRequest,
        *,
        deadline: Deadline,
        timeout_s: float,
        tier: StructuredTier | None,
    ) -> Completion:
        """Retry transport/temporary failures, never schema or generic 4xx failures."""
        attempts = 0

        async def operation() -> Completion:
            nonlocal attempts
            attempts += 1
            with observe(
                "llm_completion",
                TraceMetadata(
                    model=request.model,
                    role=current_role(),
                    attempt=attempts,
                    structured_tier=int(tier) if tier is not None else None,
                    cost_status="inferred_by_langfuse_or_unknown",
                ),
                generation=True,
            ) as observation:
                try:
                    response = await self.client.post(
                        "chat/completions",
                        json=request.model_dump(mode="json", exclude_none=True),
                        timeout=deadline.budget(timeout_s),
                    )
                except httpx.TransportError:
                    raise LlmUnavailableError() from None
                _check_status(response, tier)
                try:
                    result = Completion.model_validate_json(response.content)
                except ValidationError:
                    raise LlmResponseError() from None
                _record(request.model, tier, attempts, result)
                if observation is not None and result.usage is not None:
                    observation.usage(result.usage.prompt_tokens, result.usage.completion_tokens)
                return result

        return await run_operation(
            operation, deadline=deadline, timeout_s=timeout_s, name="llm_completion", attempts=3
        )


def _check_status(response: httpx.Response, tier: StructuredTier | None) -> None:
    if response.is_success:
        return
    if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        raise LlmRateLimitError()
    if response.status_code in _TEMPORARY:
        raise LlmUnavailableError()
    if response.status_code == HTTPStatus.BAD_REQUEST and tier is not None:
        _check_capability(response, tier)
    raise LlmRequestError()


def _check_capability(response: httpx.Response, tier: StructuredTier) -> None:
    try:
        error = ErrorResponse.model_validate_json(response.content).error
    except ValidationError:
        return
    params = {
        StructuredTier.NATIVE: {
            "response_format",
            "response_format.type",
            "response_format.json_schema",
        },
        StructuredTier.TOOL: {"tools", "tool_choice", "parallel_tool_calls"},
        StructuredTier.PROMPTED: set(),
    }
    if error.code in _CAPABILITY_CODES and error.param in params[tier]:
        raise LlmCapabilityError()


def _record(model: str, tier: StructuredTier | None, attempts: int, result: Completion) -> None:
    # Only bounded metadata reaches tracing; no input/output or exception object is exported.
    logger.info(
        "llm_completion",
        model=model,
        structured_tier=tier,
        attempts=attempts,
        prompt_tokens=result.usage.prompt_tokens if result.usage else None,
        completion_tokens=result.usage.completion_tokens if result.usage else None,
    )
    try:
        span = trace.get_current_span()
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.attempts", attempts)
        if tier is not None:
            span.set_attribute("llm.structured_tier", int(tier))
        if result.usage:
            span.set_attribute("llm.prompt_tokens", result.usage.prompt_tokens)
            span.set_attribute("llm.completion_tokens", result.usage.completion_tokens)
    except Exception:
        # An exporter failure must not turn a valid completion into a failed request.
        logger.exception("llm_trace_unavailable", exc_info=False)
