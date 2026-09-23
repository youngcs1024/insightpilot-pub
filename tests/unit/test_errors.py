"""Pure error contracts and agent failure vocabulary."""

import json
from uuid import uuid4

import pytest
from asgi_correlation_id import correlation_id
from pydantic import ValidationError as PydanticValidationError
from starlette.requests import Request

from app.agents.failures import FailureKind, NodeFailure
from app.api.exception_handlers import handle_known, handle_unknown
from app.core.errors import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    DatabaseError,
    DatabaseTimeoutError,
    DeadlineExceededError,
    HealthProbeError,
    HealthProbeTimeoutError,
    InsightPilotError,
    NotFoundError,
    ProviderProbeRetryableError,
    RateLimitError,
    RateLimitExceeded,
    UpstreamUnavailableError,
    ValidationError,
)


@pytest.mark.parametrize(
    ("error_type", "status", "code"),
    [
        (InsightPilotError, 500, "internal_error"),
        (ValidationError, 400, "VALIDATION_ERROR"),
        (AuthenticationError, 401, "AUTHENTICATION_ERROR"),
        (AuthorizationError, 403, "AUTHORIZATION_ERROR"),
        (NotFoundError, 404, "NOT_FOUND"),
        (ConflictError, 409, "CONFLICT"),
        (RateLimitError, 429, "RATE_LIMIT_EXCEEDED"),
        (UpstreamUnavailableError, 503, "UPSTREAM_UNAVAILABLE"),
        (DeadlineExceededError, 504, "DEADLINE_EXCEEDED"),
        (DatabaseError, 500, "DATABASE_ERROR"),
        (DatabaseTimeoutError, 503, "DATABASE_TIMEOUT"),
        (HealthProbeError, 503, "HEALTH_PROBE_UNAVAILABLE"),
        (HealthProbeTimeoutError, 503, "HEALTH_PROBE_TIMEOUT"),
        (ProviderProbeRetryableError, 503, "PROVIDER_PROBE_RETRYABLE"),
    ],
)
async def test_known_error_maps_to_status(
    error_type: type[InsightPilotError], status: int, code: str
) -> None:
    request_id = str(uuid4())
    token = correlation_id.set(request_id)
    try:
        response = await handle_known(
            Request({"type": "http"}), error_type("internal diagnostic", private="hidden")
        )
    finally:
        correlation_id.reset(token)
    assert response.status_code == status
    assert json.loads(response.body) == {
        "code": code,
        "message": error_type.user_message,
        "request_id": request_id,
    }


async def test_unknown_error_returns_generic_500() -> None:
    request_id = str(uuid4())
    previous = correlation_id.get()
    response = await handle_unknown(
        Request({"type": "http", "state": {"request_id": request_id}}),
        RuntimeError("private diagnostic"),
    )
    assert response.status_code == InsightPilotError.http_status
    assert json.loads(response.body) == {
        "code": "internal_error",
        "message": "An internal error occurred.",
        "request_id": request_id,
    }
    assert response.headers["X-Request-ID"] == request_id
    assert correlation_id.get() == previous


def test_internal_detail_and_context_are_separate_from_public_constants() -> None:
    error = ValidationError("private diagnostic", code="spoof", message="private")
    assert error.detail == str(error) == "private diagnostic"
    assert error.context == {"code": "spoof", "message": "private"}
    assert error.code == "VALIDATION_ERROR"
    assert error.user_message == "The request is invalid."
    assert str(ValidationError()) == ValidationError.user_message
    assert RateLimitExceeded is RateLimitError
    assert issubclass(ConflictError, DatabaseError)
    assert issubclass(UpstreamUnavailableError, DatabaseError)
    assert UpstreamUnavailableError.retryable
    assert not DatabaseTimeoutError.retryable
    assert not RateLimitError.retryable


@pytest.mark.parametrize("kind", list(FailureKind))
def test_node_failure_round_trip(kind: FailureKind) -> None:
    failure = NodeFailure(node="query", kind=kind, detail="internal", retryable=False)
    assert failure.model_dump(mode="json") == {
        "node": "query",
        "kind": kind.value,
        "detail": "internal",
        "retryable": False,
    }
    assert NodeFailure.model_validate_json(failure.model_dump_json()) == failure
    assert not isinstance(failure, InsightPilotError)


def test_failure_vocabulary_and_invalid_kind() -> None:
    assert {kind.value for kind in FailureKind} == {
        "client_disconnected",
        "interrupted",
        "context_budget_exceeded",
        "node_operation_failed",
        "route_low_confidence",
        "sql_generation_failed",
        "sql_validation_failed",
        "sql_execution_failed",
        "sql_timeout",
        "sql_correction_exhausted",
        "mcp_unavailable",
        "mcp_policy_rejected",
        "mcp_rate_limited",
        "retrieval_unavailable",
        "retrieval_no_evidence",
        "model_runtime_unavailable",
        "llm_structured_output_failed",
        "deadline_exceeded",
    }
    with pytest.raises(PydanticValidationError):
        NodeFailure.model_validate(
            {"node": "query", "kind": "failed after retry", "detail": "", "retryable": False}
        )
