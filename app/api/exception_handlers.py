"""Central safe HTTP mapping; internal diagnostics are only sent to redacted logs."""

from http import HTTPStatus
from uuid import uuid4

import structlog
from asgi_correlation_id import correlation_id
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from app.core.errors import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    DeadlineExceededError,
    InsightPilotError,
    NotFoundError,
    PasswordPolicyError,
    QuotaExceededError,
    RateLimitError,
    UpstreamUnavailableError,
    ValidationError,
)

logger = structlog.get_logger(__name__)
_HTTP_ERRORS: dict[int, type[InsightPilotError]] = {
    error.http_status: error
    for error in (
        ValidationError,
        AuthenticationError,
        AuthorizationError,
        NotFoundError,
        ConflictError,
        RateLimitError,
        UpstreamUnavailableError,
        DeadlineExceededError,
    )
}
_HTTP_STATUSES = {status.value: status for status in HTTPStatus}
_PROTOCOL_HEADERS = frozenset({"allow", "www-authenticate", "retry-after"})


class ErrorResponse(BaseModel):
    """Stable public failure fields; internal exception details are never serialized."""

    code: str
    message: str
    request_id: str | None


def _response(status: int, code: str, message: str) -> JSONResponse:
    response = ErrorResponse(code=code, message=message, request_id=correlation_id.get())
    return JSONResponse(status_code=status, content=response.model_dump())


async def handle_known(request: Request, exc: Exception) -> JSONResponse:
    """Map any project failure without exposing its diagnostics or context."""
    if not isinstance(exc, InsightPilotError):
        raise exc
    error_type = type(exc)
    logger.warning("request_failed", code=error_type.code, detail=exc.detail, context=exc.context)
    message = error_type.user_message
    if isinstance(exc, PasswordPolicyError):
        message = "Password requires: " + "; ".join(exc.failures) + "."
    response = _response(error_type.http_status, error_type.code, message)
    if isinstance(exc, AuthenticationError):
        response.headers["WWW-Authenticate"] = "Bearer"
    if isinstance(exc, QuotaExceededError):
        response.headers["Retry-After"] = str(exc.retry_after)
    return response


async def handle_unknown(request: Request, exc: Exception) -> JSONResponse:
    """Restore edge context after middleware unwinds and render a generic 500."""
    request_id = getattr(request.state, "request_id", None) or correlation_id.get() or str(uuid4())
    token = correlation_id.set(request_id)
    try:
        logger.exception(
            "unhandled_exception", exc_info=(type(exc), exc, exc.__traceback__), detail=str(exc)
        )
        response = _response(
            InsightPilotError.http_status, InsightPilotError.code, InsightPilotError.user_message
        )
        # ServerErrorMiddleware responds outside the request-ID middleware.
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        correlation_id.reset(token)


async def handle_request_validation(request: Request, exc: Exception) -> JSONResponse:
    """Keep FastAPI's 422 status without echoing user inputs or validator prose."""
    if not isinstance(exc, RequestValidationError):
        raise exc
    logger.warning("request_failed", code="REQUEST_VALIDATION_ERROR")
    return _response(422, "REQUEST_VALIDATION_ERROR", "The request input is invalid.")


async def handle_http(request: Request, exc: Exception) -> JSONResponse:
    """Normalize framework errors while retaining required protocol headers."""
    if not isinstance(exc, HTTPException):
        raise exc
    error_type = _HTTP_ERRORS.get(exc.status_code)
    if error_type is not None:
        code, message = error_type.code, error_type.user_message
    elif exc.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        code, message = InsightPilotError.code, InsightPilotError.user_message
    else:
        status = _HTTP_STATUSES.get(exc.status_code)
        code = status.name if status is not None else "HTTP_ERROR"
        message = status.phrase if status is not None else "The request could not be completed."
    logger.warning("request_failed", code=code, status_code=exc.status_code)
    response = _response(exc.status_code, code, message)
    for name, value in (exc.headers or {}).items():
        if name.lower() in _PROTOCOL_HEADERS:
            response.headers[name] = value
    return response
