"""Root-level liveness and readiness endpoints with typed response bodies."""

from typing import Literal, cast

from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from app import __version__
from app.services.health import HealthChecks, HealthService

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Process liveness independent of external dependencies."""

    status: Literal["ok"] = "ok"
    version: str = __version__
    request_id: str = Field(min_length=1, max_length=64)


class ReadyResponse(BaseModel):
    """Dependency availability without upstream exception details."""

    status: Literal["ready", "degraded"]
    checks: HealthChecks
    request_id: str = Field(min_length=1, max_length=64)


@router.get("/health")
async def health() -> HealthResponse:
    """Report a running process even when every dependency is unavailable."""
    return HealthResponse(request_id=correlation_id.get() or "unavailable")


@router.get("/ready", response_model_exclude_none=True, responses={503: {"model": ReadyResponse}})
async def ready(request: Request, response: Response) -> ReadyResponse:
    """Check dependencies concurrently through the injected lifecycle service."""
    service = cast("HealthService", request.app.state.health)
    checks = await service.check()
    request.app.state.ready = checks.ready
    response.status_code = 200 if checks.ready else 503
    return ReadyResponse(
        status="ready" if checks.ready else "degraded",
        checks=checks,
        request_id=correlation_id.get() or "unavailable",
    )
