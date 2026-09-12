"""Safe failures through the real middleware stack, using no external services."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from http import HTTPStatus
from io import StringIO
from uuid import UUID, uuid4

import httpx
import pytest
import structlog
from asgi_correlation_id import correlation_id
from fastapi import FastAPI
from pydantic import BaseModel, Field, SecretStr
from starlette.exceptions import HTTPException

from app.application import create_app
from app.core.config_models import Settings
from app.core.errors import ValidationError
from app.services.health import HealthService
from tests.fakes.health_probe import FakeProbe

UUID_VERSION = 4


class Input(BaseModel):
    """Bounded synthetic request for framework validation."""

    count: int = Field(ge=1, le=10)


@pytest.fixture
def error_app(settings: Settings) -> FastAPI:
    app = create_app(
        settings, health_service=HealthService(FakeProbe(), FakeProbe(), settings.health)
    )

    @app.get("/known")
    async def known() -> None:
        raise ValidationError(
            "ordinary diagnostic password=hunter2",
            event="spoofed",
            code="spoofed",
            request_id="spoofed",
            nested={"password": "nested-private", "value": SecretStr("wrapped-private")},
            configured=settings.database.app_password.get_secret_value(),
        )

    @app.get("/unknown")
    async def unknown() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("ordinary diagnostic password=hunter2")

    @app.post("/validated")
    async def validated(body: Input) -> Input:
        return body

    @app.get("/http/{status}")
    async def framework(status: int) -> None:
        raise HTTPException(
            status,
            detail="password=hunter2",
            headers={
                "Allow": "GET",
                "WWW-Authenticate": "Bearer",
                "Retry-After": "60",
                "X-Internal": "private",
                "X-Request-ID": "spoofed",
            },
        )

    return app


@pytest.fixture
def error_logs() -> StringIO:
    """Retain the production formatter output across pytest capture phases."""
    return StringIO()


@pytest.fixture
async def error_client(
    error_app: FastAPI, error_logs: StringIO
) -> AsyncIterator[httpx.AsyncClient]:
    async with (
        error_app.router.lifespan_context(error_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(error_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client,
    ):
        handler = logging.getLogger().handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        handler.setStream(error_logs)
        yield client


@pytest.mark.parametrize("path", ["/known", "/unknown"])
async def test_internal_detail_never_in_response_body(
    error_client: httpx.AsyncClient,
    settings: Settings,
    error_logs: StringIO,
    path: str,
) -> None:
    request_id = str(uuid4())
    response = await error_client.get(path, headers={"X-Request-ID": request_id})
    output = error_logs.getvalue()
    for private in (
        "hunter2",
        "nested-private",
        "wrapped-private",
        settings.database.app_password.get_secret_value(),
    ):
        assert private not in response.text
        assert private not in output
    assert "ordinary diagnostic" not in response.text
    assert "ordinary diagnostic" in output
    event = "request_failed" if path == "/known" else "unhandled_exception"
    records = [json.loads(line) for line in output.splitlines()]
    failure = next(record for record in records if record["event"] == event)
    assert failure["request_id"] == UUID(request_id).hex
    if path == "/known":
        assert failure["code"] == "VALIDATION_ERROR"
        assert failure["context"]["code"] == "spoofed"
    else:
        assert failure["exception_type"] == "RuntimeError"


@pytest.mark.parametrize("path", ["/known", "/unknown", "/missing", "/http/405"])
@pytest.mark.parametrize("incoming", ["valid", "missing", "invalid"])
async def test_request_id_in_every_error_body(
    error_client: httpx.AsyncClient, path: str, incoming: str
) -> None:
    request_id = str(uuid4()) if incoming == "valid" else "invalid-id"
    headers = {} if incoming == "missing" else {"X-Request-ID": request_id}
    response = await error_client.get(path, headers=headers)
    body = response.json()
    assert set(body) == {"code", "message", "request_id"}
    assert UUID(body["request_id"]).version == UUID_VERSION
    assert response.headers.get_list("X-Request-ID") == [body["request_id"]]
    if incoming == "valid":
        assert body["request_id"] == UUID(request_id).hex
    assert correlation_id.get() is None
    assert structlog.contextvars.get_contextvars() == {}


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "VALIDATION_ERROR"),
        (401, "AUTHENTICATION_ERROR"),
        (403, "AUTHORIZATION_ERROR"),
        (404, "NOT_FOUND"),
        (405, "METHOD_NOT_ALLOWED"),
        (409, "CONFLICT"),
        (429, "RATE_LIMIT_EXCEEDED"),
        (500, "internal_error"),
        (502, "internal_error"),
        (503, "UPSTREAM_UNAVAILABLE"),
        (504, "DEADLINE_EXCEEDED"),
        (499, "HTTP_ERROR"),
    ],
)
async def test_framework_errors_are_safe_and_keep_protocol_headers(
    error_client: httpx.AsyncClient, status: int, code: str
) -> None:
    response = await error_client.get(f"/http/{status}")
    assert response.status_code == status
    assert response.json()["code"] == code
    assert "hunter2" not in response.text
    assert response.headers["Allow"] == "GET"
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.headers["Retry-After"] == "60"
    assert "X-Internal" not in response.headers


@pytest.mark.parametrize("body", ['{"count":"password=hunter2"}', '{"count":100}', "{invalid"])
async def test_request_validation_keeps_422_without_echoing_input(
    error_client: httpx.AsyncClient, body: str
) -> None:
    response = await error_client.post(
        "/validated", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json() == {
        "code": "REQUEST_VALIDATION_ERROR",
        "message": "The request input is invalid.",
        "request_id": response.headers["X-Request-ID"],
    }


async def test_router_404_and_405(error_client: httpx.AsyncClient) -> None:
    missing = await error_client.get("/missing")
    wrong_method = await error_client.get("/validated")
    assert missing.status_code == HTTPStatus.NOT_FOUND
    assert missing.json()["code"] == "NOT_FOUND"
    assert wrong_method.status_code == HTTPStatus.METHOD_NOT_ALLOWED
    assert wrong_method.json()["code"] == "METHOD_NOT_ALLOWED"
    assert wrong_method.headers["Allow"] == "POST"


async def test_unknown_errors_preserve_concurrent_ids_and_next_request(
    error_client: httpx.AsyncClient, error_logs: StringIO
) -> None:
    ids = [str(uuid4()) for _ in range(8)]
    responses = await asyncio.gather(
        *(error_client.get("/unknown", headers={"X-Request-ID": request_id}) for request_id in ids)
    )
    for response, request_id in zip(responses, ids, strict=True):
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert (
            response.json()["request_id"]
            == response.headers["X-Request-ID"]
            == UUID(request_id).hex
        )
    records = [json.loads(line) for line in error_logs.getvalue().splitlines()]
    assert {r["request_id"] for r in records if r["event"] == "unhandled_exception"} == {
        UUID(value).hex for value in ids
    }
    healthy = await error_client.get("/health")
    assert healthy.status_code == HTTPStatus.OK
    assert healthy.json()["request_id"] not in ids
    assert correlation_id.get() is None
