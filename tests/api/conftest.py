"""Offline API settings and logging isolation; no developer secrets or services."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import structlog
from asgi_correlation_id import correlation_id

from app.clients.mcp_client import McpClient
from app.core.settings_base import ProcessSettings
from app.services.graph import GraphService


@pytest.fixture(autouse=True)
def isolated_api_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Restore global logging and context after each app lifecycle test."""
    monkeypatch.setattr("app.services.metrics.MetricService.validate_startup", AsyncMock())
    monkeypatch.setattr("app.services.chat.ChatService.reconcile", AsyncMock())
    monkeypatch.setattr("app.application.GraphService", lambda _: AsyncMock(spec=GraphService))
    monkeypatch.setattr("app.application.McpClient", lambda _: AsyncMock(spec=McpClient))
    for key in os.environ:
        if key.upper().startswith("IP_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(ProcessSettings, "project_root", tmp_path)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    config = structlog.get_config()
    logger_state = {
        name: (logger.level, logger.handlers[:], logger.propagate)
        for name in (
            "httpx",
            "httpx2",
            "httpcore",
            "httpcore2",
            "mcp",
            "uvicorn",
            "uvicorn.error",
            "uvicorn.access",
        )
        for logger in [logging.getLogger(name)]
    }
    token = correlation_id.set(None)
    try:
        yield
    finally:
        root.handlers, root.level = handlers, level
        for name, (old_level, old_handlers, propagate) in logger_state.items():
            logger = logging.getLogger(name)
            logger.setLevel(old_level)
            logger.handlers = old_handlers
            logger.propagate = propagate
        structlog.configure(**config)
        structlog.contextvars.clear_contextvars()
        correlation_id.reset(token)
