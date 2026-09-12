"""Stdout-only structured logging with request context and fail-closed exceptions."""

import logging
import re
import sys
from collections.abc import Mapping
from urllib.parse import quote_plus

import structlog
from asgi_correlation_id import correlation_id
from pydantic import BaseModel, SecretStr
from structlog.types import EventDict, Processor, WrappedLogger

from app.core.config_models import Settings

_SECRET_KEY = re.compile(
    r"password|passwd|secret|token|authorization|api[_-]?key|private[_-]?key|public[_-]?key|credential|cookie",
    re.IGNORECASE,
)
_CREDENTIAL = re.compile(
    r"(?:Bearer|Basic)\s+\S+|(?:postgresql(?:\+asyncpg)?|https?)://[^\s/@]+:[^\s/@]+@"
    r"|\b(?:sk-|pk-lf-)[A-Za-z0-9_-]+",
    re.IGNORECASE,
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"\b[\w-]*(?:password|passwd|secret|token|authorization|api[_-]?key)[\w-]*[\"']?\s*[:=]\s*"
    r"(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,;&}]+)",
    re.IGNORECASE,
)


def _secrets(value: object) -> set[str]:
    if isinstance(value, SecretStr):
        secret = value.get_secret_value()
        return {secret, quote_plus(secret)} if secret else set()
    if isinstance(value, Mapping):
        return {secret for item in value.values() for secret in _secrets(item)}
    if isinstance(value, (list, tuple)):
        return {secret for item in value for secret in _secrets(item)}
    return set()


def credential_field(key: str) -> bool:
    """Token usage counters are diagnostics, not authentication token fields."""
    return key not in {"prompt_tokens", "completion_tokens", "total_tokens"} and bool(
        _SECRET_KEY.search(key)
    )


class SecretRedactor:
    """Redact known credentials and credential-shaped fields before rendering."""

    def __init__(self, settings: BaseModel) -> None:
        self._values = sorted(_secrets(settings.model_dump()), key=len, reverse=True)

    def clean(self, value: object) -> object:
        """Walk structured fields without ever rendering a SecretStr value."""
        if isinstance(value, SecretStr):
            return "***"
        if isinstance(value, Mapping):
            return {
                str(key): "***" if credential_field(str(key)) else self.clean(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [self.clean(item) for item in value]
        if isinstance(value, str):
            for secret in self._values:
                value = value.replace(secret, "***")
            return _CREDENTIAL_ASSIGNMENT.sub("***", _CREDENTIAL.sub("***", value))
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return "<redacted>"

    def __call__(self, _: WrappedLogger, __: str, event: EventDict) -> EventDict:
        """Adapt structlog's untyped event boundary to safe scalar/container values."""
        return {key: "***" if credential_field(key) else self.clean(v) for key, v in event.items()}


def add_safe_context(_: WrappedLogger, __: str, event: EventDict) -> EventDict:
    """Attach correlation and retain exception type without unsafe upstream prose."""
    event["request_id"] = correlation_id.get()
    exc_info = event.pop("exc_info", None)
    if exc_info:
        exc_type = exc_info[0] if isinstance(exc_info, tuple) else sys.exc_info()[0]
        event["exception_type"] = exc_type.__name__ if exc_type else "Exception"
    event.pop("stack_info", None)
    return event


def setup_logging(settings: Settings) -> None:
    """Configure both project and third-party logs to the same safe stdout handler."""
    common: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_safe_context,
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if settings.observability.log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=common,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                SecretRedactor(settings),
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.observability.log_level)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *common,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    # SDK INFO records include opaque session IDs and HTTP connection metadata.
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "mcp"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # Uvicorn installs independent stderr/access handlers before lifespan starts.
    # Route worker logs through the same formatter/redactor from this point on.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        server_logger = logging.getLogger(name)
        server_logger.handlers = []
        server_logger.propagate = True
