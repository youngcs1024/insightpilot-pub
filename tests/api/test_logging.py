"""No credentials or upstream exception prose may reach the stdout log stream."""

import json
import logging
from urllib.parse import quote_plus

import pytest
import structlog
from pydantic import SecretStr

from app.core.config_models import Settings
from app.core.errors import HealthProbeError
from app.core.logging import setup_logging


@pytest.mark.parametrize("log_format", ["json", "console"])
def test_logs_redact_nested_secrets_and_exception_prose(
    settings: Settings, capsys: pytest.CaptureFixture[str], log_format: str
) -> None:
    settings.observability = settings.observability.model_copy(update={"log_format": log_format})
    setup_logging(settings)
    secret = settings.database.app_password.get_secret_value()
    logger = structlog.get_logger("health_test")
    logger.info(
        "secret_test",
        nested={"password": "unknown-secret", "values": [SecretStr("nested-secret"), secret]},
        detail=quote_plus(secret),
        message="Bearer secret-bearer-value",
    )
    logging.getLogger("sdk_test").warning("upstream %s", secret)
    logging.getLogger("uvicorn.error").warning("server %s", secret)
    try:
        raise HealthProbeError("arbitrary-private-upstream-data")
    except HealthProbeError:
        logger.exception("safe_failure")
    output = capsys.readouterr()
    assert output.err == ""
    for private in (
        secret,
        quote_plus(secret),
        "unknown-secret",
        "nested-secret",
        "secret-bearer-value",
        "arbitrary-private-upstream-data",
    ):
        assert private not in output.out
    assert "HealthProbeError" in output.out
    if log_format == "json":
        records = [json.loads(line) for line in output.out.splitlines()]
        assert all(
            {"event", "timestamp", "level", "logger", "request_id"} <= record.keys()
            for record in records
        )


@pytest.mark.parametrize("log_format", ["json", "console"])
@pytest.mark.parametrize(
    "diagnostic",
    [
        "password=hunter2",
        'PASSWORD = "hunter2 with spaces"',
        "{'access_token': 'hunter2 with spaces'}",
        'api-key="hunter2\\"with quote"',
        "client_secret=hunter2; operation=query",
        "authorization: Bearer hunter2",
        'payload={"password":"hunter2","count":1}',
    ],
)
def test_credential_assignments_are_redacted_in_diagnostics(
    settings: Settings, capsys: pytest.CaptureFixture[str], log_format: str, diagnostic: str
) -> None:
    settings.observability = settings.observability.model_copy(update={"log_format": log_format})
    setup_logging(settings)
    structlog.get_logger("error_test").warning(
        "request_failed", detail="ordinary diagnostic " + diagnostic
    )
    output = capsys.readouterr()
    assert output.err == ""
    assert "hunter2" not in output.out
    assert "with spaces" not in output.out
    assert "with quote" not in output.out
    assert "ordinary diagnostic" in output.out
    assert "***" in output.out


def test_usage_counts_survive_but_additional_credentials_do_not(
    settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging(settings)
    structlog.get_logger("usage_test").info(
        "llm_completion",
        prompt_tokens=17,
        completion_tokens=3,
        public_key="private-value",
        passwd="private-value",  # noqa: S106 -- synthetic logging test.
        detail="Basic private-value pk-lf-private-value",
    )
    output = capsys.readouterr().out
    assert "private-value" not in output
    record = json.loads(output)
    assert record["prompt_tokens"] == 17  # noqa: PLR2004 -- synthetic usage.
    assert record["completion_tokens"] == 3  # noqa: PLR2004 -- synthetic usage.
