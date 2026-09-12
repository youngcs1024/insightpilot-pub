"""Sequential, bounded live HTTP experiments with typed retry policy."""

import asyncio
from datetime import UTC, datetime
from http import HTTPStatus
from time import monotonic

import httpx
import structlog
from pydantic import ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.errors import ProviderProbeRetryableError
from spikes.provider.experiment import SAMPLE_COUNTS, examine, make_request, summarize
from spikes.provider.models import (
    Attempt,
    Completion,
    Observation,
    Outcome,
    ProbeKind,
    ProviderSettings,
    Report,
)

logger = structlog.get_logger()
RETRYABLE_STATUS = frozenset(
    {
        HTTPStatus.INTERNAL_SERVER_ERROR,
        HTTPStatus.BAD_GATEWAY,
        HTTPStatus.SERVICE_UNAVAILABLE,
        HTTPStatus.GATEWAY_TIMEOUT,
    }
)


def classify_status(status: int) -> Outcome:
    """Classify by HTTP status only, never by matching error messages."""
    if status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
        return Outcome.AUTH
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        return Outcome.RATE_LIMITED
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return Outcome.SERVER_ERROR
    return Outcome.REJECTED


async def exchange(
    client: httpx.AsyncClient, settings: ProviderSettings, observation: Observation
) -> None:
    """Execute one attempt, recording cancellation and sanitized failures."""
    started = monotonic()
    attempt = Attempt(number=len(observation.attempts) + 1, outcome=Outcome.UNKNOWN, elapsed_ms=0)
    retryable = False
    try:
        response = await client.post(
            settings.endpoint,
            headers={"Authorization": "Bearer " + settings.api_key.get_secret_value()},
            json=observation.request.model_dump(mode="json", exclude_none=True),
            timeout=settings.timeout_seconds,
        )
        attempt.http_status = response.status_code
        if response.status_code == HTTPStatus.OK:
            examine(Completion.model_validate_json(response.content), observation, attempt)
        else:
            attempt.outcome = classify_status(response.status_code)
            retryable = response.status_code in RETRYABLE_STATUS
    except httpx.TimeoutException:
        attempt.outcome = Outcome.TIMEOUT
    except httpx.ConnectError:
        attempt.outcome = Outcome.CONNECTION
        retryable = True
    except httpx.RequestError:
        attempt.outcome = Outcome.CONNECTION
    except ValidationError:
        attempt.outcome = Outcome.PROTOCOL
    except asyncio.CancelledError:
        attempt.outcome = Outcome.DEADLINE
        raise
    finally:
        attempt.elapsed_ms = (monotonic() - started) * 1000
        observation.attempts.append(attempt)
        observation.outcome = attempt.outcome
    logger.info(
        "provider_probe_attempt",
        probe=observation.probe.value,
        sample=observation.sample,
        attempt=attempt.number,
        outcome=attempt.outcome.value,
        elapsed_ms=round(attempt.elapsed_ms, 2),
    )
    if retryable:
        raise ProviderProbeRetryableError


async def measure(
    client: httpx.AsyncClient, settings: ProviderSettings, observation: Observation, deadline: float
) -> None:
    """Bound the whole sample, including retry backoff, by the shared deadline."""
    remaining = deadline - monotonic()
    if remaining <= 0:
        observation.outcome = Outcome.DEADLINE
        return
    try:
        async with asyncio.timeout(remaining):
            await retry_sample(client, settings, observation)
    except ProviderProbeRetryableError:
        return
    except TimeoutError:
        observation.outcome = Outcome.DEADLINE


async def retry_sample(
    client: httpx.AsyncClient, settings: ProviderSettings, observation: Observation
) -> None:
    """Apply one retry budget to the sample without nested retry layers."""
    async for retry in AsyncRetrying(
        stop=stop_after_attempt(settings.max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(ProviderProbeRetryableError),
        reraise=True,
    ):
        with retry:
            await exchange(client, settings, observation)


async def run(settings: ProviderSettings, client: httpx.AsyncClient) -> Report:
    """Run all planned samples once; never silently replace unsuccessful samples."""
    report = Report(
        evidence_source="live_api",
        started_at=datetime.now(UTC),
        timeout_seconds=settings.timeout_seconds,
        deadline_seconds=settings.deadline_seconds,
        max_attempts=settings.max_attempts,
    )
    deadline = monotonic() + settings.deadline_seconds
    auth_failed = False
    for kind in ProbeKind:
        for sample in range(1, SAMPLE_COUNTS[kind] + 1):
            observation = Observation(probe=kind, sample=sample, request=make_request(kind))
            report.observations.append(observation)
            if auth_failed:
                observation.outcome = Outcome.AUTH
                continue
            await measure(client, settings, observation, deadline)
            auth_failed = observation.outcome == Outcome.AUTH
    report.finished_at = datetime.now(UTC)
    summarize(report)
    return report
