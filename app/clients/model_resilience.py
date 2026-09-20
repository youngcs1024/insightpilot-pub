"""Model-specific failure policy and single-probe circuit admission."""

import time
from collections.abc import Callable
from http import HTTPStatus

import structlog

from app.schemas.model_runtime import ModelFailureKind
from model_runtime.errors import ModelError

logger = structlog.get_logger(__name__)
BREAKER_FAILURES = 5
BREAKER_RECOVERY_S = 30
STARTUP_BUDGET_S = 180
WARMUP_POLL_S = 1


def retryable_model_failure(exc: BaseException) -> bool:
    """Only declared transient model failures can repeat an inference request."""
    return (
        isinstance(exc, ModelError)
        and exc.retryable
        and exc.kind in {ModelFailureKind.UNAVAILABLE, ModelFailureKind.QUEUE_FULL}
        and exc.http_status >= HTTPStatus.INTERNAL_SERVER_ERROR
    )


class ModelCircuitBreaker:
    """Count final logical failures, not attempts or readiness polls."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None
        self.probing = False

    def check(self) -> None:
        """Reject unavailable admission before waiting for the inference semaphore."""
        if self.opened_at is not None and (
            self.probing or self.clock() - self.opened_at < BREAKER_RECOVERY_S
        ):
            raise ModelError()

    def enter(self) -> bool:
        """Reserve the only recovery probe without suspending the event loop."""
        self.check()
        if self.opened_at is None:
            return False
        self.probing = True
        logger.info("model_circuit_half_open")
        return True

    def success(self) -> None:
        """Only complete inference success proves recovery."""
        if self.opened_at is not None:
            logger.info("model_circuit_closed")
        self.failures = 0
        self.opened_at = None
        self.probing = False

    def failed(self) -> None:
        """A final retryable failure opens or restarts the recovery interval."""
        self.failures += 1
        if self.failures >= BREAKER_FAILURES:
            self.opened_at = self.clock()
            logger.info("model_circuit_opened", failures=self.failures)
        self.probing = False

    def release(self, probe: bool) -> None:
        """Cancellation and nonretryable errors relinquish the probe without success."""
        if probe:
            self.probing = False
