"""Model failures extend the shared project hierarchy."""

from typing import ClassVar

from app.core.errors import InsightPilotError
from app.schemas.model_runtime import ModelFailureKind


class ModelError(InsightPilotError):
    """Typed failure across the model service boundary."""

    kind: ClassVar[ModelFailureKind] = ModelFailureKind.UNAVAILABLE
    code = ModelFailureKind.UNAVAILABLE.value
    http_status = 503
    retryable = True
    user_message = "Model service is unavailable."


class ModelContractError(ModelError):
    """Invalid identity or incomplete output cannot be consumed or retried."""

    kind = ModelFailureKind.CONTRACT
    code = ModelFailureKind.CONTRACT.value
    retryable = False
    user_message = "Model contract validation failed."


class ModelOOMError(ModelError):
    """The single server-side recovery attempt exhausted this input's budget."""

    kind = ModelFailureKind.OOM
    code = ModelFailureKind.OOM.value
    retryable = False
    user_message = "Model capacity is insufficient for this request."


class ModelQueueError(ModelError):
    """Bounded waiting capacity is exhausted."""

    kind = ModelFailureKind.QUEUE_FULL
    code = ModelFailureKind.QUEUE_FULL.value
    user_message = "Model queue is full."


class ModelDeadlineError(ModelError):
    """A consumed deadline cannot be renewed by retries."""

    kind = ModelFailureKind.DEADLINE
    code = ModelFailureKind.DEADLINE.value
    http_status = 504
    retryable = False
    user_message = "Model operation deadline exceeded."


class ModelAuthError(ModelError):
    """Missing or incorrect service token."""

    kind = ModelFailureKind.AUTH
    code = ModelFailureKind.AUTH.value
    http_status = 401
    retryable = False
    user_message = "Model authentication is required."


class ModelInputError(ModelError):
    """Malformed or oversized input is rejected before admission."""

    kind = ModelFailureKind.INPUT
    code = ModelFailureKind.INPUT.value
    http_status = 422
    retryable = False
    user_message = "Model request is invalid."
