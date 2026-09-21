"""Typed result conversion and public failure mapping."""

from app.agents.contracts import Answer
from app.agents.degradation import missing_explanations, public_failure
from app.agents.contracts import EvidenceBundle
from app.agents.failures import FailureKind, NodeFailure
from app.core.errors import DeadlineExceededError, InsightPilotError
from app.db.models import Turn
from app.schemas.chat import TurnResponse


class TurnFailedError(InsightPilotError):
    """Safe graph/lifecycle outcome, never its internal failure detail."""

    user_message = "The analysis could not be completed."

    def __init__(self, reason: FailureKind, failures: list[NodeFailure] | None = None) -> None:
        super().__init__()
        self.reason = reason
        self.failures = failures or []

    @property
    def public_message(self) -> str:
        """Safe source-specific detail for the existing error response envelope."""
        return public_failure(self.reason)


class BothSourcesFailedError(TurnFailedError):
    """Name both unavailable sources while preserving the typed safe failure reason."""

    user_message = "业务数据与知识来源均未能提供可用证据，本次分析失败。"

    @property
    def public_message(self) -> str:
        return self.user_message + " ".join(missing_explanations(EvidenceBundle(), self.failures))


def turn_response(turn: Turn, *, replayed: bool = False) -> TurnResponse:
    """Validate database JSON at the boundary; do not reconstruct an old answer."""
    result = TurnResponse.model_validate(turn)
    if turn.answer is not None:
        answer = Answer.model_validate(turn.answer)
        result.answer = answer
        result.evidence_refs = answer.evidence_refs
    result.replayed = replayed
    return result


def failure_reason(error: InsightPilotError) -> FailureKind:
    """Use typed failures rather than prose matching."""
    if isinstance(error, TurnFailedError):
        return error.reason
    if isinstance(error, DeadlineExceededError):
        return FailureKind.DEADLINE_EXCEEDED
    return FailureKind.NODE_OPERATION_FAILED
