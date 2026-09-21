"""Pure node failure translation with explicit typed terminal routing."""

import structlog
from langgraph.graph import END
from langgraph.types import Command

from app.agents.failures import FailureKind, NodeFailure
from app.agents.state import AgentState
from app.core.errors import (
    ContextBudgetExceeded,
    DeadlineExceededError,
    InsightPilotError,
    LlmStructuredOutputError,
    McpPolicyRejected,
    McpUnavailableError,
    RetrievalUnavailableError,
    SqlCorrectionExhaustedError,
    SqlExecutionError,
    SqlGenerationError,
    SqlTimeoutError,
)
from model_runtime.errors import (
    ModelAuthError,
    ModelContractError,
    ModelDeadlineError,
    ModelError,
    ModelInputError,
)

logger = structlog.get_logger(__name__)


def node_failure(node: str, exc: InsightPilotError) -> NodeFailure:
    """Translate operational errors without retaining private exception prose."""
    kinds: dict[type[InsightPilotError], FailureKind] = {
        LlmStructuredOutputError: FailureKind.LLM_STRUCTURED_OUTPUT_FAILED,
        McpPolicyRejected: FailureKind.MCP_POLICY_REJECTED,
        McpUnavailableError: FailureKind.MCP_UNAVAILABLE,
        SqlTimeoutError: FailureKind.SQL_TIMEOUT,
        SqlExecutionError: FailureKind.SQL_EXECUTION_FAILED,
        SqlGenerationError: FailureKind.SQL_GENERATION_FAILED,
        SqlCorrectionExhaustedError: FailureKind.SQL_CORRECTION_EXHAUSTED,
        DeadlineExceededError: FailureKind.DEADLINE_EXCEEDED,
        ContextBudgetExceeded: FailureKind.CONTEXT_BUDGET_EXCEEDED,
        RetrievalUnavailableError: FailureKind.RETRIEVAL_UNAVAILABLE,
    }
    kind = kinds.get(type(exc), FailureKind.NODE_OPERATION_FAILED)
    if isinstance(exc, ModelDeadlineError):
        kind = FailureKind.DEADLINE_EXCEEDED
    elif isinstance(exc, ModelError) and not isinstance(
        exc, (ModelAuthError, ModelContractError, ModelInputError)
    ):
        kind = FailureKind.MODEL_RUNTIME_UNAVAILABLE
    return NodeFailure(node=node, kind=kind, detail=exc.user_message, retryable=exc.retryable)


def failed(node: str, state: AgentState, exc: InsightPilotError) -> Command[str]:
    """No exception prose becomes durable failure detail or answer content."""
    logger.exception("graph_node_failed", node=node, code=exc.code, exc_info=False)
    failure = node_failure(node, exc)
    return Command(update={"failures": [failure], "status": "failed"}, goto=END)


def specialist_failed(node: str, exc: Exception, *, deadline_expired: bool = False) -> Command[str]:
    """Record one failure without terminating or overwriting a concurrent sibling."""
    if deadline_expired:
        error: InsightPilotError = DeadlineExceededError()
    else:
        error = exc if isinstance(exc, InsightPilotError) else InsightPilotError()
    logger.exception(
        "specialist_node_failed",
        node=node,
        code=error.code,
        error_type=type(exc).__name__,
        exc_info=False,
    )
    return Command(update={"failures": [node_failure(node, error)]})
