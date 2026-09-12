"""Structured agent outcomes, independent of HTTP exceptions and retry execution."""

from enum import StrEnum

from pydantic import BaseModel


class FailureKind(StrEnum):
    """Closed failure vocabulary from the system specification, section 8.5."""

    CLIENT_DISCONNECTED = "client_disconnected"
    INTERRUPTED = "interrupted"
    CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"
    NODE_OPERATION_FAILED = "node_operation_failed"
    ROUTE_LOW_CONFIDENCE = "route_low_confidence"
    SQL_GENERATION_FAILED = "sql_generation_failed"
    SQL_VALIDATION_FAILED = "sql_validation_failed"
    SQL_EXECUTION_FAILED = "sql_execution_failed"
    SQL_TIMEOUT = "sql_timeout"
    SQL_CORRECTION_EXHAUSTED = "sql_correction_exhausted"
    MCP_UNAVAILABLE = "mcp_unavailable"
    MCP_POLICY_REJECTED = "mcp_policy_rejected"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"
    RETRIEVAL_NO_EVIDENCE = "retrieval_no_evidence"
    MODEL_RUNTIME_UNAVAILABLE = "model_runtime_unavailable"
    LLM_STRUCTURED_OUTPUT_FAILED = "llm_structured_output_failed"
    DEADLINE_EXCEEDED = "deadline_exceeded"


class NodeFailure(BaseModel):
    """An internal node outcome for agent state, never an HTTP response."""

    node: str
    kind: FailureKind
    detail: str
    retryable: bool
