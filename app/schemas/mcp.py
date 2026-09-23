"""Versioned MCP contracts; no driver, service or process configuration imports."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

RESULT_CEILING = 5000
type SqlValue = bool | int | float | str | None
PositiveRows = Annotated[int, Field(strict=True, gt=0)]


class Contract(BaseModel):
    """Reject unknown wire fields and unsafe floating-point values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DiscoveredToolResult(Contract):
    """Model-facing rendering of one discovered MCP result."""

    schema_version: Literal[1] = 1
    text: str = Field(min_length=1)


class McpReadyResponse(Contract):
    """The independent server readiness endpoint's complete wire contract."""

    ready: bool = Field(strict=True)


class QueryArguments(Contract):
    """The sole supported tool input; the server clamps a valid positive cap."""

    sql: str = Field(min_length=1, max_length=32000)
    max_rows: PositiveRows = 1000


class ColumnSpec(Contract):
    """PostgreSQL column name and native type, including on an empty result."""

    name: str
    type: str


class QueryResultPayload(Contract):
    """Rows returned by the executor, excluding the truncation sentinel."""

    schema_version: Literal[2] = 2
    executed_sql: str = Field(min_length=1, max_length=32000)
    limit_applied: bool
    execution_ms: int = Field(ge=0)
    mcp_call_id: str = Field(min_length=1, max_length=64)
    columns: list[ColumnSpec]
    rows: list[list[SqlValue]] = Field(max_length=RESULT_CEILING)
    row_count: int = Field(ge=0, le=RESULT_CEILING)
    result_truncated: bool

    @model_validator(mode="after")
    def check_shape(self) -> Self:
        """Reject a corrupt upstream result before it reaches application code."""
        if self.row_count != len(self.rows) or any(
            len(row) != len(self.columns) for row in self.rows
        ):
            raise PydanticCustomError("query_shape", "Query result shape is inconsistent")
        return self


class ValidationStatus(StrEnum):
    """Validation outcomes independent of explanatory prose."""

    VALID = "valid"
    INVALID = "invalid"
    UNSAFE = "unsafe"


class PolicyReason(StrEnum):
    """Machine-readable reasons for rejecting SQL or its execution cap."""

    INVALID_SQL = "invalid_sql"
    MULTIPLE_STATEMENTS = "multiple_statements"
    WRITE_OPERATION = "write_operation"
    BLOCKED_FUNCTION = "blocked_function"
    UNSUPPORTED_LIMIT = "unsupported_limit"
    INVALID_ARGUMENTS = "invalid_arguments"
    TABLE_NOT_ALLOWED = "table_not_allowed"
    NESTING_TOO_DEEP = "nesting_too_deep"
    UNSUPPORTED_SCOPE = "unsupported_scope"
    UNKNOWN_COLUMN = "unknown_column"
    INVALID_METRIC_BINDING = "invalid_metric_binding"
    INVALID_PERIOD = "invalid_period"
    PERIOD_TOO_LONG = "period_too_long"


class ValidationOutcome(Contract):
    """SQL policy output; only VALID SQL may reach the ordinary executor path."""

    status: ValidationStatus
    reasons: list[PolicyReason] = Field(default_factory=list)
    rewritten_sql: str = ""
    limit_applied: bool = False


class McpErrorCode(StrEnum):
    """Stable tool failures; retryability is established by local exception types."""

    POLICY_REJECTED = "MCP_POLICY_REJECTED"
    SQL_TIMEOUT = "SQL_TIMEOUT"
    SQL_EXECUTION_FAILED = "SQL_EXECUTION_FAILED"
    UNAVAILABLE = "MCP_UNAVAILABLE"
    INVALID_RESULT = "MCP_INVALID_RESULT"


class SqlErrorKind(StrEnum):
    """Safe correction eligibility, independent of driver prose."""

    UNDEFINED_TABLE = "undefined_table"
    UNDEFINED_COLUMN = "undefined_column"
    TYPE_MISMATCH = "type_mismatch"
    OTHER = "other"


class McpErrorPayload(Contract):
    """Safe error data, never SQL text or driver exception prose."""

    schema_version: Literal[2] = 2
    code: McpErrorCode
    message: str
    status: ValidationStatus | None = None
    reasons: list[PolicyReason] = Field(default_factory=list)
    column_name: str | None = Field(default=None, max_length=127)

    sql_error: SqlErrorKind = SqlErrorKind.OTHER
