"""Typed project failures with fixed public messages and internal diagnostics."""

from typing import TYPE_CHECKING, ClassVar

from app.schemas.mcp import SqlErrorKind

if TYPE_CHECKING:
    from app.schemas.mcp import PolicyReason, ValidationStatus


class InsightPilotError(Exception):
    """Base failure with a stable code and a safe public message."""

    code: ClassVar[str] = "internal_error"
    http_status: ClassVar[int] = 500
    user_message: ClassVar[str] = "An internal error occurred."
    retryable: ClassVar[bool] = False

    def __init__(self, detail: str = "", **context: object) -> None:
        super().__init__(detail or self.user_message)
        self.detail = detail
        self.context = context


class ValidationError(InsightPilotError):
    """The requested operation contains invalid business input."""

    code = "VALIDATION_ERROR"
    http_status = 400
    user_message = "The request is invalid."


class RetrievalUnavailableError(InsightPilotError):
    """Knowledge storage is unavailable; data-only requests remain independent."""

    code = "RETRIEVAL_UNAVAILABLE"
    http_status = 503
    user_message = "Knowledge retrieval is temporarily unavailable."


class RetrievalSchemaError(InsightPilotError):
    """An incompatible collection must never be silently adopted or overwritten."""

    code = "RETRIEVAL_SCHEMA_MISMATCH"
    http_status = 503
    user_message = "Knowledge storage requires an operator schema update."


class RetrievalConfigurationError(ValidationError):
    """The storage service rejected a schema, analyzer or search parameter."""

    code = "RETRIEVAL_CONFIGURATION_ERROR"


class PeriodUnresolved(ValidationError):  # noqa: N818 -- Step 2.4 public error name.
    """A period needs clarification instead of an unbounded query or a retry."""

    code = "PERIOD_UNRESOLVED"
    user_message = "The time period could not be resolved. Please clarify the dates."


class AuthenticationError(InsightPilotError):
    """The request lacks valid authentication."""

    code = "AUTHENTICATION_ERROR"
    http_status = 401
    user_message = "Authentication is required."


class TokenExpiredError(AuthenticationError):
    """An expired credential can be distinguished without exposing token data."""

    code = "token_expired"
    user_message = "The token has expired."


class PasswordPolicyError(ValidationError):
    """Only predefined policy rules are eligible for public rendering."""

    code = "PASSWORD_POLICY_ERROR"

    def __init__(self, failures: list[str]) -> None:
        super().__init__()
        self.failures = tuple(failures)


class AuthorizationError(InsightPilotError):
    """The authenticated caller cannot perform this operation."""

    code = "AUTHORIZATION_ERROR"
    http_status = 403
    user_message = "You are not authorized to perform this operation."


class NotFoundError(InsightPilotError):
    """The requested resource is unavailable."""

    code = "NOT_FOUND"
    http_status = 404
    user_message = "The requested resource was not found."


class RateLimitError(InsightPilotError):
    """The caller exhausted an applicable request limit."""

    code = "RATE_LIMIT_EXCEEDED"
    http_status = 429
    user_message = "Too many requests. Please try again later."


class QuotaExceededError(RateLimitError):
    """A quota failure carrying a safe retry delay."""

    code = "RATE_LIMIT_EXCEEDED"
    http_status = 429
    user_message = "Too many requests. Please try again later."

    def __init__(self, retry_after: int) -> None:
        super().__init__()
        self.retry_after = retry_after


RateLimitExceeded = RateLimitError


class DeadlineExceededError(InsightPilotError):
    """The request exhausted its total time budget."""

    code = "DEADLINE_EXCEEDED"
    http_status = 504
    user_message = "The request deadline was exceeded."


class ProviderProbeRetryableError(InsightPilotError):
    """A connection failure or temporary upstream failure during a probe."""

    code = "PROVIDER_PROBE_RETRYABLE"
    http_status = 503
    user_message = "The probe endpoint is temporarily unavailable."
    retryable = True


class HealthProbeError(InsightPilotError):
    """A readiness dependency could not complete its single bounded attempt."""

    code = "HEALTH_PROBE_UNAVAILABLE"
    http_status = 503
    user_message = "A required dependency is unavailable."


class HealthProbeTimeoutError(HealthProbeError):
    """The readiness operation exceeded its complete connection/check/cleanup budget."""

    code = "HEALTH_PROBE_TIMEOUT"


class DatabaseError(InsightPilotError):
    """An unclassified database failure, without exposing SQL or driver details."""

    code = "DATABASE_ERROR"
    retryable = False


class ConflictError(DatabaseError):
    """A unique database constraint rejected the operation; never retry it."""

    code = "CONFLICT"
    http_status = 409
    user_message = "The operation conflicts with existing data."


class UpstreamUnavailableError(DatabaseError):
    """A dependency is unavailable; transactions are not automatically replayed."""

    code = "UPSTREAM_UNAVAILABLE"
    http_status = 503
    user_message = "A required dependency is temporarily unavailable."
    retryable = True


class DatabaseTimeoutError(UpstreamUnavailableError):
    """A pool acquisition or database operation exceeded its time budget."""

    code = "DATABASE_TIMEOUT"
    retryable = False


class RetryNestingError(InsightPilotError):
    """A retry owner attempted to invoke another retry owner."""

    code = "RETRY_NESTING"


class OperationTimeoutError(UpstreamUnavailableError):
    """A safely retryable call exhausted its individual timeout."""

    code = "OPERATION_TIMEOUT"


class McpPolicyRejected(ValidationError):  # noqa: N818 -- public step contract.
    """A functioning trust boundary refused the query; never retry."""

    code = "MCP_POLICY_REJECTED"
    user_message = "The query was rejected by the data access policy."

    def __init__(self, status: "ValidationStatus", reasons: list["PolicyReason"]) -> None:
        super().__init__()
        self.status = status
        self.reasons = reasons


class SqlTimeoutError(DatabaseTimeoutError):
    """A server statement timeout is not a request to correct/re-run SQL."""

    code = "SQL_TIMEOUT"
    user_message = "The query exceeded the data source time limit."


class SqlExecutionError(DatabaseError):
    """A non-retryable SQL execution failure with no driver text in the response."""

    def __init__(self, kind: SqlErrorKind = SqlErrorKind.OTHER) -> None:
        """Retain only a safe correction category."""
        super().__init__()
        self.kind = kind

    code = "SQL_EXECUTION_FAILED"
    user_message = "The data source could not execute the query."


class McpUnavailableError(UpstreamUnavailableError):
    """Transport or database connectivity failure at the MCP boundary."""

    code = "MCP_UNAVAILABLE"
    user_message = "The data source is unavailable."


class McpResultError(InsightPilotError):
    """Malformed or unsupported data at the MCP boundary; never retry."""

    code = "MCP_INVALID_RESULT"
    http_status = 502
    user_message = "The data source returned an unsupported result."


class LlmConfigurationError(InsightPilotError):
    """Missing or invalid local provider capability evidence."""

    code = "LLM_CONFIGURATION_ERROR"
    user_message = "The language model configuration is invalid."


class LlmStructuredOutputError(InsightPilotError):
    """All supported structured-output strategies were exhausted."""

    code = "LLM_STRUCTURED_OUTPUT_FAILED"
    http_status = 502
    user_message = "The language model could not produce a valid structured response."


class LlmUnavailableError(UpstreamUnavailableError):
    """A temporary provider transport or server failure."""

    code = "LLM_UNAVAILABLE"
    user_message = "The language model is temporarily unavailable."


class LlmRateLimitError(LlmUnavailableError):
    """Approved Step 1.8 exception: only provider LLM 429 may retry."""

    code = "LLM_RATE_LIMITED"
    http_status = 429


class LlmRequestError(InsightPilotError):
    """A permanent provider rejection; never expose its response body."""

    code = "LLM_REQUEST_REJECTED"
    http_status = 502
    user_message = "The language model could not accept the request."


class LlmCapabilityError(LlmRequestError):
    """A typed provider rejection of the selected output capability."""


class LlmResponseError(InsightPilotError):
    """Malformed, truncated or schema-invalid provider output; never retry blindly."""

    code = "LLM_INVALID_RESPONSE"
    http_status = 502
    user_message = "The language model returned an invalid response."


class ContextBudgetExceeded(InsightPilotError):  # noqa: N818 -- design contract name.
    """Required evidence cannot fit the generation budget."""

    code = "context_budget_exceeded"


class CheckpointError(UpstreamUnavailableError):
    """Durable graph storage is unavailable or requires operator migration."""

    code = "checkpoint_unavailable"
    retryable = False


class SqlCorrectionExhaustedError(SqlExecutionError):
    """The single allowed semantic correction still failed."""

    code = "SQL_CORRECTION_EXHAUSTED"
    user_message = "The corrected query could not be executed."


class SchemaDriftError(InsightPilotError):
    """The semantic catalog does not match the live business schema."""

    code = "SCHEMA_DRIFT"
    user_message = "The schema catalog requires an update."


class SchemaMetadataError(InsightPilotError):
    """Persisted or authored metadata is malformed; never silently omit it."""

    code = "SCHEMA_METADATA_INVALID"
    user_message = "The schema metadata is invalid."


class MetricNotFound(NotFoundError):  # noqa: N818 -- Step 2.3 public error name.
    """The requested metric key or historical version is absent."""

    code = "METRIC_NOT_FOUND"
    user_message = "The requested metric definition was not found."


class UnsupportedGrain(ValidationError):  # noqa: N818 -- Step 2.3 public error name.
    """The catalog cannot interpret the requested grouping without guessing."""

    code = "UNSUPPORTED_GRAIN"
    user_message = "The metric does not support this grain."

    def __init__(self, supported: list[str]) -> None:
        super().__init__(supported=supported)
        self.supported = tuple(supported)


class MetricCatalogError(InsightPilotError):
    """Persisted catalog or template content violates its publishing contract."""

    code = "METRIC_CATALOG_INVALID"
    user_message = "The metric catalog is invalid."


class MetricNotIdentified(ValidationError):  # noqa: N818 -- roadmap public error name.
    """A quantitative question requires an identified published metric."""

    code = "METRIC_NOT_IDENTIFIED"


class InvalidMetricPatchError(ValidationError):
    """A patch cannot be expressed safely in the published query's existing scopes."""

    code = "INVALID_METRIC_PATCH"
    user_message = "Please clarify the requested metric definition."


# Compatibility for callers and historical imports; the error code is unchanged.
InvalidMetricPatch = InvalidMetricPatchError


class SqlGenerationError(InsightPilotError):
    """Generation did not supply a usable single SQL candidate."""

    code = "SQL_GENERATION_FAILED"
    http_status = 502
    user_message = "The language model could not generate a SQL query."
