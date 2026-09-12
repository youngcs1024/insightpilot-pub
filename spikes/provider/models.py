"""Versioned probe contracts and secret-bearing process settings."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL: Literal["qwen3.6-flash-2026-04-16"] = "qwen3.6-flash-2026-04-16"


class Contract(BaseModel):
    """Reject accidental changes at project-owned boundaries."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class ProviderSettings(Contract):
    """Beijing workspace credentials and bounded experiment budgets."""

    workspace_id: str = Field(pattern=r"^llm-[a-zA-Z0-9-]{1,100}$")
    api_key: SecretStr = Field(min_length=1)
    timeout_seconds: float = Field(default=60, ge=1, le=120)
    deadline_seconds: float = Field(default=900, ge=1, le=3600)
    max_attempts: int = Field(default=2, ge=1, le=3)

    @property
    def endpoint(self) -> str:
        """Construct only the approved regional workspace endpoint."""
        return (
            f"https://{self.workspace_id}.cn-beijing.maas.aliyuncs.com"
            "/compatible-mode/v1/chat/completions"
        )


class Settings(BaseSettings):
    """Load dedicated variables and only an explicitly selected dotenv file."""

    model_config = SettingsConfigDict(
        env_prefix="IP_SPIKE_",
        env_nested_delimiter="__",
        extra="forbid",
        hide_input_in_errors=True,
    )
    provider: ProviderSettings


class ProbeKind(StrEnum):
    """The eight experiments required by Step 0.4."""

    NATIVE = "native_json_schema"
    TOOL = "tool_call_structured"
    PARALLEL = "parallel_tool_calls"
    PROMPTED = "prompted_json"
    ZERO = "temperature_zero"
    CHINESE = "chinese_structured"
    LONG = "long_context"
    LATENCY = "latency"


class Outcome(StrEnum):
    """Observed outcomes, never inferred from upstream error prose."""

    SUPPORTED = "supported"
    SCHEMA_INVALID = "schema_invalid_or_ignored"
    REJECTED = "request_rejected"
    AUTH = "authentication_failed"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    CONNECTION = "connection_failed"
    TIMEOUT = "timeout"
    DEADLINE = "deadline_exceeded"
    PROTOCOL = "invalid_response"
    UNKNOWN = "unknown"


class Region(Contract):
    """Nested field in the deliberately small structured-output schema."""

    name: Literal["华东"]


class StructuredSample(Contract):
    """Three required fields including a nested object and Chinese values."""

    metric: Literal["退款率"]
    count: Literal[7]
    region: Region


class Message(Contract):
    """Synthetic input message; never populated from environment variables."""

    role: Literal["system", "user"]
    content: str


class Request(Contract):
    """Exact non-streaming OpenAI-compatible request stored as evidence."""

    model: Literal["qwen3.6-flash-2026-04-16"] = MODEL
    messages: list[Message]
    temperature: float = 0
    max_tokens: int = 512
    enable_thinking: bool = False
    stream: bool = False
    response_format: dict[str, JsonValue] | None = None
    tools: list[dict[str, JsonValue]] | None = None
    tool_choice: dict[str, JsonValue] | Literal["auto"] | None = None
    parallel_tool_calls: bool | None = None


class FunctionCall(BaseModel):
    """Relevant portion of a provider tool call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Relevant tool descriptor fields from the wire."""

    type: Literal["function"]
    function: FunctionCall


class ResponseMessage(BaseModel):
    """Provider message, with unrelated vendor extensions ignored."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Choice(BaseModel):
    """One completion and its provider-reported termination reason."""

    message: ResponseMessage
    finish_reason: str


class Usage(BaseModel):
    """Provider-reported token counts, not character-count estimates."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class Completion(BaseModel):
    """Typed response boundary; malformed successful HTTP bodies fail validation."""

    choices: list[Choice] = Field(min_length=1, max_length=1)
    usage: Usage | None = None


class Attempt(Contract):
    """Safe evidence for a single HTTP attempt, including failed retries."""

    number: int = Field(ge=1)
    outcome: Outcome
    elapsed_ms: float = Field(ge=0)
    http_status: int | None = None
    usage: Usage | None = None
    output_sha256: str | None = None
    finish_reason: Literal["stop", "length", "tool_calls", "other"] | None = None
    tool_count: int = Field(default=0, ge=0)


class Observation(Contract):
    """One planned sample with its full retry history and bounded request."""

    probe: ProbeKind
    sample: int = Field(ge=1)
    request: Request
    outcome: Outcome = Outcome.UNKNOWN
    attempts: list[Attempt] = Field(default_factory=list)


class Report(Contract):
    """Persisted capability map consumed per model by the future role registry."""

    schema_version: Literal[1] = 1
    evidence_source: Literal["live_api", "not_run"]
    model: Literal["qwen3.6-flash-2026-04-16"] = MODEL
    provider: Literal["aliyun_bailian"] = "aliyun_bailian"
    region: Literal["cn-beijing"] = "cn-beijing"
    endpoint_template: str = (
        "https://{workspace_id}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )
    started_at: datetime | None = None
    finished_at: datetime | None = None
    timeout_seconds: float | None = None
    deadline_seconds: float | None = None
    max_attempts: int | None = None
    observations: list[Observation] = Field(default_factory=list)
    recommended_tier: Literal[1, 2, 3] | None = None
    execution_complete: bool = False
