"""API configuration types, separately importable without creating the singleton."""

from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import quote_plus, urlsplit

from pydantic import Field, HttpUrl, model_validator

from app.core.llm_config import ModelRole, ModelRoleSettings
from app.core.settings_base import (
    ConfigModel,
    Environment,
    ProcessSettings,
    Secret,
    require_configuration,
)
from app.retrieval.config import RetrievalSettings


class DatabaseSettings(ConfigModel):
    """Application database credentials only; business credentials have no field."""

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=15432, ge=1, le=65535)
    app_db: str = Field(default="insightpilot_app", min_length=1)
    app_user: str = Field(default="app_rw", min_length=1)
    app_password: Secret = Field(repr=False)
    pool_size: int = Field(default=10, ge=1, le=50)
    pool_recycle_s: int = Field(default=1800, ge=60, le=86400)
    max_overflow: int = Field(default=5, ge=0, le=50)
    pool_timeout_s: float = Field(default=30, ge=0.01, le=120)
    connect_timeout_s: float = Field(default=5, ge=0.01, le=60)
    command_timeout_s: float = Field(default=10, ge=0.01, le=120)
    echo_sql: bool = False

    @property
    def app_url(self) -> str:
        """Build the asyncpg URL; callers must never log the returned credential."""
        # SQLAlchemy decodes URL userinfo with unquote, not form-style unquote_plus.
        password = quote_plus(self.app_password.get_secret_value()).replace("+", "%20")
        user = quote_plus(self.app_user).replace("+", "%20")
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"postgresql+asyncpg://{user}:{password}@{host}:{self.port}/{self.app_db}"


class LLMSettings(ConfigModel):
    """Provider connection and independent per-role generation settings."""

    base_url: HttpUrl
    model: str = Field(min_length=1)
    api_key: Secret = Field(repr=False)
    timeout_s: float = Field(default=45, ge=0.01, le=120)
    roles: dict[ModelRole, ModelRoleSettings] = Field(default_factory=dict)
    capabilities_paths: list[Path] = Field(
        default_factory=lambda: [Path("app/resources/provider_capabilities.json")],
        min_length=1,
        max_length=16,
    )

    def for_role(self, role: ModelRole) -> ModelRoleSettings:
        """Resolve legacy defaults without mutating the configured role mapping."""
        configured = self.roles.get(role, ModelRoleSettings())
        return configured.model_copy(
            deep=True,
            update={
                "model": configured.model or self.model,
                "timeout_s": configured.timeout_s
                if configured.timeout_s is not None
                else self.timeout_s,
            },
        )


class MCPSettings(ConfigModel):
    """Authenticated HTTP boundary, with no business database connection field."""

    base_url: HttpUrl = HttpUrl("http://mcp:8001/mcp")
    auth_token: Secret = Field(repr=False)
    timeout_s: float = Field(default=15, ge=0.01, le=120)


class ObservabilitySettings(ConfigModel):
    """Optional tracing credentials and environment-sensitive logging defaults."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    log_format: Literal["console", "json"] = "console"
    langfuse_enabled: bool = False
    langfuse_base_url: HttpUrl | None = None
    langfuse_public_key: Secret | None = Field(default=None, repr=False)
    langfuse_secret_key: Secret | None = Field(default=None, repr=False)
    langfuse_mask_disabled: bool = False

    @model_validator(mode="after")
    def check_tracing(self) -> Self:
        """Enabling tracing requires a complete connection configuration."""
        require_configuration(
            not self.langfuse_mask_disabled, "trace masking cannot be disabled in any environment"
        )
        if self.langfuse_enabled:
            require_configuration(
                self.langfuse_base_url is not None
                and self.langfuse_public_key is not None
                and self.langfuse_secret_key is not None,
                "enabled Langfuse requires langfuse_base_url, langfuse_public_key and langfuse_secret_key",
            )
        return self


class ModelRuntimeClientSettings(ConfigModel):
    """Only the model HTTP credential is exposed to application clients."""

    base_url: HttpUrl = HttpUrl("http://model-tunnel:8100")
    auth_token: Secret = Field(repr=False)
    embed_timeout_s: float = Field(default=20, ge=0.01, le=120)
    rerank_timeout_s: float = Field(default=30, ge=0.01, le=120)
    embed_batch: int = Field(default=16, ge=1, le=16)
    max_concurrency: int = Field(default=1, ge=1, le=1)
    embed_revision: str = Field(
        default="5617a9f61b028005a4858fdac845db406aefb181", pattern=r"^[0-9a-f]{40}$"
    )
    rerank_revision: str = Field(
        default="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", pattern=r"^[0-9a-f]{40}$"
    )
    precision: Literal["fp16", "fp32"] = "fp16"


class HTTPSettings(ConfigModel):
    """Request budgets and explicit browser origins for the API edge."""

    request_timeout_s: float = Field(default=90, ge=1, le=600)
    shutdown_timeout_s: float = Field(default=10, ge=0.01, le=60)
    cors_origins: list[str] = Field(default_factory=list)
    cors_allow_credentials: bool = False

    @model_validator(mode="after")
    def check_origins(self) -> Self:
        """Only accept explicit HTTP origins, never wildcard credential access."""
        for origin in self.cors_origins:
            parsed = urlsplit(origin)
            require_configuration(
                parsed.scheme in {"http", "https"}
                and bool(parsed.netloc)
                and parsed.username is None
                and parsed.password is None
                and not parsed.path
                and not parsed.query
                and not parsed.fragment
                and "*" not in origin,
                "cors_origins must contain explicit HTTP origins without paths or credentials",
            )
        return self


class HealthSettings(ConfigModel):
    """Whole-operation budgets, including connection setup and cleanup."""

    postgresql_timeout_s: float = Field(default=2, ge=0.01, le=10)
    mcp_timeout_s: float = Field(default=3, ge=0.01, le=10)


class SecuritySettings(ConfigModel):
    """Required signing material and bounded credential lifetimes."""

    jwt_secret: Secret = Field(min_length=32, repr=False)
    access_ttl_minutes: int = Field(default=30, ge=1, le=60)
    refresh_ttl_days: int = Field(default=14, ge=1, le=30)
    bcrypt_rounds: int = Field(default=12, ge=4, le=16)


class RateRule(ConfigModel):
    """A bounded fixed-window quota."""

    requests: int = Field(ge=1, le=100000)
    seconds: int = Field(ge=1, le=86400)


class RateLimitSettings(ConfigModel):
    """All listed quotas apply; development uses one in-memory instance."""

    register_rules: list[RateRule] = Field(
        default_factory=lambda: [RateRule(requests=5, seconds=3600)], min_length=1, max_length=10
    )
    login_rules: list[RateRule] = Field(
        default_factory=lambda: [RateRule(requests=10, seconds=60)], min_length=1, max_length=10
    )
    refresh_rules: list[RateRule] = Field(
        default_factory=lambda: [RateRule(requests=20, seconds=3600)], min_length=1, max_length=10
    )
    conversations_rules: list[RateRule] = Field(
        default_factory=lambda: [RateRule(requests=60, seconds=60)], min_length=1, max_length=10
    )
    messages_rules: list[RateRule] = Field(
        default_factory=lambda: [RateRule(requests=20, seconds=60)], min_length=1, max_length=10
    )
    me_rules: list[RateRule] = Field(
        default_factory=lambda: [RateRule(requests=60, seconds=60)], min_length=1, max_length=10
    )


class SchemaCatalogSettings(ConfigModel):
    """Bounded reference-data caching; no persistent state is kept only in memory."""

    ttl_s: float = Field(default=300, ge=1, le=3600)
    operation_timeout_s: float = Field(default=30, ge=0.1, le=120)


class SchemaStrategy(StrEnum):
    """Reserved selection strategies; optional algorithms arrive in Step 2.5a."""

    FULL = "full"
    HEURISTIC = "heuristic"
    RETRIEVED = "retrieved"


class SanitySettings(ConfigModel):
    """Conservative expectations; result aliases are matched exactly and uniquely."""

    extreme_magnitude: Decimal = Field(
        default=Decimal("1000000000000"), gt=0, le=Decimal("1e100"), allow_inf_nan=False
    )
    money_columns: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(
        default_factory=list, max_length=200
    )
    nonzero_columns: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(
        default_factory=list, max_length=200
    )
    expected_max_rows: int | None = Field(default=None, ge=0, le=1_000_000_000)


class DataAgentSettings(ConfigModel):
    """Only implemented strategies may be enabled."""

    schema_strategy: SchemaStrategy = SchemaStrategy.FULL
    sanity: SanitySettings = Field(default_factory=SanitySettings)

    @model_validator(mode="after")
    def supported_strategy(self) -> Self:
        """Reject unavailable strategies before accepting requests."""
        require_configuration(
            self.schema_strategy is SchemaStrategy.FULL,
            "Only the full schema strategy is implemented; others require Step 2.5a",
        )
        return self


class RouterSettings(ConfigModel):
    """Classification acceptance threshold; equality is accepted."""

    min_confidence: float = Field(default=0.6, ge=0, le=1, allow_inf_nan=False)


class Settings(ProcessSettings):
    """Validated API process configuration with conditional service requirements."""

    process_name = "api"
    security: SecuritySettings = Field(default_factory=dict, validate_default=True)
    rate_limits: RateLimitSettings = Field(default_factory=RateLimitSettings)
    http: HTTPSettings = Field(default_factory=HTTPSettings)
    health: HealthSettings = Field(default_factory=HealthSettings)
    database: DatabaseSettings = Field(default_factory=dict, validate_default=True)
    llm: LLMSettings = Field(default_factory=dict, validate_default=True)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    mcp: MCPSettings = Field(default_factory=dict, validate_default=True)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    schema_catalog: SchemaCatalogSettings = Field(default_factory=SchemaCatalogSettings)
    data_agent: DataAgentSettings = Field(default_factory=DataAgentSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    semantic_memory_enabled: bool = False
    model_runtime: ModelRuntimeClientSettings | None = None

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Apply defaults only to unspecified fields and validate enabled features."""
        if self.environment is not Environment.DEVELOPMENT:
            defaults = {"log_level": "INFO", "log_format": "json"}
            changes = {
                key: value
                for key, value in defaults.items()
                if key not in self.observability.model_fields_set
            }
            self.observability = self.observability.model_copy(update=changes)
        if self.environment is Environment.PRODUCTION:
            require_configuration(
                not self.observability.langfuse_mask_disabled,
                "trace masking cannot be disabled in production",
            )
        if self.retrieval.enabled or self.semantic_memory_enabled:
            require_configuration(
                self.model_runtime is not None,
                "retrieval or semantic memory requires model_runtime with auth_token",
            )
        return self
