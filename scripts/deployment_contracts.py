"""Independently reviewed service environment keys, never derived from Compose."""

from types import MappingProxyType

from pydantic import BaseModel

API_ENVIRONMENT_KEYS = frozenset(
    {
        "IP_DATABASE__HOST",
        "IP_DATABASE__PORT",
        "IP_DATABASE__APP_PASSWORD",
        "IP_MCP__BASE_URL",
        "IP_MCP__AUTH_TOKEN",
        "IP_SECURITY__JWT_SECRET",
        "IP_LLM__BASE_URL",
        "IP_LLM__MODEL",
        "IP_LLM__API_KEY",
        "IP_LLM__TIMEOUT_S",
        "IP_LLM__ROLES",
        "IP_LLM__CAPABILITIES_PATHS",
        "IP_OBSERVABILITY",
        "IP_OBSERVABILITY__LOG_FORMAT",
        "IP_DATA_AGENT",
        "IP_ROUTER",
        "IP_RETRIEVAL",
        "IP_MODEL_RUNTIME",
    }
)
SERVICE_ENVIRONMENT_KEYS = MappingProxyType(
    {
        "model-tunnel": frozenset(),
        "model-diagnostics": frozenset({"IP_MODEL_RUNTIME"}),
        "etcd": frozenset(
            {
                "ETCD_AUTO_COMPACTION_MODE",
                "ETCD_AUTO_COMPACTION_RETENTION",
                "ETCD_QUOTA_BACKEND_BYTES",
                "ETCD_SNAPSHOT_COUNT",
            }
        ),
        "minio": frozenset({"MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD"}),
        "milvus": frozenset(
            {
                "ETCD_ENDPOINTS",
                "MINIO_ADDRESS",
                "MINIO_ACCESS_KEY_ID",
                "MINIO_SECRET_ACCESS_KEY",
            }
        ),
        "api": API_ENVIRONMENT_KEYS,
        "mcp": frozenset({"IP_BUSINESS__HOST", "IP_BUSINESS__PASSWORD", "IP_MCP__AUTH_TOKEN"}),
        "postgres": frozenset(
            {
                "POSTGRES_PASSWORD",
                "IP_BOOTSTRAP_APP_PASSWORD",
                "IP_BOOTSTRAP_ETL_PASSWORD",
                "IP_BOOTSTRAP_MCP_PASSWORD",
            }
        ),
        "migrate": frozenset(
            {"IP_MIGRATION__HOST", "IP_MIGRATION__PORT", "IP_MIGRATION__PASSWORD"}
        ),
        "seed": frozenset({"IP_SEED__HOST", "IP_SEED__PORT", "IP_SEED__PASSWORD"}),
    }
)


class EnvironmentIssues(BaseModel):
    """Only key names cross the diagnostic boundary; never configuration values."""

    missing: list[str]
    unexpected: list[str]
    forbidden: list[str]

    @property
    def passed(self) -> bool:
        """An exact allowlist does not override the independent credential prohibition."""
        return not (self.missing or self.unexpected or self.forbidden)


def forbidden_api_keys(keys: set[str]) -> list[str]:
    """Forbid business/operator credentials even if the allowlist is edited incorrectly."""
    prefixes = (
        "IP_BUSINESS",
        "IP_MIGRATION",
        "IP_BOOTSTRAP",
        "IP_SEED",
        "IP_OPERATOR",
        "PG",
        "IP_MINIO",
        "MINIO_",
    )
    standalone = {"DATABASE_URL", "BUSINESS_DATABASE_URL", "POSTGRES_PASSWORD", "POSTGRES_USER"}
    return sorted(
        key for key in keys if key.upper().startswith(prefixes) or key.upper() in standalone
    )


def environment_issues(
    service: str, keys: set[str], *, partial: bool = False, remote: bool = False
) -> EnvironmentIssues:
    """Compare declared or runtime keys to an independent, service-specific contract."""
    expected = (MODEL_SERVER_ENVIRONMENT_KEYS if remote else SERVICE_ENVIRONMENT_KEYS)[service]
    return EnvironmentIssues(
        missing=[] if partial else sorted(expected - keys),
        unexpected=sorted(keys - expected),
        forbidden=forbidden_api_keys(keys) if service == "api" else [],
    )


MODEL_SERVER_ENVIRONMENT_KEYS = MappingProxyType(
    {
        "model-runtime": frozenset(
            {
                "IP_MODEL_SERVER__AUTH_TOKEN",
                "IP_MODEL_SERVER__EMBED_REVISION",
                "IP_MODEL_SERVER__RERANK_REVISION",
                "IP_MODEL_SERVER__DEVICE",
                "IP_MODEL_SERVER__PRECISION",
                "IP_MODEL_SERVER__WORKERS",
                "IP_MODEL_SERVER__MAX_CONCURRENCY",
                "IP_MODEL_SERVER__QUEUE_CAPACITY",
                "IP_MODEL_SERVER__EMBED_BATCH",
                "IP_MODEL_SERVER__RERANK_BATCH",
                "IP_MODEL_SERVER__EMBED_MAX_LENGTH",
                "IP_MODEL_SERVER__RERANK_MAX_LENGTH",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_HOME",
            }
        )
    }
)
