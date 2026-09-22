"""Independent test-deployment isolation contract; no Docker or secret loading."""

from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from scripts.deployment_contracts import (
    API_ENVIRONMENT_KEYS,
    SERVICE_ENVIRONMENT_KEYS,
    forbidden_api_keys,
)

E2E_ENVIRONMENT_KEYS = {
    **{
        name: keys
        for name, keys in SERVICE_ENVIRONMENT_KEYS.items()
        if name in {"postgres", "migrate", "seed", "mcp", "etcd", "minio", "milvus"}
    },
    "api": API_ENVIRONMENT_KEYS | {"IP_RATE_LIMITS"},
    "inference": frozenset(),
    "ingest": frozenset(
        {
            "IP_DATABASE__HOST",
            "IP_DATABASE__PORT",
            "IP_DATABASE__APP_PASSWORD",
            "IP_RETRIEVAL",
            "IP_MODEL_RUNTIME",
        }
    ),
}
MODEL_CONFIG = '{"base_url":"http://inference:8100","auth_token":"e2e-synthetic-only"}'
RETRIEVAL_CONFIG = '{"enabled":true,"milvus":{"uri":"http://milvus:19530"}}'


class Service(BaseModel):
    """Reject alternate credentials and escape hatches, while allowing image settings."""

    model_config = ConfigDict(strict=True, hide_input_in_errors=True)
    environment: dict[str, str] = Field(default_factory=dict)
    networks: list[str]
    volumes: list[str] = Field(default_factory=list)
    ports: list[str] = Field(default_factory=list)
    env_file: None = None
    extends: None = None
    secrets: None = None
    configs: None = None
    privileged: bool = False
    network_mode: None = None
    container_name: None = None
    gpus: None = None
    devices: list[str] = Field(default_factory=list)


class Composition(BaseModel):
    """All named resources must be local to one fixture-owned Compose project."""

    model_config = ConfigDict(strict=True, hide_input_in_errors=True)
    services: dict[str, Service]
    networks: dict[str, dict[str, JsonValue]]
    volumes: dict[str, dict[str, JsonValue]]


def isolation_issues(path: Path) -> list[str]:
    """Called after the shared YAML duplicate/include/alias validation."""
    try:
        document = Composition.model_validate(yaml.safe_load(path.read_text()))
    except (OSError, ValidationError, yaml.YAMLError):
        return ["e2e: invalid isolation configuration"]
    issues = []
    if set(document.services) != set(E2E_ENVIRONMENT_KEYS):
        issues.append("e2e: unexpected service set")
    if document.networks != {"backend": {"internal": True}, "egress": {}}:
        issues.append("e2e: networks must be isolated")
    if document.volumes != {name: {} for name in ("pgdata", "etcddata", "miniodata", "milvusdata")}:
        issues.append("e2e: volumes must be project-owned")
    for name, service in document.services.items():
        issues.extend(service_issues(name, service))
    return issues


def service_issues(name: str, service: Service) -> list[str]:
    """Compare names and fixed test endpoints, never print environment values."""
    issues = []
    if set(service.environment) != E2E_ENVIRONMENT_KEYS.get(name, frozenset()):
        issues.append(f"e2e: {name}: environment keys differ")
    if service.privileged or service.devices or set(service.networks) - {"backend", "egress"}:
        issues.append(f"e2e: {name}: unsafe isolation")
    expected_volumes = {
        "postgres": [
            "./docker/postgres/init:/docker-entrypoint-initdb.d:ro",
            "pgdata:/var/lib/postgresql/data",
        ],
        "etcd": ["etcddata:/etcd"],
        "minio": ["miniodata:/minio_data"],
        "milvus": ["milvusdata:/var/lib/milvus"],
    }
    if service.volumes != expected_volumes.get(name, []):
        issues.append(f"e2e: {name}: unreviewed mount")
    expected_ports = {
        "api": ["127.0.0.1:${IP_API_HOST_PORT:?required}:8000"],
        "inference": ["127.0.0.1:${IP_E2E_INFERENCE_PORT:?required}:8100"],
    }
    if service.ports != expected_ports.get(name, []):
        issues.append(f"e2e: {name}: unreviewed host port")
    if name == "api":
        if forbidden_api_keys(set(service.environment)):
            issues.append("e2e: api: forbidden credentials")
        expected = {
            "IP_LLM__BASE_URL": "http://inference:8100/v1",
            "IP_OBSERVABILITY": '{"langfuse_enabled":false,"log_format":"json"}',
            "IP_LLM__API_KEY": "e2e-synthetic-only",
        }
        if any(service.environment.get(key) != value for key, value in expected.items()):
            issues.append("e2e: api: external inference or tracing")
    if name in {"api", "ingest"} and (
        service.environment.get("IP_MODEL_RUNTIME") != MODEL_CONFIG
        or service.environment.get("IP_RETRIEVAL") != RETRIEVAL_CONFIG
    ):
        issues.append(f"e2e: {name}: real stores and substitute inference required")
    return issues
