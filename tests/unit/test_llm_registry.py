"""Capability evidence must be present, complete and cover every configured model."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config_models import LLMSettings, Settings
from app.core.errors import LlmConfigurationError
from app.core.llm_config import ModelRole, ModelRoleSettings
from app.services.llm.registry import CapabilityReport, ModelRegistry, StructuredTier

ROLE_COUNT, DEFAULT_TOKENS, DEFAULT_TIMEOUT = 6, 2048, 45
OVERRIDE_TEMPERATURE, OVERRIDE_TIMEOUT, OVERRIDE_TOKENS = 0.5, 12, 512
MODEL = "qwen3.6-flash-2026-04-16"


def test_public_capability_resource_contains_only_runtime_fields() -> None:
    path = Path(__file__).resolve().parents[2] / "app/resources/provider_capabilities.json"
    body = json.loads(path.read_text())
    assert set(body) == set(CapabilityReport.model_fields)
    assert CapabilityReport.model_validate(body).recommended_tier == StructuredTier.NATIVE


def settings() -> LLMSettings:
    return LLMSettings(
        base_url="https://provider.invalid/v1", model=MODEL, api_key="synthetic-only"
    )


def test_committed_report_and_six_roles() -> None:
    registry = ModelRegistry.load(settings())
    assert len(ModelRole) == ROLE_COUNT
    for role in ModelRole:
        assert registry.role(role).model == MODEL
        assert registry.role(role).max_tokens == DEFAULT_TOKENS
        assert registry.role(role).temperature == 0
        assert registry.role(role).timeout_s == DEFAULT_TIMEOUT
        assert registry.role(role).fallback_models == []
    assert registry.tier(MODEL) is StructuredTier.NATIVE


def test_role_overrides_and_registry_snapshot() -> None:
    config = settings()
    config.roles[ModelRole.SQL] = ModelRoleSettings(temperature=0.5, timeout_s=12, max_tokens=512)
    registry = ModelRegistry.load(config)
    config.roles[ModelRole.SQL].temperature = 1
    config.roles[ModelRole.SQL].fallback_models.append("unknown")
    assert registry.role(ModelRole.SQL).temperature == OVERRIDE_TEMPERATURE
    assert registry.role(ModelRole.SQL).timeout_s == OVERRIDE_TIMEOUT
    assert registry.role(ModelRole.SQL).max_tokens == OVERRIDE_TOKENS
    copy = registry.role(ModelRole.SQL)
    copy.fallback_models.append("unknown")
    assert registry.role(ModelRole.SQL).fallback_models == []


@pytest.mark.parametrize(
    "body",
    [
        "",
        "{}",
        '{"schema_version":2}',
        '{"schema_version":1,"model":"x","evidence_source":"not_run","execution_complete":false,"recommended_tier":null}',
    ],
)
def test_invalid_or_incomplete_evidence_rejected(tmp_path: Path, body: str) -> None:
    file = tmp_path / "capabilities.json"
    file.write_text(body)
    config = settings()
    config.capabilities_paths = [file]
    with pytest.raises(LlmConfigurationError):
        ModelRegistry.load(config)


def test_missing_evidence_rejected(tmp_path: Path) -> None:
    config = settings()
    config.capabilities_paths = [tmp_path / "missing.json"]
    with pytest.raises(LlmConfigurationError):
        ModelRegistry.load(config)


@pytest.mark.parametrize("fallback", ["unknown", MODEL])
def test_unprobed_or_repeated_model_rejected(fallback: str) -> None:
    config = settings()
    config.roles[ModelRole.SQL] = ModelRoleSettings(fallback_models=[fallback])
    with pytest.raises(LlmConfigurationError):
        ModelRegistry.load(config)


def test_multiple_reports_are_indexed_per_model(tmp_path: Path) -> None:
    file = tmp_path / "second.json"
    file.write_text(
        CapabilityReport(
            schema_version=1,
            evidence_source="live_api",
            execution_complete=True,
            model="second",
            recommended_tier=StructuredTier.TOOL,
        ).model_dump_json()
    )
    config = settings()
    config.capabilities_paths.append(file)
    config.roles[ModelRole.SQL] = ModelRoleSettings(fallback_models=["second"])
    registry = ModelRegistry.load(config)
    assert registry.tier("second") is StructuredTier.TOOL
    assert registry.tier(MODEL) is StructuredTier.NATIVE


@pytest.mark.parametrize(
    ("field", "value"), [("temperature", -1), ("max_tokens", 0), ("timeout_s", 0)]
)
def test_role_bounds(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        ModelRoleSettings.model_validate({field: value})


def test_roles_from_environment_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "IP_LLM__ROLES", '{"sql":{"temperature":0.5,"max_tokens":512,"timeout_s":12}}'
    )
    config = _api_settings()
    assert config.llm.for_role(ModelRole.SQL).temperature == OVERRIDE_TEMPERATURE
    monkeypatch.setenv("IP_LLM__ROLES", '{"typo":{"temperature":0}}')
    with pytest.raises(ValidationError):
        _api_settings()


def _api_settings() -> Settings:
    return Settings(
        _env_file=None,
        llm=settings().model_dump(exclude={"roles"}),
        database={"app_password": "synthetic"},
        security={"jwt_secret": "synthetic-jwt-key-at-least-32-characters"},
        mcp={"auth_token": "synthetic"},
    )
