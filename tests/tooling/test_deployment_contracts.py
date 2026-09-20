"""Independent policy catches Compose drift before real container acceptance."""

from pathlib import Path

import pytest
import yaml

from scripts.check_deployment_contracts import (
    ContractInputError,
    check_files,
    document_issues,
    load_compose,
    main,
)
from scripts.deployment_contracts import (
    API_ENVIRONMENT_KEYS,
    SERVICE_ENVIRONMENT_KEYS,
    environment_issues,
    forbidden_api_keys,
)

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_VALUE = "private-value-do-not-print"


@pytest.fixture
def compose_files(tmp_path: Path) -> tuple[Path, Path]:
    base, overlay = tmp_path / "base.yml", tmp_path / "dev.yml"
    base.write_text((ROOT / "docker-compose.yml").read_text())
    overlay.write_text((ROOT / "docker-compose.dev.yml").read_text())
    return base, overlay


def test_repository_declarations_match_independent_contract() -> None:
    assert check_files(ROOT / "docker-compose.yml", ROOT / "docker-compose.dev.yml") == []
    assert "IP_DATA_AGENT" in API_ENVIRONMENT_KEYS
    assert "IP_ROUTER" in API_ENVIRONMENT_KEYS


@pytest.mark.parametrize(
    "service", [name for name, keys in SERVICE_ENVIRONMENT_KEYS.items() if keys]
)
def test_missing_and_extra_keys_are_independently_reported(service: str) -> None:
    expected = set(SERVICE_ENVIRONMENT_KEYS[service])
    missing = min(expected)
    keys = (expected - {missing}) | {"UNREVIEWED_CONFIG"}
    issues = environment_issues(service, keys)
    assert issues.missing == [missing]
    assert issues.unexpected == ["UNREVIEWED_CONFIG"]
    assert not issues.passed


@pytest.mark.parametrize(
    "credential",
    [
        "IP_BUSINESS__PASSWORD",
        "IP_MIGRATION",
        "IP_BOOTSTRAP__APP_PASSWORD",
        "IP_SEED",
        "IP_OPERATOR_BIZ_URL",
        "PGPASSWORD",
        "DATABASE_URL",
        "POSTGRES_PASSWORD",
        "IP_MINIO__PASSWORD",
        "MINIO_ROOT_PASSWORD",
    ],
)
def test_api_credential_prohibition_does_not_depend_on_allowlist(credential: str) -> None:
    assert forbidden_api_keys({credential}) == [credential]
    assert environment_issues("api", set(API_ENVIRONMENT_KEYS) | {credential}).forbidden == [
        credential
    ]


@pytest.mark.parametrize("service", ["postgres", "mcp", "migrate", "seed"])
def test_api_only_configuration_rejected_for_other_services(service: str) -> None:
    issues = environment_issues(service, set(SERVICE_ENVIRONMENT_KEYS[service]) | {"IP_DATA_AGENT"})
    assert issues.unexpected == ["IP_DATA_AGENT"]


@pytest.mark.parametrize("which", ["base", "overlay"])
def test_environment_additions_cannot_bypass_policy(
    compose_files: tuple[Path, Path], which: str
) -> None:
    base, overlay = compose_files
    path = base if which == "base" else overlay
    document = yaml.safe_load(path.read_text())
    document["services"].setdefault("api", {}).setdefault("environment", {})[
        "IP_BUSINESS__PASSWORD"
    ] = PRIVATE_VALUE
    path.write_text(yaml.safe_dump(document))
    issues = check_files(base, overlay)
    assert any("api: forbidden key: IP_BUSINESS__PASSWORD" in issue for issue in issues)
    assert PRIVATE_VALUE not in str(issues)


def test_base_missing_key_cannot_be_hidden_by_overlay(compose_files: tuple[Path, Path]) -> None:
    base, overlay = compose_files
    document = yaml.safe_load(base.read_text())
    document["services"]["api"]["environment"].pop("IP_DATA_AGENT")
    base.write_text(yaml.safe_dump(document))
    overlay.write_text('services: {api: {environment: {IP_DATA_AGENT: "{}"}}}')
    assert "base: api: missing key: IP_DATA_AGENT" in check_files(base, overlay)


def test_overlay_can_override_an_explicit_existing_key(compose_files: tuple[Path, Path]) -> None:
    base, overlay = compose_files
    overlay.write_text('services: {api: {environment: {IP_DATA_AGENT: "{}"}}}')
    assert check_files(base, overlay) == []


@pytest.mark.parametrize("field", ["env_file", "extends", "secrets", "configs"])
def test_alternate_sources_are_rejected_even_if_null(
    compose_files: tuple[Path, Path], field: str
) -> None:
    base, overlay = compose_files
    overlay.write_text(f"services: {{api: {{{field}: null}}}}")
    assert any(f"api: unsupported source: {field}" in item for item in check_files(base, overlay))


@pytest.mark.parametrize(
    "source",
    [
        "",
        "services: [",
        "services: []",
        "services: {}",
        "services: {api: null}",
        "services: {api: {environment: [IP_DATA_AGENT=private-value]}}",
        "services: {api: {environment: {IP_DATA_AGENT: null}}}",
        "services: {api: {environment: {IP_DATA_AGENT: false}}}",
        "services: {api: {env_file: private-file}}",
        "include: private-file\nservices: {api: {}}",
        "include: null\nservices: {api: {}}",
        "services: {api: {}, api: {environment: {IP_BUSINESS__PASSWORD: private}}}",
        "services: {api: {environment: {IP_DATA_AGENT: '{}', IP_DATA_AGENT: '{}'}}}",
        "services: {api: {<<: {environment: {IP_BUSINESS__PASSWORD: private}}}}",
        "services: &recursive {api: *recursive}",
        "services: {api: {environment: !reset {}}}",
    ],
)
def test_unsupported_yaml_has_no_raw_value_diagnostics(tmp_path: Path, source: str) -> None:
    path = tmp_path / "invalid.yml"
    path.write_text(source)
    with pytest.raises(ContractInputError) as caught:
        load_compose(path)
    assert "private" not in str(caught.value)


def test_missing_input_and_unknown_service_fail_closed(compose_files: tuple[Path, Path]) -> None:
    base, overlay = compose_files
    assert check_files(base.with_name("missing.yml"), overlay) == [
        "base: invalid or unsupported Compose document"
    ]
    overlay.write_text("services: {unreviewed: {}}")
    assert check_files(base, overlay) == ["development: unknown service: unreviewed"]


def test_base_requires_every_current_service(compose_files: tuple[Path, Path]) -> None:
    _, overlay = compose_files
    issues = document_issues(load_compose(overlay))
    assert "missing service: api" in issues


@pytest.mark.parametrize("valid", [True, False])
def test_quality_entrypoint_exit_status_and_safe_diagnostics(
    compose_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    valid: bool,
) -> None:
    base, overlay = compose_files
    if not valid:
        overlay.write_text(
            'services: {api: {environment: {IP_BUSINESS__PASSWORD: "private-value"}}}'
        )
    monkeypatch.setattr(
        "sys.argv", ["check-contracts", "--base", str(base), "--overlay", str(overlay)]
    )
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == (0 if valid else 1)
    output = capsys.readouterr().out
    assert ("PASS" if valid else "FAIL") in output
    assert "private-value" not in output
