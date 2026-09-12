"""Check Compose environment drift without Docker, interpolation or reading secrets."""

import argparse
from pathlib import Path

import yaml  # type: ignore[import-untyped]  # PyYAML ships no typing marker.
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode  # type: ignore[import-untyped]

from app.core.errors import InsightPilotError
from scripts.deployment_contracts import (
    MODEL_SERVER_ENVIRONMENT_KEYS,
    SERVICE_ENVIRONMENT_KEYS,
    environment_issues,
)

ROOT = Path(__file__).resolve().parents[1]
MAX_YAML_DEPTH = 64
ALTERNATE_SOURCES = frozenset({"env_file", "extends", "secrets", "configs"})


class ContractInputError(InsightPilotError):
    """Unsupported input fails closed without including YAML values or parser prose."""


class ServiceInput(BaseModel):
    """Project only environment fields; unrelated Compose settings remain with Compose."""

    model_config = ConfigDict(strict=True, hide_input_in_errors=True)

    environment: dict[str, str] = Field(default_factory=dict)
    env_file: None = None
    extends: None = None
    secrets: None = None
    configs: None = None


class ComposeInput(BaseModel):
    """Includes must be explicitly reviewed before adding more configuration sources."""

    model_config = ConfigDict(strict=True, hide_input_in_errors=True)

    services: dict[str, ServiceInput] = Field(min_length=1)
    include: None = None


def check_yaml(node: Node, ancestors: frozenset[int] = frozenset()) -> None:
    """Reject duplicate keys, merges, recursive aliases and excessive nesting."""
    if id(node) in ancestors or len(ancestors) >= MAX_YAML_DEPTH:
        raise ContractInputError()
    ancestors = ancestors | {id(node)}
    if isinstance(node, MappingNode):
        keys: set[str] = set()
        for key, value in node.value:
            if (
                not isinstance(key, ScalarNode)
                or key.tag != "tag:yaml.org,2002:str"
                or key.value in keys
            ):
                raise ContractInputError()
            keys.add(key.value)
            check_yaml(value, ancestors)
    elif isinstance(node, SequenceNode):
        for value in node.value:
            check_yaml(value, ancestors)


def load_compose(path: Path) -> ComposeInput:
    """No interpolation or environment lookup occurs; diagnostics never echo input."""
    try:
        source = path.read_text(encoding="utf-8")
        node = yaml.compose(source)
        if node is None:
            raise ContractInputError()
        check_yaml(node)
        document = ComposeInput.model_validate(yaml.safe_load(source))
        if "include" in document.model_fields_set:
            raise ContractInputError()
        return document
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, RecursionError) as exc:
        raise ContractInputError() from exc


def document_issues(
    document: ComposeInput, *, partial: bool = False, remote: bool = False
) -> list[str]:
    """Overlay declarations may omit keys, but may not add an unreviewed input source."""
    issues: list[str] = []
    actual = set(document.services)
    contracts = MODEL_SERVER_ENVIRONMENT_KEYS if remote else SERVICE_ENVIRONMENT_KEYS
    expected = set(contracts)
    issues.extend(f"unknown service: {name}" for name in sorted(actual - expected))
    if not partial:
        issues.extend(f"missing service: {name}" for name in sorted(expected - actual))
    for name in sorted(actual & expected):
        service = document.services[name]
        unsupported = ALTERNATE_SOURCES & service.model_fields_set
        issues.extend(f"{name}: unsupported source: {key}" for key in sorted(unsupported))
        keys = environment_issues(name, set(service.environment), partial=partial, remote=remote)
        for category in ("missing", "unexpected", "forbidden"):
            issues.extend(f"{name}: {category} key: {key}" for key in getattr(keys, category))
    return issues


def check_files(base: Path, overlay: Path) -> list[str]:
    """Inspect both inputs independently so overlays cannot conceal base drift."""
    issues: list[str] = []
    for label, path, partial in (("base", base, False), ("development", overlay, True)):
        try:
            document = load_compose(path)
        except ContractInputError:
            issues.append(f"{label}: invalid or unsupported Compose document")
            continue
        issues.extend(f"{label}: {issue}" for issue in document_issues(document, partial=partial))
    return issues


def main() -> None:
    """Fail visibly in quality before waiting for the real container security tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "docker-compose.yml")
    parser.add_argument("--overlay", type=Path, default=ROOT / "docker-compose.dev.yml")
    arguments = parser.parse_args()
    issues = check_files(arguments.base, arguments.overlay)
    try:
        issues.extend(
            document_issues(load_compose(ROOT / "docker-compose.model-server.yml"), remote=True)
        )
    except ContractInputError:
        issues.append("remote: invalid or unsupported Compose document")
    print("Deployment environment contracts: " + ("FAIL" if issues else "PASS"))
    for issue in issues:
        # JSON-style escaping keeps newlines and workflow command syntax inert.
        print(f"- {issue!r}")
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
