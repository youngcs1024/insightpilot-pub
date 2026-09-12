"""The publication boundary rejects private material even if it is forcibly staged."""

import gzip

import pytest

from scripts import public_files
from scripts.ci_changes import DiscoveryError
from scripts.public_files import PublicIssue, allowed_path, content_issues


@pytest.mark.parametrize(
    "path",
    [
        "app/main.py",
        "tests/unit/test_auth.py",
        ".github/workflows/ci.yml",
        ".env.api.container.example",
        "app/agents/prompts/system.md",
        "app/services/llm/prompts/structured_json.md",
        "app/resources/provider_capabilities.json",
        "tests/fixtures/seed_traps.json",
        "scripts/data/tiktoken-LICENSE",
    ],
)
def test_required_source_resources_are_allowed(path: str) -> None:
    assert allowed_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "AGENTS.md",
        "README.md",
        "PROJECT_OVERVIEW.md",
        "docs/provider_capabilities.json",
        "data/seed/DESIGN.md",
        "docs/evidence/archive.tar",
        "evals/reports/latest.json",
        "data/seed/out/orders.csv",
        ".env.deployment",
        "app/.env.example",
        "app/notes.md",
        "app/raw.json",
        "tests/logs/output.py",
        "RagMate/backend/app.py",
        "scripts/.agents/config.py",
        "scripts/../docs/example.py",
        "/app/main.py",
    ],
)
def test_private_or_unreviewed_paths_are_rejected(path: str) -> None:
    assert not allowed_path(path)


@pytest.mark.parametrize("compressed", [False, True])
def test_secrets_are_rejected_without_echoing_values(compressed: bool) -> None:
    secret = b"sk-" + b"a" * 32
    content = gzip.compress(secret) if compressed else secret
    name = "scripts/data/cl100k_base.tiktoken.gz" if compressed else "app/main.py"
    issues = content_issues(name, content)
    assert issues == [PublicIssue(path=name, reason="credential-shaped content")]
    assert secret.decode() not in issues[0].model_dump_json()


def test_only_explicit_synthetic_tokens_are_exempt() -> None:
    token = next(iter(public_files.SYNTHETIC_TOKENS)).encode()
    assert not content_issues("tests/example.py", token)
    assert content_issues("tests/example.py", b"sk-" + b"b" * 32)


def test_invalid_compressed_resource_fails() -> None:
    assert content_issues("scripts/data/cl100k_base.tiktoken.gz", b"invalid")


@pytest.mark.parametrize("mode", ["120000", "160000"])
def test_index_rejects_symlinks_and_gitlinks(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    def index_only(_arguments: list[str]) -> bytes:
        return f"{mode} {'a' * 40} 0\tapp/main.py\0".encode()

    monkeypatch.setattr(public_files, "git_bytes", index_only)
    assert public_files.audit_index() == [
        PublicIssue(path="app/main.py", reason="outside public source boundary")
    ]


def test_cli_reports_unavailable_index(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> list[PublicIssue]:
        raise DiscoveryError()

    monkeypatch.setattr(public_files, "audit_index", unavailable)
    assert public_files.main() == 1
