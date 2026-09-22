"""Audit the Git index against the public source boundary without reading local secrets."""

import gzip
import re
from pathlib import PurePosixPath

from pydantic import BaseModel

from scripts.ci_changes import DiscoveryError, git_bytes

SOURCE_ROOTS = frozenset(
    {
        "app",
        "tests",
        "scripts",
        "alembic",
        "data",
        "evals",
        "mcp_server",
        "model_runtime",
        "model_tunnel",
        "spikes",
    }
)
ROOT_FILES = frozenset(
    {
        ".gitignore",
        ".dockerignore",
        ".python-version",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        "alembic.ini",
        "docker-compose.yml",
        "docker-compose.dev.yml",
        "docker-compose.e2e.yml",
        "docker-compose.model-server.yml",
    }
)
RESOURCE_FILES = frozenset(
    {
        "app/core/limiter.env",
        "app/resources/provider_capabilities.json",
        "tests/fixtures/seed_traps.json",
        "tests/gpu/pytest.ini",
        "alembic/app/data/0006_schema_metadata.json",
        "alembic/app/data/0007_metric_catalog.json",
        "data/seed/metrics.yaml",
        "data/seed/schema_metadata.yaml",
        "evals/datasets/nl2sql/cases.yaml",
        "evals/datasets/routing/cases.yaml",
        "evals/datasets/routing/selected.yaml",
        "evals/datasets/retrieval/queries.yaml",
        "evals/datasets/retrieval/judgments.yaml",
        "evals/datasets/retrieval/development.yaml",
        "evals/datasets/retrieval/frozen.yaml",
        "evals/datasets/retrieval/selected.yaml",
        "scripts/data/cl100k_base.tiktoken.gz",
        "scripts/data/tiktoken-LICENSE",
        "spikes/mcp_v2/example.env",
        "spikes/provider/.env.example",
        "spikes/capacity/ssh_config.example",
        "spikes/capacity/compose.yml",
        "alembic/app/script.py.mako",
        "alembic/business/script.py.mako",
        ".github/actions/setup-python/action.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/docker.yml",
        "docker/storage-volume/.initialized",
    }
)
PROMPT_ROOTS = frozenset({"app/agents/prompts", "app/services/llm/prompts"})
PRIVATE_PARTS = frozenset(
    {
        "docs",
        "reports",
        "out",
        "logs",
        "evidence",
        ".git",
        ".agents",
        ".codex",
        ".venv",
        "__pycache__",
    }
)
SYNTHETIC_TOKENS = frozenset({"sk-public-boundary-synthetic-only-000000"})
SECRET_PATTERNS = (
    re.compile(rb"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})"),
    re.compile(rb"\b(?:github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|hf_[A-Za-z0-9]{25,})"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
)


class PublicIssue(BaseModel):
    """A safe finding that never includes file content or credential values."""

    path: str
    reason: str


def allowed_path(name: str) -> bool:
    """Allow source roots and explicit resources; private areas always win."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or PRIVATE_PARTS.intersection(path.parts):
        return False
    if name in ROOT_FILES or name in RESOURCE_FILES:
        return True
    if path.parent == PurePosixPath(".") and re.fullmatch(r"\.env(?:\.[a-z-]+)*\.example", name):
        return True
    if path.suffix == ".md":
        return path.parent.as_posix() in PROMPT_ROOTS or _corpus_resource(path)
    if not path.parts or any(part.startswith(".") for part in path.parts):
        return False
    return _corpus_resource(path) or _source_path(path)


def _corpus_resource(path: PurePosixPath) -> bool:
    return (
        path.parts[:2] == ("data", "corpus")
        and not any(part.startswith(".") for part in path.parts)
        and (
            path.as_posix() == "data/corpus/MANIFEST.yaml"
            or path.suffix in {".md", ".xlsx", ".pdf"}
            or path.name.endswith((".xlsx.meta.yaml", ".pdf.meta.yaml"))
        )
    )


def _source_path(path: PurePosixPath) -> bool:
    if path.parts[0] == "docker":
        return path.name.startswith("Dockerfile.") or path.suffix in {".sql", ".sh"}
    if path.parts[0] not in SOURCE_ROOTS:
        return False
    return (
        path.suffix in {".py", ".sh"}
        or (
            path.parent.as_posix() in {"mcp_server", "model_runtime"}
            and path.name in {"pyproject.toml", "uv.lock"}
        )
        or (path.parent.as_posix() == "spikes/capacity" and path.name.startswith("Dockerfile."))
    )


def content_issues(name: str, content: bytes) -> list[PublicIssue]:
    """Check explicit compressed resources and recognizable credential formats."""
    if name.endswith(".gz"):
        try:
            content = gzip.decompress(content)
        except (OSError, EOFError):
            return [PublicIssue(path=name, reason="invalid compressed resource")]
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(content):
            if match.group().decode("ascii") not in SYNTHETIC_TOKENS:
                return [PublicIssue(path=name, reason="credential-shaped content")]
    return []


def audit_index() -> list[PublicIssue]:
    """Inspect staged blobs, rejecting symlinks, gitlinks and unresolved index entries."""
    issues: list[PublicIssue] = []
    for entry in git_bytes(["ls-files", "--stage", "-z"]).split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        name = raw_name.decode("utf-8")
        if mode not in {"100644", "100755"} or stage != "0" or not allowed_path(name):
            issues.append(PublicIssue(path=name, reason="outside public source boundary"))
            continue
        issues.extend(content_issues(name, git_bytes(["cat-file", "blob", oid])))
    return issues


def main() -> int:
    """Return failure for unreviewed paths, recognizable secrets or unreadable index data."""
    try:
        issues = audit_index()
    except (DiscoveryError, UnicodeError):
        print("Public source index could not be audited.")
        return 1
    for issue in issues:
        print(issue.model_dump_json())
    if issues:
        return 1
    print("Public source boundary: all indexed files are allowed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
