"""Live composition metadata contracts with injected model-free services."""

import json
from pathlib import Path

import pytest

from data.seed.contracts import TABLE_NAMES, FileManifest, Manifest, Parameters
from evals.harness.contracts import EvaluationError, Options
from evals.harness.runtime import fresh, snapshot, verify_seed_counts
from tests.agents.support import context
from tests.eval.support import payload


def manifest() -> Manifest:
    return Manifest(
        parameters=Parameters(),
        files=[FileManifest(table=table, rows=1, sha256="a" * 64) for table in TABLE_NAMES],
    )


async def test_snapshot_checks_seed_and_never_exports_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(manifest().model_dump_json())
    ctx = context(mcp_results=[payload([[1] * len(TABLE_NAMES)], ["int8"] * len(TABLE_NAMES))])
    monkeypatch.setattr(
        "evals.harness.runtime.git_value", lambda *args: "a" * 40 if args[0] == "rev-parse" else ""
    )
    result = await snapshot(ctx, Options(seed_manifest=path))
    assert result.model == "test"
    assert result.catalog_versions["gmv"] == 1
    assert set(result.prompt_hashes) == {
        "metric_intent.md",
        "sql_generate.md",
        "sql_correct.md",
        "schema.md",
        "structured_json.md",
        "structured_repair.md",
    }
    serialized = result.model_dump_json()
    assert all(
        secret not in serialized
        for secret in ("test-key", "test-password", "test-mcp", "example.invalid")
    )
    assert json.loads(serialized)["git_sha"] == "a" * 40


async def test_seed_mismatch_rejects_live_evidence() -> None:
    ctx = context(mcp_results=[payload([[0] * len(TABLE_NAMES)], ["int8"] * len(TABLE_NAMES))])
    with pytest.raises(EvaluationError):
        await verify_seed_counts(ctx, manifest())


async def test_custom_seed_parameters_rejected_before_mcp(tmp_path: Path) -> None:
    value = manifest()
    value.parameters.seed = 7
    path = tmp_path / "manifest.json"
    path.write_text(value.model_dump_json())
    ctx = context()
    with pytest.raises(EvaluationError):
        await snapshot(ctx, Options(seed_manifest=path))
    assert not ctx.mcp.calls


def test_independent_attempts_share_services_not_deadlines_or_ids() -> None:
    original = context()
    first = fresh(original)
    second = fresh(original)
    assert first.identity != second.identity
    assert first.deadline is not second.deadline
    assert first.mcp is second.mcp is original.mcp
    assert first.now == second.now
