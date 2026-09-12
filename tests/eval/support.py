"""Small, explicitly scripted artifacts for harness tests."""

from datetime import UTC, datetime

from app.core.llm_config import ModelRoleSettings
from app.schemas.mcp import ColumnSpec, QueryResultPayload, SqlValue
from data.seed.contracts import Manifest, Parameters
from evals.harness.contracts import ConfigSnapshot


def payload(rows: list[list[SqlValue]], types: list[str] | None = None) -> QueryResultPayload:
    """Build an executed-result contract with configurable native PostgreSQL types."""
    types = types or ["numeric"]
    return QueryResultPayload(
        executed_sql="SELECT 42",
        limit_applied=True,
        execution_ms=1,
        mcp_call_id="scripted",
        columns=[ColumnSpec(name=f"c{n}", type=t) for n, t in enumerate(types)],
        rows=rows,
        row_count=len(rows),
        result_truncated=False,
    )


def config() -> ConfigSnapshot:
    """Never represent a scripted model score as real quality evidence."""
    return ConfigSnapshot(
        model="scripted-test-model",
        sql_role=ModelRoleSettings(model="scripted-test-model"),
        prompt_hashes={"sql_generate.md": "a" * 64},
        catalog_versions={"gmv": 1},
        catalog_hash="b" * 64,
        schema_hash="c" * 64,
        dataset_hash="d" * 64,
        seed_manifest=Manifest(parameters=Parameters(), files=[]),
        git_sha="e" * 40,
        source_dirty=False,
        reference_time=datetime(2026, 9, 9, tzinfo=UTC),
    )
