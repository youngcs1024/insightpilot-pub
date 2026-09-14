"""Storage resource evidence cannot be overwritten or bypassed by green job labels."""

from pathlib import Path

import pytest

from scripts.ci_diagnostics import DiagnosticRecord, evidence_categories, render, successful_record
from scripts.ci_storage import CleanupState, StorageAssessment, assess_storage
from tests.storage_evidence_support import complete_stack

UPLOAD_MARGIN_MINUTES = 5


@pytest.mark.parametrize(
    "damage",
    ["missing", "malformed", "stale", "duplicate", "pending", "oom", "no_final", "stopped"],
)
def test_resource_evidence_rejects_incomplete_runs(tmp_path: Path, damage: str) -> None:
    stack = complete_stack()
    expected = stack.identity.model_copy()
    if damage == "stale":
        stack.identity.run_attempt = "9"
    elif damage == "pending":
        stack.collections[0].state = CleanupState.PENDING
    elif damage == "oom":
        stack.samples[-1].containers[0].oom_killed = True
    elif damage == "no_final":
        stack.samples.pop()
    elif damage == "stopped":
        stack.samples[-1].containers[0].running = False
    if damage != "missing":
        (tmp_path / "lifecycle.json").write_text(
            "{" if damage == "malformed" else stack.model_dump_json()
        )
    if damage == "duplicate":
        (tmp_path / "duplicate").mkdir()
        (tmp_path / "duplicate/lifecycle.json").write_text(stack.model_dump_json())
    assert not assess_storage(tmp_path, expected).accepted


def test_independent_stacks_and_green_storage_require_complete_evidence(tmp_path: Path) -> None:
    for suffix in ("a", "b"):
        path = tmp_path / suffix
        path.mkdir()
        (path / "lifecycle.json").write_text(complete_stack(suffix * 12).model_dump_json())
    assessed = assess_storage(tmp_path, complete_stack().identity)
    assert assessed.accepted
    value = DiagnosticRecord(
        **complete_stack().identity.model_dump(),
        stage="storage",
        steps={name: {"outcome": "success"} for name in ("setup", "tests", "test_report")},
        report_valid=True,
        counts={"passed": 1},
    )
    assert not successful_record(value)
    value.storage = assessed
    assert successful_record(value)
    assert "unresolved=0" in render(value)
    value.storage.stacks[0].collections[0].state = CleanupState.FAILED
    assert not successful_record(value)
    assert "cleanup_failure" in evidence_categories(value)
    value.storage = StorageAssessment()
    assert "artifact_error" in evidence_categories(value)


def test_storage_workflow_keeps_budget_and_identity() -> None:
    import yaml  # noqa: PLC0415 -- workflow reader, no runtime dependency.

    root = Path(__file__).resolve().parents[2]
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"]
    storage = jobs["storage"]
    tests = next(step for step in storage["steps"] if step.get("id") == "tests")
    assert storage["timeout-minutes"] - tests["timeout-minutes"] >= UPLOAD_MARGIN_MINUTES
    for name in ("TESTED_SHA", "RUN_ID", "RUN_ATTEMPT"):
        assert "CI_STORAGE_" + name in tests["env"]
    assert any(
        "--storage-resources milvus-evidence" in step.get("run", "") for step in storage["steps"]
    )
    assert "storage_tracking.pytest_runtest_makereport" in (root / "tests/conftest.py").read_text()
    assert not any(
        "from tests.milvus_support import milvus_stack" in path.read_text()
        for path in (root / "tests/integration").glob("*.py")
    )


@pytest.mark.parametrize("foreign", [False, True])
def test_resource_snapshot_uses_exact_compose_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign: bool
) -> None:
    import json  # noqa: PLC0415 -- test adapter.

    from scripts.ci_process import CommandEvidence, CommandRecorder, CommandState  # noqa: PLC0415
    from scripts.deployment import DeploymentError  # noqa: PLC0415
    from tests.milvus_support import MilvusStack  # noqa: PLC0415

    evidence = complete_stack()
    evidence.samples.clear()
    recorder = CommandRecorder(directory=tmp_path / "commands", cwd=tmp_path, environment={})
    stack = MilvusStack(
        uri="http://localhost:19530",
        command=["docker", "compose", "-p", evidence.project],
        recorder=recorder,
        evidence=evidence,
        directory=tmp_path,
    )

    def run(_self: CommandRecorder, stage: str, command: list[str]) -> CommandEvidence:
        if stage == "resource-id":
            assert command[: len(stack.command)] == stack.command
            output = "owned-container"
        elif stage == "resource-state":
            assert command[-1] == "owned-container"
            output = json.dumps(
                {
                    "project": "foreign" if foreign else evidence.project,
                    "running": True,
                    "oom_killed": False,
                    "restarts": 1,
                    "limit_bytes": 3 * 1024**3,
                }
            )
        else:
            assert command[-1] == "owned-container"
            output = "100MiB / 3GiB"
        return CommandEvidence(stage=stage, state=CommandState.SUCCESS, elapsed_s=0, stdout=output)

    monkeypatch.setattr(CommandRecorder, "run", run)
    if foreign:
        with pytest.raises(DeploymentError, match="another project"):
            stack.sample("startup")
        assert not evidence.samples
    else:
        stack.sample("startup")
        assert {row.service for row in evidence.samples[0].containers} == {
            "milvus",
            "etcd",
            "minio",
        }
        assert (tmp_path / "lifecycle.json").is_file()
