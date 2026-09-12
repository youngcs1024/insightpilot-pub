"""Partial, corrupt and failed partitions remain useful diagnostics, never acceptance."""

from pathlib import Path

import pytest
from coverage import CoverageData

from scripts.ci_coverage import collect
from scripts.ci_evidence import CheckEvidence
from scripts.ci_partitions import PARTITIONS
from scripts.ci_policy import Result


def partition(root: Path, name: str) -> None:
    data = CoverageData(basename=str(root / f".coverage.{name}"))
    files = {}
    for directory in ("core", "services", "agents"):
        path = root / "app" / directory / "sample.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n")
        files[str(path)] = {1}
    data.add_lines(files)
    data.write()


@pytest.mark.parametrize("target", PARTITIONS)
@pytest.mark.parametrize("other", ["missing", "empty", "corrupt", "valid"])
def test_available_partition_is_reported_even_when_other_is_unusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    other: str,
    target: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in PARTITIONS:
        if name != target:
            partition(tmp_path, name)
    if other == "valid":
        partition(tmp_path, target)
    elif other != "missing":
        (tmp_path / f".coverage.{target}").write_bytes(
            b"bad database" if other == "corrupt" else b""
        )
    summary = tmp_path / "summary"
    passed = collect(tmp_path, summary, (Result.SUCCESS, Result.SUCCESS, Result.SUCCESS))
    assert passed is (other == "valid")
    assert (tmp_path / ".coverage.json").is_file()
    assert "app/agents" in summary.read_text()
    for name in PARTITIONS:
        if name != target:
            assert (tmp_path / f".coverage.{name}").is_file()


@pytest.mark.parametrize("target", range(3))
@pytest.mark.parametrize("outcome", [Result.FAILURE, Result.CANCELLED, Result.SKIPPED])
def test_full_measurements_cannot_override_unsuccessful_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: Result,
    target: int,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("unit", "integration", "storage"):
        partition(tmp_path, name)
    summary = tmp_path / "summary"
    outcomes = [Result.SUCCESS, Result.SUCCESS, Result.SUCCESS]
    outcomes[target] = outcome
    assert not collect(tmp_path, summary, tuple(outcomes))
    assert "diagnostic only" in summary.read_text()
    assert "Coverage acceptance: FAIL" in summary.read_text()


def test_no_artifact_cannot_produce_coverage_acceptance(tmp_path: Path) -> None:
    assert not collect(
        tmp_path, tmp_path / "summary", (Result.SUCCESS, Result.SUCCESS, Result.SUCCESS)
    )
    assert not (tmp_path / ".coverage.json").exists()


def test_saved_assessment_separates_thresholds_from_upstream_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("unit", "integration", "storage"):
        partition(tmp_path, name)
    output = tmp_path / "assessment.json"
    assert not collect(
        tmp_path,
        tmp_path / "summary",
        (Result.SUCCESS, Result.FAILURE, Result.SUCCESS),
        output=output,
    )
    value = CheckEvidence.model_validate_json(output.read_text())
    assert value.checks_passed
    assert value.evidence_valid
    assert not value.accepted
