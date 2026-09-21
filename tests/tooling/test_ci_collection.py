"""Exercise real collection, disjoint node IDs and raw workflow prerequisites."""

import json
import shlex
import sys
from itertools import product
from pathlib import Path

import pytest
import yaml

from scripts.ci_collection import CollectionReport
from scripts.ci_policy import Reason, full_plan
from scripts.ci_result import final_result
from tests.tooling.support import ROOT, condition_holds, needs_for

pytest_plugins = ("pytester",)


@pytest.mark.parametrize("missing_import", [False, True])
def test_real_collection_pipeline_preserves_errors_without_executing_tests(
    pytester: pytest.Pytester, missing_import: bool
) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    quality = jobs["quality"]
    step = next(step for step in quality["steps"] if step.get("id") == "collection")
    assert quality["outputs"]["collection_outcome"] == "${{ steps.collection.outcome }}"
    assert "services" not in quality
    pytester.makeini(f"[pytest]\npythonpath = {ROOT}\nmarkers = integration\n")
    directory = pytester.path / "tests" / "integration"
    directory.mkdir(parents=True)
    (directory / "test_import.py").write_text(
        ("" if missing_import else "from contextlib import asynccontextmanager\n")
        + "from pathlib import Path\nimport pytest\n"
        "pytestmark = pytest.mark.integration\n"
        "@asynccontextmanager\nasync def operator_change():\n    yield\n"
        "@pytest.fixture(autouse=True)\ndef setup():\n    Path('fixture-ran').touch()\n"
        "def test_body():\n    Path('body-ran').touch()\n"
    )
    script = step["run"].replace(".venv/bin/pytest", shlex.quote(sys.executable) + " -m pytest")
    result = pytester.run("/bin/bash", "-e", "-c", script, timeout=60)
    expected = pytest.ExitCode.INTERRUPTED if missing_import else pytest.ExitCode.OK
    assert result.ret == expected
    evidence = CollectionReport.model_validate_json(
        (pytester.path / "quality-collection.json").read_text()
    )
    assert evidence.raw_exit == expected
    assert evidence.accepted is (not missing_import)
    assert bool(evidence.failed_nodes) is missing_import
    assert "NameError" not in evidence.model_dump_json()
    log = (pytester.path / "quality-collection.log").read_text()
    assert "test_import.py" in log
    if missing_import:
        assert "NameError" in log
    assert not (pytester.path / "fixture-ran").exists()
    assert not (pytester.path / "body-ran").exists()
    assert not list(pytester.path.glob(".coverage*"))


def test_real_repository_partitions_are_complete_and_disjoint(pytester: pytest.Pytester) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    selections = {"all": "not gpu and not external"}
    for name in ("unit", "integration", "storage"):
        step = next(step for step in jobs[name]["steps"] if step.get("id") == "tests")
        arguments = shlex.split(step["run"].splitlines()[1].split("2>&1", 1)[0])
        selections[name] = arguments[arguments.index("-m") + 1]
        paths = [
            arg for arg in arguments[1:] if not arg.startswith("-") and arg != selections[name]
        ]
        assert paths in ([], ["tests"])
    collected = {}
    for name, selection in selections.items():
        result = pytester.runpytest_subprocess(
            str(ROOT / "tests"),
            "-c",
            str(ROOT / "pyproject.toml"),
            "--collect-only",
            "--no-cov",
            "-q",
            "-m",
            selection,
            timeout=60,
        )
        assert result.ret == pytest.ExitCode.OK
        identities = [
            line for line in result.outlines if line.startswith("tests/") and "::" in line
        ]
        assert identities
        assert len(identities) == len(set(identities))
        collected[name] = set(identities)
    for left, right in (("unit", "integration"), ("unit", "storage"), ("integration", "storage")):
        assert collected[left].isdisjoint(collected[right])
    assert collected["unit"] | collected["integration"] | collected["storage"] == collected["all"]
    chaos = "tests/e2e/test_chaos.py::test_mcp_killed_midturn"
    assert chaos in collected["storage"]
    assert chaos not in collected["unit"] | collected["integration"]
    for directory in ("api", "agents", "tooling", "security", "unit", "integration", "e2e"):
        assert any(node.startswith(f"tests/{directory}/") for node in collected["all"])


@pytest.mark.parametrize(
    ("collection", "quality", "cancelled", "runs"),
    [
        ("success", "success", False, True),
        ("success", "failure", False, True),
        ("failure", "failure", False, False),
        ("cancelled", "cancelled", False, False),
        ("skipped", "failure", False, False),
        ("", "failure", False, False),
        ("invalid", "failure", False, False),
        ("success", "cancelled", False, False),
        ("success", "success", True, False),
    ],
)
def test_actual_job_conditions_use_collection_not_quality_success(
    tmp_path: Path, collection: str, quality: str, cancelled: bool, runs: bool
) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    values = {
        "needs.changes.result": "success",
        "needs.changes.outputs.checks": "true",
        "needs.quality.result": quality,
        "needs.quality.outputs.collection_outcome": collection,
    }
    for name in ("unit", "integration", "storage"):
        assert condition_holds(jobs[name]["if"], values, cancelled=cancelled) is runs
    plan = full_plan("a" * 40, Reason.MANUAL)
    needs = needs_for(plan)
    needs["quality"] = {"result": quality, "outputs": {"collection_outcome": collection}}
    for name in ("unit", "integration", "storage"):
        needs[name]["result"] = "success" if runs else "skipped"
    summary = tmp_path / "summary"
    assert final_result(
        plan.model_dump_json(), json.dumps(needs), summary, tested_sha=plan.tested_sha
    ) is (runs and quality == "success")
    if collection == "failure":
        assert "collection_error" in summary.read_text()
        assert "not_run/upstream_blocked" in summary.read_text()


@pytest.mark.parametrize("ids", list(product(("", "101"), repeat=3)))
def test_only_uploaded_coverage_is_downloaded(ids: tuple[str, str, str]) -> None:
    steps = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]["coverage"][
        "steps"
    ]
    values = {"steps.setup.outcome": "success"}
    for name, artifact_id in zip(("unit", "integration", "storage"), ids, strict=True):
        values[f"needs.{name}.outputs.coverage_artifact_id"] = artifact_id
        download = next(step for step in steps if step.get("id") == f"{name}_download")
        assert condition_holds(download["if"], values) is bool(artifact_id)
        explanation = next(
            step for step in steps if step.get("name") == f"Explain absent {name} coverage"
        )
        assert condition_holds(explanation["if"], values) is (not artifact_id)


@pytest.mark.parametrize("valid", [False, True])
def test_storage_marker_contract_precedes_deselection(
    pytester: pytest.Pytester, valid: bool
) -> None:
    pytester.makeini("[pytest]\nmarkers =\n integration\n storage\n")
    pytester.makeconftest((ROOT / "tests/collection_support.py").read_text())
    pytester.makepyfile(
        "import pytest\n@pytest.mark.storage\n"
        + ("@pytest.mark.integration\n" if valid else "")
        + "def test_storage():\n    raise AssertionError('must not execute')\n"
    )
    result = pytester.runpytest_subprocess("--collect-only", "-m", "not storage")
    assert result.ret == (
        pytest.ExitCode.NO_TESTS_COLLECTED if valid else pytest.ExitCode.USAGE_ERROR
    )
    if not valid:
        assert "storage requires integration" in "\n".join(result.outlines + result.errlines)


def test_dedicated_modules_are_excluded_before_import(pytester: pytest.Pytester) -> None:
    pytester.makeconftest("collect_ignore = ['gpu', 'live']")
    for name in ("gpu", "live"):
        directory = pytester.path / name
        directory.mkdir()
        (directory / "test_unavailable.py").write_text("import unavailable_dedicated_runtime\n")
    pytester.makepyfile("def test_ordinary():\n    pass\n")
    result = pytester.runpytest_subprocess("--collect-only")
    assert result.ret == pytest.ExitCode.OK
    assert 'collect_ignore = ["gpu", "live"]' in (ROOT / "tests/conftest.py").read_text()


def test_importlib_keeps_same_name_modules_and_shared_fixtures(pytester: pytest.Pytester) -> None:
    pytester.makeini(f"[pytest]\npythonpath = . {ROOT}\naddopts = --import-mode=importlib\n")
    pytester.makepyfile(shared_helper="VALUE = 17")
    pytester.makeconftest(
        "import pytest\nfrom shared_helper import VALUE\n"
        "@pytest.fixture\ndef shared():\n    return VALUE\n"
    )
    for directory, identity in (("unit", "unit"), ("integration", "integration")):
        path = pytester.path / directory
        path.mkdir()
        (path / "test_same.py").write_text(
            f"IDENTITY = {identity!r}\n"
            f"def test_identity(shared):\n    assert IDENTITY == {identity!r}\n"
            "    assert shared == 17\n"
        )
    result = pytester.runpytest_subprocess("-q", timeout=60)
    result.assert_outcomes(passed=2)
