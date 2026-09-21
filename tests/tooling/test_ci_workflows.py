"""Guard the pipeline's blocking stages and complete, disjoint test selection."""

import re
import subprocess
from pathlib import Path

import yaml

from scripts.ci_result import QUALITY_CHECKS, QUALITY_COMMANDS
from tests.tooling.support import condition_holds

ROOT = Path(__file__).resolve().parents[2]
TEST_PARTITIONS = ("unit", "integration", "storage")


def test_ci_dependencies_parallelize_work_and_always_join_results() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    assert set(jobs) == {
        "changes",
        "quality",
        "unit",
        "integration",
        "storage",
        "coverage",
        "build",
        "ci-result",
    }
    assert jobs["quality"]["needs"] == "changes"
    assert jobs["build"]["needs"] == "changes"
    for name in TEST_PARTITIONS:
        assert set(jobs[name]["needs"]) == {"changes", "quality"}
    assert set(jobs["coverage"]["needs"]) == {"changes", "unit", "integration", "storage"}
    assert set(jobs["ci-result"]["needs"]) == set(jobs) - {"ci-result"}
    assert jobs["ci-result"]["if"] == "always()"
    assert jobs["ci-result"]["name"] == "ci-result"
    assert jobs["build"]["uses"] == "./.github/workflows/docker.yml"
    assert jobs["build"]["if"] == "needs.changes.outputs.images != '[]'"
    assert jobs["build"]["with"]["images"] == "${{ needs.changes.outputs.images }}"
    assert workflow["permissions"] == {"contents": "read"}
    assert jobs["changes"]["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert "paths-ignore" not in (ROOT / ".github/workflows/ci.yml").read_text()
    selection_steps = jobs["changes"]["steps"]
    boundary = next(step for step in selection_steps if step.get("id") == "public_files")
    selection = next(step for step in selection_steps if step.get("id") == "select")
    assert "scripts.public_files" in boundary["run"]
    assert selection_steps.index(boundary) < selection_steps.index(selection)
    assert "if" not in boundary


def test_ci_test_partitions_and_independent_coverage_gate() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    source = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "pytest tests/integration/test_migrations.py" not in source
    assert "scripts.ci_result migrations" in source
    assert "make install-mcp ENV=test" in source
    assert 'INSIGHTPILOT_TEST_REQUIRE_DOCKER: "true"' in source
    assert "docker/postgres/init/01_bootstrap.sql" in source
    assert "pgvector/pgvector:0.8.5-pg16" in source
    coverage = jobs["coverage"]["steps"]
    for name in TEST_PARTITIONS:
        download = next(step for step in coverage if step.get("id") == f"{name}_download")
        assert download["with"]["artifact-ids"] == (
            "${{ needs." + name + ".outputs.coverage_artifact_id }}"
        )
        assert download["with"]["path"] == "."
        assert jobs[name]["outputs"]["coverage_artifact_id"] == (
            "${{ steps.coverage_upload.outputs.artifact-id }}"
        )

    commands = "\n".join(step.get("run", "") for step in coverage)
    assert "scripts.ci_coverage" in commands
    assert '--unit "$UNIT_OUTCOME" --integration "$INTEGRATION_OUTCOME"' in commands
    quality = jobs["quality"]["steps"]
    checks = [step for step in quality if step.get("id") in QUALITY_CHECKS]
    assert {step["id"] for step in checks} == set(QUALITY_CHECKS)
    for step in checks:
        assert QUALITY_COMMANDS[step["id"]] in step["run"]
        assert condition_holds(step["if"], {"steps.setup.outcome": "success"})
        assert not condition_holds(step["if"], {"steps.setup.outcome": "failure"})
        assert not condition_holds(step["if"], {"steps.setup.outcome": "success"}, cancelled=True)
    assert sum(step.get("id") == "setup" for step in quality) == 1
    report = next(step for step in quality if "scripts.ci_result quality" in step.get("run", ""))
    assert report["if"] == "${{ always() && steps.setup.outcome == 'success' }}"
    assert "scripts.ci_result quality" in report["run"]
    assert jobs["quality"]["if"] == "needs.changes.outputs.checks == 'true'"
    for name in TEST_PARTITIONS:
        steps = jobs[name]["steps"]
        artifact = next(
            step for step in steps if step.get("with", {}).get("name") == f"{name}-coverage"
        )
        assert artifact["with"]["path"] == f".coverage.{name}"
        assert artifact["with"]["if-no-files-found"] == "error"
        assert artifact["with"]["include-hidden-files"] is True


def test_workflows_pin_actions_and_do_not_publish_or_ignore_failures() -> None:
    paths = [*sorted((ROOT / ".github/workflows").glob("*.yml"))]
    paths.append(ROOT / ".github/actions/setup-python/action.yml")
    for path in paths:
        source = path.read_text()
        assert "continue-on-error" not in source
        assert "pytest-rerunfailures" not in source
        assert "secrets." not in source
        for action in re.findall(r"uses: (\S+)", source):
            assert action.startswith("./") or re.fullmatch(r"[\w/-]+@[0-9a-f]{40}", action)
    docker = yaml.safe_load((ROOT / ".github/workflows/docker.yml").read_text())
    images = docker["jobs"]["images"]
    assert images["strategy"]["fail-fast"] is False
    assert images["strategy"]["matrix"]["image"] == "${{ fromJSON(inputs.images) }}"
    build = next(
        step["with"] for step in images["steps"] if "with" in step and "file" in step["with"]
    )
    assert build["push"] is False
    assert "build-args" not in build


def test_setup_caches_the_actual_project_directory_and_uses_lock() -> None:
    source = (ROOT / ".github/actions/setup-python/action.yml").read_text()
    assert "path: .uv-cache" in source
    assert "hashFiles('uv.lock', 'mcp_server/uv.lock')" in source
    assert 'version: "0.11.32"' in source
    assert "make install ENV=test" in source


def test_image_workflow_checks_the_loaded_runtime() -> None:
    source = (ROOT / ".github/workflows/docker.yml").read_text()
    assert "load: true" in source
    assert "500000000" in source
    assert "whoami" in source
    assert "shutil.which" in source
    assert '"gcc", "g++", "make", "uv"' in source


def test_setup_failures_and_partial_reports_are_explicit() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    for name, job in jobs.items():
        if name == "build":
            continue
        fallback = next(
            step for step in job["steps"] if step.get("name") == "Explain unavailable setup"
        )
        assert "always()" in fallback["if"]
        assert "steps.setup.outcome != 'success'" in fallback["if"]
        assert ".venv" not in fallback["run"]
    for name in TEST_PARTITIONS:
        steps = jobs[name]["steps"]
        report = next(step for step in steps if "scripts.ci_result tests" in step.get("run", ""))
        assert "steps.setup.outcome == 'success'" in report["if"]
        assert report["if"] == "${{ always() && steps.setup.outcome == 'success' }}"
        assert report["env"]["TEST_OUTCOME"] == "${{ steps.tests.outcome }}"
        assert '--outcome "$TEST_OUTCOME"' in report["run"]
        for step in steps:
            if "upload-artifact@" in step.get("uses", ""):
                assert "always()" in step["if"]
                assert "hashFiles(" in step["if"]
    gate = next(
        step
        for step in jobs["ci-result"]["steps"]
        if "scripts.ci_result result" in step.get("run", "")
    )
    assert '--tested-sha "$CI_TESTED_SHA"' in gate["run"]
    assert gate["env"]["CI_TESTED_SHA"] == "${{ github.sha }}"


def test_log_pipelines_preserve_command_failure(tmp_path: Path) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    for name in ("quality", *TEST_PARTITIONS):
        for step in jobs[name]["steps"]:
            script = step.get("run", "")
            if " | tee " not in script:
                continue
            assert step["shell"] == "bash"
            assert script.startswith("set -o pipefail\n")
            # Keep the actual pipeline suffix and options, substitute only the producer.
            diagnostic = script.splitlines()[0] + "\n(exit 7) 2>&1" + script.split("2>&1", 1)[1]
            result = subprocess.run(  # noqa: S603 -- repository script with a fixed failing producer.
                ["/bin/bash", "-e", "-c", diagnostic],
                cwd=tmp_path,
                capture_output=True,
                check=False,
                timeout=5,
            )
            assert result.returncode == 7  # noqa: PLR2004 -- fixed failing producer exit code.


def test_test_timeouts_leave_room_for_diagnostics_without_shortening_execution() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    for name, step_minutes in (("unit", 8), ("integration", 14), ("storage", 12)):
        job = jobs[name]
        tests = next(step for step in job["steps"] if step.get("id") == "tests")
        assert tests["timeout-minutes"] >= step_minutes
        assert job["timeout-minutes"] > tests["timeout-minutes"]
    migrations = next(
        step
        for step in jobs["integration"]["steps"]
        if "scripts.ci_result migrations" in step.get("run", "")
    )
    assert migrations["if"] == "${{ always() && steps.setup.outcome == 'success' }}"
    assert '--outcome "$TEST_OUTCOME"' in migrations["run"]


def test_contract_check_runs_first_and_is_required_by_quality() -> None:
    steps = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]["quality"][
        "steps"
    ]
    checks = [step for step in steps if step.get("id") in QUALITY_CHECKS]
    assert {step["id"] for step in checks} == set(QUALITY_CHECKS)
    assert checks[0]["id"] == "contracts"
    assert "quality-contracts.log" in checks[0]["run"]


def test_ruff_annotations_reuse_the_same_blocking_check() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["quality"]["steps"]
    lint = next(step for step in steps if step.get("id") == "lint")
    assert lint["env"]["RUFF_OUTPUT_FORMAT"] == "github"
    assert lint["run"].startswith("set -o pipefail\n")
    assert "tee quality-lint.log" in lint["run"]
    assert sum("make lint-check" in step.get("run", "") for step in steps) == 1


def test_format_artifact_only_follows_failed_format_without_write_permissions() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    quality = workflow["jobs"]["quality"]
    steps = quality["steps"]
    repair = next(step for step in steps if step.get("id") == "format_repair")
    assert repair["if"] == (
        "${{ !cancelled() && steps.setup.outcome == 'success' && "
        "(steps.lint.outcome == 'failure' || steps.format.outcome == 'failure') }}"
    )
    assert "make ruff-patch ENV=test" in repair["run"]
    assert repair["env"]["CI_TESTED_SHA"] == "${{ github.sha }}"
    assert "github.run_attempt" in repair["env"]["CI_FORMAT_OUTPUT"]
    upload = next(step for step in steps if step.get("with", {}).get("name") == "format-repair")
    assert "always()" in upload["if"]
    assert "steps.format_repair.outcome != 'skipped'" in upload["if"]
    assert upload["with"]["if-no-files-found"] == "warn"
    assert "runner.temp" in upload["with"]["path"]
    assert quality["outputs"]["quality_details"] == "${{ steps.report.outputs.quality_details }}"
    report = next(step for step in steps if step.get("id") == "report")
    assert '--output "$GITHUB_OUTPUT"' in report["run"]
    assert '--tested-sha "$CI_TESTED_SHA"' in report["run"]
    assert workflow["permissions"] == {"contents": "read"}
    assert "permissions" not in quality
    assert "git push" not in str(steps)
    assert "git commit" not in str(steps)


def test_both_runtime_images_execute_shared_sql_policy_smoke() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/docker.yml").read_text())
    job = workflow["jobs"]["images"]
    assert "inputs.images" in job["strategy"]["matrix"]["image"]
    smoke = next(step for step in job["steps"] if step.get("name") == "Verify runtime image")
    commands = smoke["run"]
    assert 'docker run --rm "$IMAGE" python -c' in commands
    assert "from app.core.sql_policy.sql_validator import SQLValidator" in commands
    assert (
        'policy.validate("SELECT order_id FROM biz.orders").status is ValidationStatus.VALID'
        in commands
    )
    assert 'policy.validate("DELETE FROM biz.orders").status is ValidationStatus.UNSAFE' in commands
    assert (
        "COPY app/core/sql_policy ./app/core/sql_policy"
        in Path("docker/Dockerfile.mcp").read_text()
    )


def test_failed_partitions_still_collect_coverage_and_stage_evidence() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    assert jobs["coverage"]["if"] == (
        "${{ !cancelled() && needs.changes.outputs.checks == 'true' }}"
    )
    for step in jobs["coverage"]["steps"]:
        if "download-artifact@" in step.get("uses", "") or step.get("id") == "coverage":
            values = {
                "steps.setup.outcome": "success",
                "needs.unit.outputs.coverage_artifact_id": "101",
                "needs.integration.outputs.coverage_artifact_id": "102",
                "needs.storage.outputs.coverage_artifact_id": "103",
            }
            assert condition_holds(step["if"], values)
            assert not condition_holds(step["if"], values, cancelled=True)
    for name in ("quality", "unit", "integration", "storage", "coverage"):
        recorder = next(
            step for step in jobs[name]["steps"] if step.get("name") == f"Record {name} diagnostics"
        )
        assert "always()" in recorder["if"]
        for field in ("CI_TESTED_SHA", "CI_RUN_ID", "CI_RUN_ATTEMPT"):
            assert field in recorder["env"]
    final = jobs["ci-result"]["steps"]
    download = next(step for step in final if step.get("id") == "diagnostics_download")
    assert download["with"]["merge-multiple"] is False
    assert "run-id" in final[-1]["run"]
    assert "run-attempt" in final[-1]["run"]
    assert "always()" in final[-1]["if"]


def test_image_diagnostics_record_both_build_and_smoke_outcomes() -> None:
    steps = yaml.safe_load((ROOT / ".github/workflows/docker.yml").read_text())["jobs"]["images"][
        "steps"
    ]
    assert {"build", "smoke", "setup"} <= {step.get("id") for step in steps}
    recorder = next(step for step in steps if "scripts.ci_diagnostics" in step.get("run", ""))
    assert "always()" in recorder["if"]
    assert recorder["env"]["CI_STEPS"] == "${{ toJSON(steps) }}"
    assert "matrix.image" in recorder["run"]


def test_dependency_audit_and_phase_assessments_reach_current_run_artifacts() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    quality = jobs["quality"]["steps"]
    audit = next(step for step in quality if step.get("id") == "dependencies")
    assert "scripts.ci_dependencies --output dependency-evidence" in audit["run"]
    artifact = next(
        step for step in quality if step.get("with", {}).get("name") == "dependency-evidence"
    )
    assert "always()" in artifact["if"]
    coverage = next(step for step in jobs["coverage"]["steps"] if step.get("id") == "coverage")
    assert "--output coverage-assessment.json" in coverage["run"]
    record = next(
        step
        for step in jobs["coverage"]["steps"]
        if step.get("name") == "Record coverage diagnostics"
    )
    assert "--assessment coverage-assessment.json" in record["run"]


def test_type_check_runs_once_and_uploads_even_after_failure() -> None:
    steps = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]["quality"][
        "steps"
    ]
    types = next(step for step in steps if step.get("id") == "types")
    assert types["run"].startswith("set -o pipefail\n")
    assert "make typecheck-report ENV=test" in types["run"]
    assert sum("typecheck" in step.get("run", "") for step in steps) == 1
    upload = next(step for step in steps if step.get("id") == "types_upload")
    assert "always()" in upload["if"]
    assert upload["with"]["name"] == "type-evidence"
    recorder = next(step for step in steps if step.get("name") == "Record quality diagnostics")
    assert "--types type-evidence/report.json" in recorder["run"]


def test_storage_job_is_isolated_and_marker_expressions_match_shared_contract() -> None:
    from scripts.ci_partitions import SELECTIONS  # noqa: PLC0415 -- pure workflow contract.

    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    storage = jobs["storage"]
    assert "services" not in storage
    assert "install-mcp" not in str(storage)
    assert "scripts.ci_result migrations" not in str(storage)
    for name, selection in SELECTIONS.items():
        step = next(step for step in jobs[name]["steps"] if step.get("id") == "tests")
        assert f'-m "{selection}"' in step["run"]
    assert "milvus-evidence/" in str(storage)


def test_structure_check_precedes_collection_and_uploads_have_diagnostic_ids() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    steps = jobs["quality"]["steps"]
    ids = [step.get("id") for step in steps]
    assert ids.index("contracts") < ids.index("structure") < ids.index("collection")
    structure = next(step for step in steps if step.get("id") == "structure")
    assert "scripts.check_test_structure" in structure["run"]
    assert "pytest" not in structure["run"]
    upload = next(step for step in steps if step.get("id") == "dependencies_upload")
    assert upload["with"]["name"] == "dependency-evidence"
