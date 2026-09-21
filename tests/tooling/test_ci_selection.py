"""Change selection must save work without hiding unverified functional inputs."""

# Fixed workflow outputs and synthetic GitHub IDs are test data.
# ruff: noqa: PLR2004

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import SecretStr, ValidationError

from app.core.errors import ValidationError as ProjectValidationError
from scripts import ci_changes
from scripts.ci_changes import (
    BaselineContext,
    DiscoveryError,
    Github,
    Run,
    Settings,
    choose_baseline,
    diff_paths,
    safe_discover,
    write_plan,
)
from scripts.ci_policy import Plan, Reason, classify, full_plan
from scripts.ci_result import MIGRATION_CASES, final_result, migration_summary
from tests.tooling.support import condition_holds, needs_for

BASE = "a" * 40
HEAD = "b" * 40
TARGET = "c" * 40


@pytest.mark.parametrize(
    ("path", "checks", "images"),
    [
        ("README.md", False, []),
        ("AGENTS.md", False, []),
        ("PROJECT_OVERVIEW.md", False, []),
        ("docs/TESTING.md", False, []),
        ("docs/roadmap/PHASE-02-data-agent.md", False, []),
        ("docs/evidence/step-2.4/README.md", False, []),
        ("docs/evidence/step-2.4/tests.txt", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
        ("docs/new-area/policy.md", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
        ("tests/unit/test_periods.py", True, []),
        ("tests/conftest.py", True, []),
        ("tests/e2e/test_chaos.py", True, []),
        ("app/services/periods.py", True, ["api"]),
        ("app/agents/prompts/sql.md", True, ["api"]),
        ("app/services/llm/prompts/repair.md", True, ["api"]),
        ("app/core/errors.py", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
        ("app/schemas/new.py", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
        ("app/__init__.py", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
        ("mcp_server/tools/schema.py", True, ["mcp"]),
        ("mcp_server/uv.lock", True, ["mcp"]),
        ("data/seed/DESIGN.md", True, ["api"]),
        ("data/seed/metrics.yaml", True, ["api"]),
        ("data/corpus/promo_2026_summer.md", True, ["api"]),
        ("data/corpus/MANIFEST.yaml", True, ["api"]),
        ("data/corpus/policy/warranty.pdf", True, ["api"]),
        ("data/corpus/sop/category_codes.xlsx", True, ["api"]),
        ("data/corpus/policy/warranty.pdf.meta.yaml", True, ["api"]),
        ("alembic/app/data/0007_metric_catalog.json", True, ["api"]),
        ("alembic.ini", True, ["api"]),
        ("pyproject.toml", True, ["api"]),
        ("uv.lock", True, ["api"]),
        ("app/resources/provider_capabilities.json", True, ["api"]),
        ("scripts/ci_policy.py", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
        (".github/workflows/ci.yml", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
        (".dockerignore", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
        ("docker/Dockerfile.api", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
        ("docker-compose.yml", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
        ("Makefile", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
        ("future/component.py", True, ["api", "mcp", "model-runtime", "model-tunnel"]),
    ],
)
def test_classification(path: str, checks: bool, images: list[str]) -> None:
    plan = classify([path], baseline=BASE, tested_sha=HEAD)
    assert plan.checks is checks
    assert plan.images == images
    assert Plan.model_validate_json(plan.model_dump_json()) == plan


def test_union_and_empty_diff() -> None:
    plan = classify(
        ["README.md", "app/services/periods.py", "mcp_server/server.py"],
        baseline=BASE,
        tested_sha=HEAD,
    )
    assert plan.images == ["api", "mcp"]
    assert not classify([], baseline=BASE, tested_sha=HEAD).checks


@pytest.mark.parametrize(
    "updates",
    [
        {"checks": False, "images": ["api"]},
        {"images": ["api", "api"]},
        {"images": ["unknown"]},
        {"checks": "false"},
        {"version": 2},
        {"unexpected": True},
        {"reason": "manual_full_run", "images": []},
    ],
)
def test_invalid_plan_rejected(updates: dict[str, object]) -> None:
    data = json.loads(full_plan(HEAD, Reason.MANUAL).model_dump_json())
    data.update(updates)
    with pytest.raises((ValidationError, ProjectValidationError)):
        Plan.model_validate_json(json.dumps(data))


def run_record(**updates: object) -> Run:
    data: dict[str, object] = {
        "id": 1,
        "head_sha": BASE,
        "head_branch": "main",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
    }
    data.update(updates)
    return Run.model_validate(data)


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", None])
def test_failed_or_incomplete_push_does_not_advance_baseline(conclusion: str | None) -> None:
    runs = [run_record(id=2, head_sha=TARGET, conclusion=conclusion), run_record()]
    assert (
        choose_baseline(
            runs,
            context=BaselineContext(branch="main", target=HEAD, current_run=3),
            is_ancestor=lambda _old, _new: True,
            trusted=lambda _run: True,
        )
        == BASE
    )


def test_baseline_excludes_nonancestor_other_branch_untrusted_and_current_run() -> None:
    runs = [
        run_record(id=9),
        run_record(id=8, head_branch="other"),
        run_record(id=7, head_sha=TARGET),
        run_record(id=6, event="pull_request"),
        run_record(id=5),
        run_record(id=4, status="in_progress"),
        run_record(),
    ]
    assert (
        choose_baseline(
            runs,
            context=BaselineContext(branch="main", target=HEAD, current_run=9),
            is_ancestor=lambda old, _new: old != TARGET,
            trusted=lambda run: run.id == 1,
        )
        == BASE
    )


def test_baseline_search_is_bounded() -> None:
    runs = [run_record(id=2)] * 100 + [run_record()]
    assert (
        choose_baseline(
            runs,
            context=BaselineContext(branch="main", target=HEAD, current_run=3),
            is_ancestor=lambda _old, _new: True,
            trusted=lambda run: run.id == 1,
        )
        is None
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    event = tmp_path / "event.json"
    event.write_text("{}")
    return Settings(
        _env_file=None,
        token=SecretStr("dummy-ci-token"),
        repository="owner/project",
        event="push",
        tested_sha=HEAD,
        run_id=3,
        event_path=event,
        output=tmp_path / "output",
        summary=tmp_path / "summary",
    )


@pytest.fixture
def github() -> Iterator[Github]:
    def response(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/jobs"):
            return httpx.Response(
                200,
                json={"total_count": 1, "jobs": [{"name": "ci-result", "conclusion": "success"}]},
            )
        return httpx.Response(200, json={"workflow_runs": [run_record().model_dump()]})

    with httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(response)
    ) as client:
        yield Github(client, "owner/project")


def test_main_diff_includes_changes_since_success_not_push_before(
    settings: Settings, github: Github, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.event_path.write_text(json.dumps({"before": TARGET}))
    monkeypatch.setattr(ci_changes, "ancestor", lambda _old, _new: True)

    def changed(old: str, new: str) -> list[str]:
        assert (old, new) == (BASE, HEAD)
        return ["README.md", "app/services/previously_failed.py"]

    monkeypatch.setattr(ci_changes, "diff_paths", changed)
    assert safe_discover(settings, github).checks


def test_pr_includes_unverified_target_changes_and_tests_merge_sha(
    settings: Settings, github: Github, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.event = "pull_request"
    settings.event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "base": {"sha": TARGET, "ref": "main"},
                    "head": {"sha": "d" * 40, "ref": "feature"},
                }
            }
        )
    )
    monkeypatch.setattr(ci_changes, "ancestor", lambda _old, _new: True)
    monkeypatch.setattr(ci_changes, "git_bytes", lambda _args: BASE.encode())
    calls: list[tuple[str, str]] = []

    def changed(old: str, new: str) -> list[str]:
        calls.append((old, new))
        return ["app/core/errors.py"] if new == TARGET else ["README.md"]

    monkeypatch.setattr(ci_changes, "diff_paths", changed)
    plan = safe_discover(settings, github)
    assert calls == [(BASE, TARGET), (BASE, "d" * 40)]
    assert plan.tested_sha == HEAD
    assert plan.images == ["api", "mcp", "model-runtime", "model-tunnel"]


def test_no_ancestor_and_manual_force_full(
    settings: Settings, github: Github, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ci_changes, "ancestor", lambda _old, _new: False)
    assert safe_discover(settings, github).reason == Reason.BASELINE
    settings.event = "workflow_dispatch"
    assert safe_discover(settings, github) == full_plan(HEAD, Reason.MANUAL)


@pytest.mark.parametrize("status", [403, 404, 500])
def test_api_failure_forces_full_without_retry(settings: Settings, status: int) -> None:
    calls = []

    def response(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status)

    with httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(response)
    ) as client:
        plan = safe_discover(settings, Github(client, "owner/project"))
    assert plan == full_plan(HEAD, Reason.DISCOVERY)
    assert len(calls) == 1


def test_git_failure_forces_full(
    settings: Settings, github: Github, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(_old: str, _new: str) -> bool:
        raise DiscoveryError()

    monkeypatch.setattr(ci_changes, "ancestor", unavailable)
    assert safe_discover(settings, github) == full_plan(HEAD, Reason.DISCOVERY)


@pytest.mark.parametrize(
    "jobs",
    [
        {"total_count": 0, "jobs": []},
        {"total_count": 1, "jobs": [{"name": "ci-result", "conclusion": "failure"}]},
    ],
)
def test_success_without_successful_final_gate_is_untrusted(jobs: dict[str, object]) -> None:
    with httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=jobs)),
    ) as client:
        assert not Github(client, "owner/project").trusted(run_record())


def test_nul_paths_preserve_renames_deletions_and_shell_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = ["app/old.py", "docs/new.md", "tests/a b\n`literal`$(text).py"]

    def command(arguments: list[str]) -> bytes:
        assert arguments == ["diff", "--name-only", "--no-renames", "-z", BASE, HEAD, "--"]
        return b"\0".join(name.encode() for name in names) + b"\0"

    monkeypatch.setattr(ci_changes, "git_bytes", command)
    assert diff_paths(BASE, HEAD) == names
    assert classify(diff_paths(BASE, HEAD), baseline=BASE, tested_sha=HEAD).images == ["api"]


def test_git_is_bounded_and_does_not_use_a_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ci_changes.shutil, "which", lambda _name: "/usr/bin/git")

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert command == ["/usr/bin/git", "status"]
        assert kwargs == {"capture_output": True, "timeout": 30, "check": False}
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(ci_changes.subprocess, "run", execute)
    assert ci_changes.git(["status"]).returncode == 0


def test_missing_git_executable_is_a_typed_discovery_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ci_changes.shutil, "which", lambda _name: None)
    with pytest.raises(DiscoveryError, match="Git executable unavailable"):
        ci_changes.git(["status"])


def test_outputs_cannot_be_injected_with_newlines_or_summary_markup(tmp_path: Path) -> None:
    path = "tests/evil\nchecks=false\n<script>name.py"
    plan = classify([path], baseline=BASE, tested_sha=HEAD)
    output, summary = tmp_path / "out", tmp_path / "summary"
    write_plan(plan, output, summary)
    lines = output.read_text().splitlines()
    assert len(lines) == 3
    assert Plan.model_validate_json(lines[0].removeprefix("plan=")) == plan
    assert "<script>" not in summary.read_text()


@pytest.mark.parametrize("paths", [["README.md"], ["tests/test_x.py"], ["app/main.py"]])
def test_final_gate_accepts_only_planned_success_and_skips(paths: list[str]) -> None:
    plan = classify(paths, baseline=BASE, tested_sha=HEAD)
    assert final_result(
        plan.model_dump_json(), json.dumps(needs_for(plan)), tested_sha=plan.tested_sha
    )


@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", "", "invalid", None])
def test_successful_job_labels_cannot_replace_collection_evidence(outcome: str | None) -> None:
    plan = full_plan(HEAD, Reason.MANUAL)
    needs = needs_for(plan)
    needs["quality"]["outputs"] = {} if outcome is None else {"collection_outcome": outcome}
    assert not final_result(plan.model_dump_json(), json.dumps(needs), tested_sha=HEAD)


def test_document_only_selection_does_not_require_collection(tmp_path: Path) -> None:
    plan = classify(["docs/TESTING.md"], baseline=BASE, tested_sha=HEAD)
    needs = needs_for(plan)
    needs["quality"].pop("outputs")
    jobs = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())["jobs"]
    values = {"needs.changes.result": "success", "needs.changes.outputs.checks": "false"}
    for name in ("unit", "integration", "storage"):
        assert not condition_holds(jobs[name]["if"], values)
    summary = tmp_path / "summary"
    assert final_result(plan.model_dump_json(), json.dumps(needs), summary, tested_sha=HEAD)
    assert "planned omission" in summary.read_text()


@pytest.mark.parametrize(
    "job", ["changes", "quality", "unit", "integration", "storage", "coverage", "build"]
)
@pytest.mark.parametrize("result", ["skipped", "failure", "cancelled", None])
def test_required_job_cannot_disappear_or_fail(job: str, result: str | None) -> None:
    plan = full_plan(HEAD, Reason.MANUAL)
    needs = needs_for(plan)
    if result is None:
        needs.pop(job)
    else:
        needs[job]["result"] = result
    assert not final_result(plan.model_dump_json(), json.dumps(needs), tested_sha=plan.tested_sha)


def test_final_gate_rejects_missing_outputs_and_unplanned_execution() -> None:
    plan = classify(["README.md"], baseline=BASE, tested_sha=HEAD)
    needs = needs_for(plan)
    needs["quality"]["result"] = "failure"
    assert not final_result(plan.model_dump_json(), json.dumps(needs), tested_sha=plan.tested_sha)
    assert not final_result("", json.dumps(needs), tested_sha=plan.tested_sha)
    assert not final_result(plan.model_dump_json(), "{}", tested_sha=plan.tested_sha)


@pytest.mark.parametrize("child", ["", "<failure/>", "<error/>", "<skipped/>"])
def test_migration_report_requires_executed_passing_cases(tmp_path: Path, child: str) -> None:
    report, summary = tmp_path / "results.xml", tmp_path / "summary"
    cases = "".join(
        f'<testcase classname="tests.integration.test_migrations" name="{name}">{child}</testcase>'
        for name in MIGRATION_CASES
    )
    report.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    assert migration_summary(report, summary) is (not child)
    assert "Migration acceptance" in summary.read_text()


def test_missing_or_empty_migration_report_is_not_success(tmp_path: Path) -> None:
    report, summary = tmp_path / "results.xml", tmp_path / "summary"
    assert not migration_summary(report, summary)
    report.write_text("<testsuites/>")
    assert not migration_summary(report, summary)


def test_truncated_job_list_forces_full(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ci_changes, "ancestor", lambda _old, _new: True)

    def response(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/jobs"):
            return httpx.Response(200, json={"total_count": 101, "jobs": []})
        return httpx.Response(200, json={"workflow_runs": [run_record().model_dump()]})

    with httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(response)
    ) as client:
        assert safe_discover(settings, Github(client, "owner/project")) == full_plan(
            HEAD, Reason.DISCOVERY
        )


def test_malformed_metadata_forces_full(settings: Settings) -> None:
    with httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, text="not JSON")),
    ) as client:
        assert safe_discover(settings, Github(client, "owner/project")) == full_plan(
            HEAD, Reason.DISCOVERY
        )


def test_http_timeout_and_expired_budget_force_full(settings: Settings) -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated", request=request)

    with httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(timeout)
    ) as client:
        github = Github(client, "owner/project")
        assert safe_discover(settings, github).reason == Reason.DISCOVERY
        github.deadline = 0
        assert safe_discover(settings, github).reason == Reason.DISCOVERY


def test_git_timeout_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(ci_changes.subprocess, "run", timeout)
    with pytest.raises(DiscoveryError):
        ci_changes.git(["status"])


def test_missing_history_is_not_treated_as_a_nonancestor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ci_changes, "git", lambda args: subprocess.CompletedProcess(args, 128, b"", b"")
    )
    with pytest.raises(DiscoveryError):
        ci_changes.ancestor(BASE, HEAD)


@pytest.mark.parametrize("image", ["api", "mcp", "model-runtime", "model-tunnel"])
def test_all_docker_copy_inputs_select_their_image(image: str) -> None:
    root = Path(__file__).resolve().parents[2]
    for line in (root / f"docker/Dockerfile.{image}").read_text().splitlines():
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        for source in line.split()[1:-1]:
            # Representative files inside directory COPY sources; glob inputs
            # only need their conservative parent prefix for classification.
            path = source.removeprefix("./")
            if (root / path).is_dir():
                path += "/representative.py"
            assert image in classify([path], baseline=BASE, tested_sha=HEAD).images, source


def test_cloud_contract_scenarios_cover_docs_and_tests_without_network(tmp_path: Path) -> None:
    for name, paths in (("docs", ["docs/TESTING.md"]), ("tests", ["tests/unit/new.py"])):
        plan = classify(paths, baseline=BASE, tested_sha=HEAD)
        output, summary = tmp_path / f"{name}.out", tmp_path / f"{name}.md"
        write_plan(plan, output, summary)
        transported = output.read_text().splitlines()[0].removeprefix("plan=")
        assert final_result(transported, json.dumps(needs_for(plan)), tested_sha=plan.tested_sha)
        assert plan.checks is (name == "tests")
        assert not plan.images


def test_ci_metadata_example_documents_every_typed_setting() -> None:
    source = (Path(__file__).resolve().parents[2] / ".env.ci.example").read_text()
    for name in Settings.model_fields:
        assert f"IP_CI_{name.upper()}=" in source
    assert Settings.model_config.get("env_file") is None


def test_plan_without_trusted_baseline_cannot_skip_checks() -> None:
    plan = classify(["README.md"], baseline=BASE, tested_sha=HEAD)
    data = json.loads(plan.model_dump_json())
    data["baseline"] = None
    assert not final_result(
        json.dumps(data), json.dumps(needs_for(plan)), tested_sha=plan.tested_sha
    )


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_docs_following_failed_code_keep_the_trusted_successful_baseline(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, event: str
) -> None:
    settings.event = event
    if event == "pull_request":
        settings.event_path.write_text(
            json.dumps(
                {
                    "pull_request": {
                        "base": {"sha": TARGET, "ref": "main"},
                        "head": {"sha": "d" * 40, "ref": "docs"},
                    }
                }
            )
        )
    else:
        settings.event_path.write_text(json.dumps({"before": TARGET}))
    monkeypatch.setattr(ci_changes, "ancestor", lambda _old, _new: True)
    monkeypatch.setattr(ci_changes, "git_bytes", lambda _args: BASE.encode())
    diffs: list[tuple[str, str]] = []

    def changed(old: str, new: str) -> list[str]:
        diffs.append((old, new))
        return ["README.md"] if new == "d" * 40 else ["app/core/errors.py", "README.md"]

    def response(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/jobs"):
            assert request.url.path.endswith("/runs/1/jobs")
            return httpx.Response(
                200,
                json={"total_count": 1, "jobs": [{"name": "ci-result", "conclusion": "success"}]},
            )
        return httpx.Response(
            200,
            json={
                "workflow_runs": [
                    run_record(id=2, head_sha=TARGET, conclusion="failure").model_dump(),
                    run_record().model_dump(),
                ]
            },
        )

    monkeypatch.setattr(ci_changes, "diff_paths", changed)
    with httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(response)
    ) as client:
        plan = safe_discover(settings, Github(client, "owner/project"))
    assert plan.baseline == BASE
    assert plan.tested_sha == HEAD
    assert plan.checks
    assert plan.images == ["api", "mcp", "model-runtime", "model-tunnel"]
    assert plan.changed_paths == ["README.md", "app/core/errors.py"]
    expected = [(BASE, TARGET), (BASE, "d" * 40)] if event == "pull_request" else [(BASE, HEAD)]
    assert diffs == expected


def test_pr_base_advance_invalidates_old_merge_evidence(
    settings: Settings, github: Github, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.event = "pull_request"
    monkeypatch.setattr(ci_changes, "ancestor", lambda _old, _new: True)
    monkeypatch.setattr(ci_changes, "git_bytes", lambda _args: BASE.encode())
    monkeypatch.setattr(
        ci_changes,
        "diff_paths",
        lambda old, new: ["app/core/errors.py"] if new == TARGET else ["README.md"],
    )
    plans: list[Plan] = []
    for target, merge_sha in ((BASE, HEAD), (TARGET, "e" * 40)):
        settings.tested_sha = merge_sha
        settings.event_path.write_text(
            json.dumps(
                {
                    "pull_request": {
                        "base": {"sha": target, "ref": "main"},
                        "head": {"sha": "d" * 40, "ref": "docs"},
                    }
                }
            )
        )
        plans.append(safe_discover(settings, github))
    previous, current = plans
    assert not previous.checks
    assert current.checks
    assert current.images == ["api", "mcp", "model-runtime", "model-tunnel"]
    assert current.tested_sha != previous.tested_sha
    assert not final_result(
        previous.model_dump_json(),
        json.dumps(needs_for(previous)),
        tested_sha=current.tested_sha,
    )
    assert not final_result(
        current.model_dump_json(),
        json.dumps(needs_for(previous)),
        tested_sha=current.tested_sha,
    )
    assert final_result(
        current.model_dump_json(),
        json.dumps(needs_for(current)),
        tested_sha=current.tested_sha,
    )
