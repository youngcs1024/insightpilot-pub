"""Select CI work using complete Git history and a trusted successful ancestor."""

import argparse
import json
import shutil
import subprocess
from collections.abc import Callable
from html import escape
from pathlib import Path
from time import monotonic
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import InsightPilotError
from scripts.ci_policy import Plan, Reason, classify, full_plan

Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class DiscoveryError(InsightPilotError):
    """Metadata discovery is unavailable; do full validation without retrying."""


class Settings(BaseSettings):
    """Explicit GitHub-only metadata; never read a project dotenv file."""

    model_config = SettingsConfigDict(env_prefix="IP_CI_", extra="ignore")

    token: SecretStr
    repository: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    event: Literal["push", "pull_request", "workflow_dispatch"]
    tested_sha: Sha
    run_id: int = Field(gt=0)
    event_path: Path
    output: Path
    summary: Path


class Ref(BaseModel):
    """Pinned GitHub pull-request ref."""

    sha: Sha
    ref: str


class PullRequest(BaseModel):
    """Only immutable commit references are passed to Git."""

    base: Ref
    head: Ref


class Event(BaseModel):
    """Relevant subset of a GitHub event payload."""

    pull_request: PullRequest | None = None


class BaselineContext(BaseModel):
    """Pinned target and current run excluded from the ancestry search."""

    branch: str
    target: Sha
    current_run: int = Field(gt=0)


class Run(BaseModel):
    """Successful status alone is insufficient to trust a run."""

    id: int
    head_sha: Sha
    head_branch: str | None
    event: str
    status: str
    conclusion: str | None


class Runs(BaseModel):
    """Bounded GitHub workflow run listing."""

    workflow_runs: list[Run]


class Job(BaseModel):
    """Gate result in a completed workflow attempt."""

    name: str
    conclusion: str | None


class Jobs(BaseModel):
    """Job list, including a count to detect truncated responses."""

    total_count: int
    jobs: list[Job]


def git(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Run read-only Git with argument boundaries and a bounded, zero-retry policy."""
    executable = shutil.which("git")
    if executable is None:
        raise DiscoveryError("Git executable unavailable")
    try:
        return subprocess.run(  # noqa: S603 -- resolved git executable; no shell.
            [str(Path(executable).resolve()), *arguments],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DiscoveryError("Git discovery unavailable") from error


def git_bytes(arguments: list[str]) -> bytes:
    """Require successful Git output without exposing command diagnostics."""
    result = git(arguments)
    if result.returncode != 0:
        raise DiscoveryError("Git discovery failed")
    return result.stdout


def ancestor(older: str, newer: str) -> bool:
    """Distinguish a non-ancestor from a missing or unreadable commit."""
    result = git(["merge-base", "--is-ancestor", older, newer])
    if result.returncode not in {0, 1}:
        raise DiscoveryError("Git ancestry unavailable")
    return result.returncode == 0


def diff_paths(older: str, newer: str) -> list[str]:
    """Disable rename detection so both old and new paths appear in a NUL list."""
    raw = git_bytes(["diff", "--name-only", "--no-renames", "-z", older, newer, "--"])
    return [name.decode("utf-8") for name in raw.split(b"\0") if name]


class Github:
    """Read CI metadata with fixed endpoints, timeouts and no automatic retries."""

    def __init__(self, client: httpx.Client, repository: str) -> None:
        self.client = client
        self.prefix = f"/repos/{repository}/actions"
        self.deadline = monotonic() + 60

    def get(self, path: str, params: dict[str, str | int]) -> str:
        """Return JSON for immediate typed parsing, failing closed on HTTP errors."""
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise DiscoveryError("GitHub discovery budget exhausted")
        try:
            response = self.client.get(
                self.prefix + path, params=params, timeout=min(15, remaining)
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise DiscoveryError("GitHub metadata unavailable") from error
        return response.text

    def runs(self, branch: str) -> list[Run]:
        """Read at most the latest 100 runs of this workflow on the target branch."""
        return Runs.model_validate_json(
            self.get("/workflows/ci.yml/runs", {"branch": branch, "per_page": 100})
        ).workflow_runs

    def trusted(self, run: Run) -> bool:
        """Require a successful final gate in this repository."""
        jobs = Jobs.model_validate_json(self.get(f"/runs/{run.id}/jobs", {"per_page": 100}))
        if jobs.total_count != len(jobs.jobs):
            raise DiscoveryError("Truncated GitHub job listing")
        gates = [job for job in jobs.jobs if job.name == "ci-result"]
        return len(gates) == 1 and gates[0].conclusion == "success"


def choose_baseline(
    runs: list[Run],
    *,
    context: BaselineContext,
    is_ancestor: Callable[[str, str], bool],
    trusted: Callable[[Run], bool],
) -> str | None:
    """Ignore failed/cancelled/concurrent runs and non-ancestor successes."""
    for run in runs[:100]:
        if (
            run.id == context.current_run
            or run.head_branch != context.branch
            or run.event not in {"push", "workflow_dispatch"}
            or run.status != "completed"
            or run.conclusion != "success"
            or not is_ancestor(run.head_sha, context.target)
        ):
            continue
        if trusted(run):
            return run.head_sha
    return None


def discover(settings: Settings, github: Github) -> Plan:
    """Include unverified base changes in PRs and unverified prior pushes on main."""
    if settings.event == "workflow_dispatch":
        return full_plan(settings.tested_sha, Reason.MANUAL)
    event = Event.model_validate_json(settings.event_path.read_text())
    pr = event.pull_request
    if settings.event == "pull_request" and pr is None:
        raise DiscoveryError("Missing pull request references")
    branch = pr.base.ref if pr else "main"
    target = pr.base.sha if pr else settings.tested_sha
    baseline = choose_baseline(
        github.runs(branch),
        context=BaselineContext(branch=branch, target=target, current_run=settings.run_id),
        is_ancestor=ancestor,
        trusted=github.trusted,
    )
    if baseline is None:
        return full_plan(settings.tested_sha, Reason.BASELINE)
    paths = diff_paths(baseline, target)
    if pr:
        merge_base = git_bytes(["merge-base", pr.base.sha, pr.head.sha]).decode().strip()
        # Validate Git's output before passing it as another argument.
        merge_base = Ref(sha=merge_base, ref="merge-base").sha
        paths.extend(diff_paths(merge_base, pr.head.sha))
    return classify(paths, baseline=baseline, tested_sha=settings.tested_sha)


def safe_discover(settings: Settings, github: Github) -> Plan:
    """Discovery failure costs extra checks, never a false green skip."""
    try:
        return discover(settings, github)
    except (DiscoveryError, ValidationError, OSError, UnicodeError):
        return full_plan(settings.tested_sha, Reason.DISCOVERY)


def write_plan(plan: Plan, output: Path, summary: Path) -> None:
    """Emit compact workflow outputs and an escaped human-readable audit record."""
    with output.open("a") as stream:
        stream.write(f"plan={plan.model_dump_json()}\n")
        stream.write(f"checks={str(plan.checks).lower()}\n")
        stream.write(f"images={json.dumps(plan.images, separators=(',', ':'))}\n")
    with summary.open("a") as stream:
        stream.write("## CI selection\n\n")
        stream.write(f"Reason: `{plan.reason.value}`; baseline: `{plan.baseline}`.\n\n")
        stream.write(f"Tested commit: `{plan.tested_sha}`.\n\n")
        stream.write(f"Quality / both test groups / coverage: **{plan.checks}**.\n\n")
        stream.write(f"Images: `{','.join(plan.images) or 'none'}`.\n\n")
        stream.write("Omitted jobs have no affected inputs under the documented policy.\n\n")
        stream.write("Changed paths (JSON escaped):\n\n")
        # HTML escaping prevents paths from injecting summary markup or fences.
        stream.write(f"<pre>{escape(json.dumps(plan.changed_paths, ensure_ascii=True))}</pre>\n")


def main() -> None:
    """Compute the selection once; metadata credentials never reach test processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    settings = Settings()
    with httpx.Client(
        base_url="https://api.github.com",
        headers={
            "Authorization": f"Bearer {settings.token.get_secret_value()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        follow_redirects=False,
        timeout=15,
    ) as client:
        plan = safe_discover(settings, Github(client, settings.repository))
    write_plan(plan, settings.output, settings.summary)


if __name__ == "__main__":
    main()
