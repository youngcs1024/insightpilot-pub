"""Produce a SHA-bound Ruff diff artifact without formatting checkout files."""

import argparse
import subprocess
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.core.errors import InsightPilotError

Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
TIMEOUT_SECONDS = 60


class PatchStatus(StrEnum):
    """Only AVAILABLE permits displaying patch application instructions."""

    AVAILABLE = "available"
    NO_DIFF = "no_diff"
    TOOL_FAILED = "tool_failed"
    TIMEOUT = "timeout"
    SHA_MISMATCH = "sha_mismatch"
    DIRTY_CHECKOUT = "dirty_checkout"
    INVALID_PATCH = "invalid_patch"


class PatchRequest(BaseModel):
    """Explicit invocation data; no dotenv or application settings are loaded."""

    tested_sha: Sha
    output_dir: Path
    summary: Path
    run_url: str = Field(pattern=r"^https://github\.com/[\w.-]+/[\w.-]+/actions/runs/[0-9]+$")
    command: list[str] = Field(min_length=4)
    safe_fixes: bool = False


class LintIssue(BaseModel):
    """Bounded remediation identity; raw Ruff JSON remains a separate artifact."""

    code: str = Field(min_length=1, max_length=40)
    filename: str = Field(min_length=1, max_length=500)


class PatchEvidence(BaseModel):
    """Diagnostic metadata never substitutes for the format check's outcome."""

    version: Literal[1] = 1
    tested_sha: Sha
    status: PatchStatus = PatchStatus.TOOL_FAILED
    tool_version: str | None = None
    exit_code: int | None = None
    timeout_seconds: int = TIMEOUT_SECONDS
    remaining_lint: bool = False
    lint_exit_code: int | None = None
    format_exit_code: int | None = None
    verification_exit_code: int | None = None
    remaining_count: int | None = None
    remaining_issues: list[LintIssue] = Field(default_factory=list, max_length=50)


class PatchError(InsightPilotError):
    """Invalid artifact invocation; never execute a potentially mutating command."""


class PatchDeadlineExceededError(PatchError):
    """The shared process budget cannot be renewed for another command."""


class Runner:
    """One total deadline across read-only Git, version, diff and patch validation."""

    def __init__(self, log: Path) -> None:
        self.log = log
        self.deadline = monotonic() + TIMEOUT_SECONDS

    def run(
        self, command: list[str], *, data: bytes | None = None, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        """Keep stdout separate so Ruff's statistics never enter the patch."""
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise PatchDeadlineExceededError("Format repair process budget exhausted")
        result = subprocess.run(  # noqa: S603 -- explicit argument list, no shell.
            command,
            input=data,
            capture_output=True,
            timeout=remaining,
            check=False,
            cwd=cwd,
        )
        with self.log.open("ab") as stream:
            stream.write(result.stderr)
        return result


def inspect_checkout(runner: Runner, tested_sha: str) -> PatchStatus | None:
    """Only a clean tracked checkout of the requested SHA can supply a patch."""
    head = runner.run(["git", "rev-parse", "HEAD"])
    if head.returncode:
        return PatchStatus.TOOL_FAILED
    if head.stdout.decode().strip() != tested_sha:
        return PatchStatus.SHA_MISMATCH
    dirty = runner.run(["git", "status", "--porcelain", "--untracked-files=no"])
    if dirty.returncode:
        return PatchStatus.TOOL_FAILED
    return PatchStatus.DIRTY_CHECKOUT if dirty.stdout else None


def collect(request: PatchRequest, runner: Runner, evidence: PatchEvidence) -> bytes:
    """Validate before publishing; failed/partial output is never a usable patch."""
    version = runner.run([request.command[0], "--version"])
    if version.returncode:
        return b""
    evidence.tool_version = version.stdout.decode().strip()[:200]
    result = runner.run(request.command)
    evidence.exit_code = result.returncode
    if result.returncode == 0 and not result.stdout:
        evidence.status = PatchStatus.NO_DIFF
        return b""
    if result.returncode != 1 or not result.stdout:
        return b""
    check = runner.run(["git", "apply", "-p0", "--check", "-"], data=result.stdout)
    if check.returncode:
        evidence.status = PatchStatus.INVALID_PATCH
        return b""
    problem = inspect_checkout(runner, request.tested_sha)
    if problem is not None:
        evidence.status = problem
        return b""
    evidence.status = PatchStatus.AVAILABLE
    return result.stdout


def generate(
    request: PatchRequest,
    collector: Callable[[PatchRequest, Runner, PatchEvidence], bytes] = collect,
) -> PatchEvidence:
    """Write a fresh artifact directory; never overwrite an earlier repair bundle."""
    if request.command[1:3] != ["format", "--diff"] or any(
        argument.startswith("--") for argument in request.command[3:]
    ):
        raise PatchError("Expected a read-only Ruff format --diff invocation with target paths")
    request.output_dir.mkdir(parents=True, exist_ok=False)
    log = request.output_dir / "diagnostics.log"
    log.write_bytes(b"")
    evidence = PatchEvidence(tested_sha=request.tested_sha)
    patch = b""
    try:
        runner = Runner(log)
        problem = inspect_checkout(runner, request.tested_sha)
        if problem is None:
            patch = collector(request, runner, evidence)
        else:
            evidence.status = problem
    except PatchDeadlineExceededError:
        evidence.status = PatchStatus.TIMEOUT
    except subprocess.TimeoutExpired as error:
        evidence.status = PatchStatus.TIMEOUT
        if error.stderr:
            with log.open("ab") as stream:
                stream.write(error.stderr)
    except (OSError, UnicodeError):
        evidence.status = PatchStatus.TOOL_FAILED
    if evidence.status is PatchStatus.AVAILABLE:
        (request.output_dir / "format.patch").write_bytes(patch)
    (request.output_dir / "metadata.json").write_text(evidence.model_dump_json(indent=2) + "\n")
    with log.open("a") as stream:
        stream.write(f"\nPatch status: {evidence.status.value}\n")
    write_summary(request, evidence)
    return evidence


def write_summary(request: PatchRequest, evidence: PatchEvidence) -> None:
    """Bind remediation instructions to the exact checkout and a validated patch."""
    lines = [
        "## Ruff repair artifact\n" if request.safe_fixes else "## Format repair artifact\n",
        f"Status: `{evidence.status.value}`; tested SHA: `{request.tested_sha}`.\n",
        f"Download **format-repair** from [this run's artifacts]({request.run_url}#artifacts).\n",
        "This diagnostic does not change any failed quality check.\n",
        f"Remaining lint requires manual repair: {evidence.remaining_lint}.\n",
    ]
    lines.extend(
        f"- Manual repair: `{issue.code}` in `{issue.filename}`."
        for issue in evidence.remaining_issues
    )
    if evidence.status is PatchStatus.AVAILABLE:
        lines.extend(
            [
                "Extract the artifact at the repository root. Review metadata.json and the patch.",
                "Apply only from the recorded commit with a clean working tree:\n",
                "```bash",
                f'test "$(git rev-parse HEAD)" = "{request.tested_sha}" &&',
                "git apply -p0 --check format.patch &&",
                "git apply -p0 format.patch",
                "```",
            ]
        )
    else:
        lines.append(
            "No applicable patch is offered; inspect diagnostics.log and quality-format.log."
        )
    with request.summary.open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> None:
    """Accept only explicit workflow inputs, preserving diagnostic failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tested-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    request = PatchRequest(
        tested_sha=args.tested_sha,
        output_dir=args.output_dir,
        summary=args.summary,
        run_url=args.run_url,
        command=command,
    )
    evidence = generate(request)
    raise SystemExit(0 if evidence.status in {PatchStatus.AVAILABLE, PatchStatus.NO_DIFF} else 1)


if __name__ == "__main__":
    main()
