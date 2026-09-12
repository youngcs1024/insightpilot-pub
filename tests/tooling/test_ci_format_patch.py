"""Read-only repair evidence uses substitute tools, never a full-project formatter."""

import json
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.ci_format_patch import (
    TIMEOUT_SECONDS,
    PatchError,
    PatchRequest,
    PatchStatus,
    generate,
    main,
)
from tests.tooling.support import SHA, git_executable, run_make

PATCH = b"--- sample.py\n+++ sample.py\n@@ -1 +1 @@\n-x=1\n+x = 1\n\n"


class ProcessStub:
    """Replay process responses and retain exact argv/stdin for contract assertions."""

    def __init__(self, replies: list[tuple[int, bytes, bytes] | Exception]) -> None:
        self.replies = deque(replies)
        self.calls: list[tuple[list[str], bytes | None]] = []

    def run(
        self,
        command: list[str],
        *,
        input: bytes | None,
        capture_output: bool,
        timeout: float,
        check: bool,
        **options: object,
    ) -> subprocess.CompletedProcess[bytes]:
        assert options["cwd"] is None
        assert capture_output
        assert not check
        assert 0 < timeout <= TIMEOUT_SECONDS
        self.calls.append((command, input))
        reply = self.replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        code, stdout, stderr = reply
        return subprocess.CompletedProcess(command, code, stdout, stderr)


def request(tmp_path: Path) -> PatchRequest:
    return PatchRequest(
        tested_sha=SHA,
        output_dir=tmp_path / "repair artifact",
        summary=tmp_path / "summary.md",
        run_url="https://github.com/example/project/actions/runs/123",
        command=["/a path/ruff", "format", "--diff", "a path/sample.py"],
    )


def clean_checkout() -> list[tuple[int, bytes, bytes]]:
    return [(0, SHA.encode() + b"\n", b""), (0, b"", b"")]


def replies(code: int = 1, patch: bytes = PATCH) -> list[tuple[int, bytes, bytes] | Exception]:
    return [
        *clean_checkout(),
        (0, b"ruff 0.16.6\n", b""),
        (code, patch, b"1 file would be reformatted\n"),
        (0, b"", b""),
        *clean_checkout(),
    ]


def test_available_patch_is_sha_bound_and_statistics_are_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = ProcessStub(replies())
    monkeypatch.setattr(subprocess, "run", stub.run)
    value = request(tmp_path)
    source = tmp_path / "sample.py"
    source.write_bytes(b"x=1\n")
    evidence = generate(value)
    assert evidence.status is PatchStatus.AVAILABLE
    assert evidence.tested_sha == SHA
    assert evidence.exit_code == 1
    assert evidence.tool_version == "ruff 0.16.6"
    assert (value.output_dir / "format.patch").read_bytes() == PATCH
    assert b"1 file would be reformatted" in (value.output_dir / "diagnostics.log").read_bytes()
    assert source.read_bytes() == b"x=1\n"
    assert (value.command, None) in stub.calls
    assert (["git", "apply", "-p0", "--check", "-"], PATCH) in stub.calls
    assert not stub.replies
    summary = value.summary.read_text()
    assert SHA in summary
    assert "#artifacts" in summary
    assert "git apply -p0 --check format.patch" in summary
    assert json.loads((value.output_dir / "metadata.json").read_text())["status"] == "available"


@pytest.mark.parametrize(
    ("code", "stdout", "expected"),
    [
        (0, b"", PatchStatus.NO_DIFF),
        (2, PATCH, PatchStatus.TOOL_FAILED),
        (1, b"", PatchStatus.TOOL_FAILED),
        (0, PATCH, PatchStatus.TOOL_FAILED),
    ],
)
def test_nonpatch_outcomes_never_offer_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    stdout: bytes,
    expected: PatchStatus,
) -> None:
    stub = ProcessStub(replies(code, stdout))
    monkeypatch.setattr(subprocess, "run", stub.run)
    value = request(tmp_path)
    assert generate(value).status is expected
    assert not (value.output_dir / "format.patch").exists()
    assert "git apply" not in value.summary.read_text()


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        (subprocess.TimeoutExpired("ruff", 60), PatchStatus.TIMEOUT),
        (OSError("unavailable"), PatchStatus.TOOL_FAILED),
    ],
)
def test_process_failure_keeps_metadata_without_partial_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: Exception, expected: PatchStatus
) -> None:
    stub = ProcessStub([*clean_checkout(), (0, b"ruff 0.16.6", b""), problem])
    monkeypatch.setattr(subprocess, "run", stub.run)
    value = request(tmp_path)
    assert generate(value).status is expected
    assert (value.output_dir / "metadata.json").exists()
    assert not (value.output_dir / "format.patch").exists()


@pytest.mark.parametrize("after_diff", [False, True])
def test_sha_mismatch_never_publishes_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_diff: bool
) -> None:
    responses = replies()[:5] if after_diff else []
    responses.append((0, b"b" * 40, b""))
    stub = ProcessStub(responses)
    monkeypatch.setattr(subprocess, "run", stub.run)
    value = request(tmp_path)
    assert generate(value).status is PatchStatus.SHA_MISMATCH
    assert not (value.output_dir / "format.patch").exists()


def test_dirty_checkout_and_invalid_patch_are_not_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = ProcessStub([(0, SHA.encode(), b""), (0, b" M sample.py", b"")])
    monkeypatch.setattr(subprocess, "run", stub.run)
    assert generate(request(tmp_path)).status is PatchStatus.DIRTY_CHECKOUT
    responses = [*replies()[:4], (1, b"", b"invalid patch")]
    monkeypatch.setattr(subprocess, "run", ProcessStub(responses).run)
    value = request(tmp_path / "second")
    assert generate(value).status is PatchStatus.INVALID_PATCH
    assert not (value.output_dir / "format.patch").exists()


@pytest.mark.parametrize("arguments", [["format", "app"], ["format", "--diff", "--check", "app"]])
def test_non_diff_invocation_is_rejected(tmp_path: Path, arguments: list[str]) -> None:
    value = request(tmp_path).model_copy(update={"command": ["ruff", *arguments]})
    with pytest.raises(PatchError):
        generate(value)
    assert not value.output_dir.exists()


def test_invalid_sha_is_rejected_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        PatchRequest.model_validate({**request(tmp_path).model_dump(), "tested_sha": "bad sha"})


def test_validated_artifact_applies_to_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = request(tmp_path)
    with monkeypatch.context() as patcher:
        patcher.setattr(subprocess, "run", ProcessStub(replies()).run)
        generate(value)
    source = tmp_path / "sample.py"
    source.write_bytes(b"x=1\n")
    git = git_executable()
    for flags in (["--check"], []):
        applied = subprocess.run(  # noqa: S603 -- resolved Git; temporary fixture only.
            [git, "apply", "-p0", *flags, str(value.output_dir / "format.patch")],
            cwd=tmp_path,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert applied.returncode == 0, applied.stderr
    assert source.read_bytes() == b"x = 1\n"


def test_cli_preserves_diagnostic_failure_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = request(tmp_path)
    monkeypatch.setattr(subprocess, "run", ProcessStub(replies(2)).run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ci_format_patch",
            "--tested-sha",
            SHA,
            "--output-dir",
            str(value.output_dir),
            "--summary",
            str(value.summary),
            "--run-url",
            value.run_url,
            "--",
            *value.command,
        ],
    )
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 1


def test_make_check_and_diff_share_targets_without_executing_tools() -> None:
    dry = run_make("-n", "format-check", "format-patch", "ENV=test")
    assert dry.returncode == 0
    lines = dry.stdout.splitlines()
    check = next(line for line in lines if "ruff format --check " in line)
    diff = next(line for line in lines if "ruff format --diff " in line)
    assert check.split("ruff format --check ")[1] == diff.split("ruff format --diff ")[1]
    assert "--locked" in check
    assert "--locked" in diff


def test_missing_git_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(AssertionError, match="Git is required"):
        git_executable()


@pytest.mark.parametrize("completed_commands", [0, 1])
def test_exhausted_budget_starts_no_further_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed_commands: int
) -> None:
    times = iter([0.0, *([1.0] * completed_commands), float(TIMEOUT_SECONDS)])
    monkeypatch.setattr("scripts.ci_format_patch.monotonic", lambda: next(times))
    stub = ProcessStub(replies())
    monkeypatch.setattr(subprocess, "run", stub.run)
    value = request(tmp_path)
    assert generate(value).status is PatchStatus.TIMEOUT
    assert len(stub.calls) == completed_commands
    assert not (value.output_dir / "format.patch").exists()


@pytest.mark.parametrize(
    ("matching_sha", "check_exit", "expected"),
    [
        (False, 0, "rev-parse\n"),
        (True, 1, "rev-parse\ncheck\n"),
        (True, 0, "rev-parse\ncheck\napply\n"),
    ],
)
def test_summary_commands_short_circuit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    matching_sha: bool,
    check_exit: int,
    expected: str,
) -> None:
    value = request(tmp_path)
    with monkeypatch.context() as patcher:
        patcher.setattr(subprocess, "run", ProcessStub(replies()).run)
        generate(value)
    script = value.summary.read_text().split("```bash\n", 1)[1].split("```", 1)[0]
    fake_git = tmp_path / "git"
    actual_sha = SHA if matching_sha else "b" * 40
    fake_git.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        f"  rev-parse) echo rev-parse >> calls; echo {actual_sha} ;;\n"
        '  apply) case "$3" in\n'
        f"    --check) echo check >> calls; exit {check_exit} ;;\n"
        "    *) echo apply >> calls ;;\n"
        "  esac ;;\n"
        "esac\n"
    )
    fake_git.chmod(0o700)
    result = subprocess.run(  # noqa: S603 -- generated instructions, fixture-only Git stub.
        ["/bin/bash", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(tmp_path)},
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert (tmp_path / "calls").read_text() == expected
    assert (result.returncode == 0) is (matching_sha and check_exit == 0)
