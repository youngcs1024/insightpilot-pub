"""Safe repair composition uses fixture sources and substitute Ruff processes only."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.ci_format_patch import PatchEvidence, PatchRequest, PatchStatus, Runner, generate
from scripts.ci_ruff_repair import collect_safe, combined_diff, verify_remaining
from tests.tooling.support import SHA, git_executable


class RepairTools:
    """Model safe fixes and formatting on the same temporary copy."""

    def __init__(self, *, lint_exit: int = 0, invalid: bool = False) -> None:
        self.lint_exit = lint_exit
        self.invalid = invalid
        self.calls: list[list[str]] = []

    def run(
        self,
        command: list[str],
        *,
        input: bytes | None,
        cwd: Path | None,
        **options: object,
    ) -> subprocess.CompletedProcess[bytes]:
        assert options["capture_output"]
        assert not options["check"]
        assert isinstance(options["timeout"], float)
        assert options["timeout"] > 0
        self.calls.append(command)
        stdout, code = b"", 0
        if command[1:3] == ["rev-parse", "HEAD"]:
            stdout = SHA.encode()
        elif command[1] == "--version":
            stdout = b"ruff fixture"
        elif command[1] == "ls-files":
            stdout = b"sample.py\0pyproject.toml\0"
        elif command[1] == "check" and "--fix" in command:
            assert cwd is not None
            assert "--fix" in command
            assert "--no-unsafe-fixes" in command
            assert (cwd / "pyproject.toml").is_file()
            (cwd / "sample.py").write_text("value=1\n")
            code = self.lint_exit
        elif command[1] == "check":
            assert "--output-format=json" in command
            assert cwd is not None
            assert (cwd / "sample.py").read_text() == "value = 1\n"
            code = self.lint_exit
            stdout = json.dumps(
                [{"code": "PERF401", "filename": str(cwd / "sample.py")}] if code else []
            ).encode()
        elif command[1] == "format":
            assert cwd is not None
            assert (cwd / "sample.py").read_text() == "value=1\n"
            (cwd / "sample.py").write_text("value = 1\n")
        elif command[1] == "apply":
            assert "--check" in command
            assert input is not None
            assert b"-import os" in input
            assert b"+value = 1" in input
            code = int(self.invalid)
        return subprocess.CompletedProcess(command, code, stdout, b"")


def request(tmp_path: Path) -> PatchRequest:
    return PatchRequest(
        tested_sha=SHA,
        output_dir=tmp_path / "bundle",
        summary=tmp_path / "summary",
        run_url="https://github.com/example/project/actions/runs/123",
        command=["/fixture/ruff", "format", "--diff", "sample.py", "pyproject.toml"],
        safe_fixes=True,
    )


@pytest.mark.parametrize(
    ("lint_exit", "expected", "remaining"),
    [
        (0, PatchStatus.AVAILABLE, False),
        (1, PatchStatus.AVAILABLE, True),
        (2, PatchStatus.TOOL_FAILED, False),
    ],
)
def test_combined_patch_never_modifies_checkout_or_hides_remaining_lint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lint_exit: int,
    expected: PatchStatus,
    remaining: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "sample.py"
    source.write_text("import os\nvalue=1\n")
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n")
    runner = RepairTools(lint_exit=lint_exit)
    monkeypatch.setattr(subprocess, "run", runner.run)
    value = request(tmp_path)
    result = generate(value, collect_safe)
    assert source.read_text() == "import os\nvalue=1\n"
    assert result.remaining_lint is remaining
    assert (value.output_dir / "format.patch").exists() is (expected is PatchStatus.AVAILABLE)
    assert result.status is expected


def test_combined_patch_still_requires_applicability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sample.py").write_text("import os\nvalue=1\n")
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n")
    monkeypatch.setattr(subprocess, "run", RepairTools(invalid=True).run)
    value = request(tmp_path)
    assert generate(value, collect_safe).status is PatchStatus.INVALID_PATCH
    assert not (value.output_dir / "format.patch").exists()


@pytest.mark.parametrize("target", ["../outside.py", "/outside.py"])
def test_target_cannot_escape_the_isolated_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    runner = RepairTools()
    monkeypatch.setattr(subprocess, "run", runner.run)
    value = request(tmp_path).model_copy(update={"command": ["ruff", "format", "--diff", target]})
    assert generate(value, collect_safe).status is PatchStatus.TOOL_FAILED
    assert not any(command[1] in {"check", "format"} for command in runner.calls)


def test_symlink_target_never_reaches_mutating_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("unchanged\n")
    (tmp_path / "sample.py").symlink_to(outside)
    runner = RepairTools()
    monkeypatch.setattr(subprocess, "run", runner.run)
    value = request(tmp_path)
    assert generate(value, collect_safe).status is PatchStatus.TOOL_FAILED
    assert not any(command[1] in {"check", "format"} for command in runner.calls)
    assert outside.read_text() == "unchanged\n"


def test_timeout_during_safe_fix_never_publishes_partial_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sample.py").write_text("import os\nvalue=1\n")
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n")
    runner = RepairTools()

    def timeout(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        if command[1] == "check":
            raise subprocess.TimeoutExpired(command, 60)
        return runner.run(command, **options)

    monkeypatch.setattr(subprocess, "run", timeout)
    value = request(tmp_path)
    assert generate(value, collect_safe).status is PatchStatus.TIMEOUT
    assert not (value.output_dir / "format.patch").exists()


@pytest.mark.parametrize(
    ("exit_code", "payload"),
    [
        (0, b"bad json"),
        (1, b"[]"),
        (2, b"[]"),
        (0, b'[{"code":"F401","filename":"sample.py"}]'),
        (1, b'[{"code":"F401","filename":"../outside.py"}]'),
    ],
)
def test_final_verification_rejects_corrupt_or_contradictory_evidence(
    tmp_path: Path,
    exit_code: int,
    payload: bytes,
) -> None:
    value = request(tmp_path)
    value.output_dir.mkdir()
    runner = Mock(spec=Runner)
    runner.run.return_value = subprocess.CompletedProcess([], exit_code, payload, b"")
    evidence = PatchEvidence(tested_sha=SHA)
    assert not verify_remaining(value, runner, evidence, tmp_path)
    assert evidence.verification_exit_code == exit_code


def test_final_verification_can_clear_pre_format_lint(tmp_path: Path) -> None:
    value = request(tmp_path)
    value.output_dir.mkdir()
    runner = Mock(spec=Runner)
    runner.run.return_value = subprocess.CompletedProcess([], 0, b"[]", b"")
    evidence = PatchEvidence(tested_sha=SHA, lint_exit_code=1, remaining_lint=True)
    assert verify_remaining(value, runner, evidence, tmp_path)
    assert not evidence.remaining_lint
    assert evidence.remaining_count == 0
    assert evidence.lint_exit_code == 1


def test_final_verification_keeps_bounded_manual_work(tmp_path: Path) -> None:
    value = request(tmp_path)
    value.output_dir.mkdir()
    runner = Mock(spec=Runner)
    payload = json.dumps([{"code": "F401", "filename": "sample.py"}] * 60).encode()
    runner.run.return_value = subprocess.CompletedProcess([], 1, payload, b"")
    evidence = PatchEvidence(tested_sha=SHA)
    assert verify_remaining(value, runner, evidence, tmp_path)
    assert evidence.remaining_count == 60  # noqa: PLR2004 -- complete diagnostic count.
    assert len(evidence.remaining_issues) == 50  # noqa: PLR2004 -- bounded summary contract.
    assert evidence.remaining_lint


@pytest.mark.parametrize(
    "contents",
    [
        {"one.py": ("value=1\n", "value = 1\n"), "two.py": ("other=2\n", "other = 2\n")},
        {"one.py": ("value=1", "value = 1\n")},
        {"one.py": ("value=1\n", "value = 1")},
        {"one.py": ("value = 1\n", "value = 1\n")},
    ],
)
def test_generated_diff_applies_to_real_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: dict[str, tuple[str, str]],
) -> None:
    monkeypatch.chdir(tmp_path)
    copy = tmp_path / "copy"
    copy.mkdir()
    for name, (before, after) in contents.items():
        Path(name).write_text(before)
        (copy / name).write_text(after)
    patch = combined_diff([Path(name) for name in contents], copy)
    for name, (before, _) in contents.items():
        assert Path(name).read_text() == before
    if all(before == after for before, after in contents.values()):
        assert patch == b""
        return
    git = git_executable()
    for flags in (["--check"], []):
        applied = subprocess.run(  # noqa: S603 -- real Git on pytest-owned temporary files only.
            [git, "apply", "-p0", *flags, "-"],
            input=patch,
            cwd=tmp_path,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert applied.returncode == 0, applied.stderr
    for name, (_, after) in contents.items():
        assert Path(name).read_text() == after
