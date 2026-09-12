"""Compose safe Ruff fixes and formatting in a disposable copy of tracked targets."""

import argparse
import difflib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import TypeAdapter, ValidationError

from scripts.ci_format_patch import (
    LintIssue,
    PatchEvidence,
    PatchRequest,
    PatchStatus,
    Runner,
    generate,
    inspect_checkout,
)

MAX_SNAPSHOT_BYTES = 100_000_000
MAX_SNAPSHOT_FILES = 10_000


def snapshot(request: PatchRequest, runner: Runner, destination: Path) -> list[Path]:
    """Copy tracked regular files only; never follow a repository symlink."""
    result = runner.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            *request.command[3:],
            ".gitignore",
            "ruff.toml",
            ".ruff.toml",
        ]
    )
    if result.returncode:
        return []
    names = [Path(name) for name in result.stdout.decode().split("\0") if name]
    if len(names) > MAX_SNAPSHOT_FILES:
        return []
    total = 0
    for name in names:
        if (
            name.is_absolute()
            or ".." in name.parts
            or any(p.is_symlink() for p in (name, *name.parents))
        ):
            return []
        total += name.stat().st_size
        if not name.is_file() or total > MAX_SNAPSHOT_BYTES:
            return []
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    return names


def combined_diff(names: list[Path], destination: Path) -> bytes:
    """Produce one applicable patch, including files without a final newline."""
    chunks: list[str] = []
    for name in names:
        before, after = name.read_bytes(), (destination / name).read_bytes()
        if before == after:
            continue
        chunks.extend(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in difflib.unified_diff(
                before.decode().splitlines(keepends=True),
                after.decode().splitlines(keepends=True),
                fromfile=name.as_posix(),
                tofile=name.as_posix(),
            )
        )
    return "".join(chunks).encode()


def collect_safe(request: PatchRequest, runner: Runner, evidence: PatchEvidence) -> bytes:
    """Only isolated safe fixes may write; the source checkout receives check-only Git calls."""
    if any(Path(arg).is_absolute() or ".." in Path(arg).parts for arg in request.command[3:]):
        return b""
    executable = str(Path(request.command[0]).resolve())
    version = runner.run([executable, "--version"])
    if version.returncode:
        return b""
    evidence.tool_version = version.stdout.decode().strip()[:200]
    with TemporaryDirectory(prefix="ci-ruff-") as directory:
        destination = Path(directory)
        names = snapshot(request, runner, destination)
        if not names:
            return b""
        patch = repair_copy(request, runner, evidence, destination, names)
        if patch is None:
            return b""

    return validated_patch(request, runner, evidence, patch)


def validated_patch(
    request: PatchRequest, runner: Runner, evidence: PatchEvidence, patch: bytes
) -> bytes:
    """Publish only a complete diff applicable to the same clean source commit."""
    problem = inspect_checkout(runner, request.tested_sha)
    if problem is not None:
        evidence.status = problem
        return b""
    if not patch:
        evidence.status = PatchStatus.NO_DIFF
        return b""
    if runner.run(["git", "apply", "-p0", "--check", "-"], data=patch).returncode:
        evidence.status = PatchStatus.INVALID_PATCH
        return b""
    problem = inspect_checkout(runner, request.tested_sha)
    if problem is not None:
        evidence.status = problem
        return b""
    evidence.status = PatchStatus.AVAILABLE
    return patch


def repair_copy(
    request: PatchRequest,
    runner: Runner,
    evidence: PatchEvidence,
    destination: Path,
    names: list[Path],
) -> bytes | None:
    """Run only the pinned formatter's safe edit modes inside the snapshot."""
    executable = str(Path(request.command[0]).resolve())
    if runner.run(["git", "init", "--quiet", str(destination)]).returncode:
        return None
    fixed = runner.run(
        [executable, "check", "--fix", "--no-unsafe-fixes", *request.command[3:]],
        cwd=destination,
    )
    evidence.remaining_lint = fixed.returncode == 1
    evidence.lint_exit_code = fixed.returncode
    with runner.log.open("ab") as stream:
        stream.write(fixed.stdout)
    if fixed.returncode not in {0, 1}:
        return None
    formatted = runner.run([executable, "format", *request.command[3:]], cwd=destination)
    evidence.exit_code = formatted.returncode
    evidence.format_exit_code = formatted.returncode
    if formatted.returncode:
        return None
    if not verify_remaining(request, runner, evidence, destination):
        return None
    return combined_diff(names, destination)


def verify_remaining(
    request: PatchRequest, runner: Runner, evidence: PatchEvidence, destination: Path
) -> bool:
    """Verify the final copy, reporting manual work rather than the pre-format state."""
    result = runner.run(
        [
            str(Path(request.command[0]).resolve()),
            "check",
            "--output-format=json",
            *request.command[3:],
        ],
        cwd=destination,
    )
    evidence.verification_exit_code = result.returncode
    (request.output_dir / "remaining-lint.json").write_bytes(result.stdout)
    if result.returncode not in {0, 1}:
        return False
    try:
        issues = TypeAdapter(list[LintIssue]).validate_json(result.stdout)
        for issue in issues:
            path = Path(issue.filename)
            path = path if path.is_absolute() else destination / path
            issue.filename = path.resolve().relative_to(destination.resolve()).as_posix()
    except (ValueError, ValidationError):
        return False
    if bool(result.returncode) != bool(issues):
        return False
    evidence.remaining_lint = bool(issues)
    evidence.remaining_count = len(issues)
    evidence.remaining_issues = issues[:50]
    return True


def main() -> None:
    """Keep the legacy bundle name while offering one combined safe repair patch."""
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
        safe_fixes=True,
    )
    evidence = generate(request, collect_safe)
    raise SystemExit(0 if evidence.status in {PatchStatus.AVAILABLE, PatchStatus.NO_DIFF} else 1)


if __name__ == "__main__":
    main()
