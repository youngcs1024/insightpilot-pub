"""Collect six independent Linux dependency resolutions without installing packages."""

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import tomllib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = "x86_64-manylinux_2_28"
PROJECTS = {"api": Path("."), "mcp": Path("mcp_server"), "model-runtime": Path("model_runtime")}
PYTHONS = ("3.12", "3.13")
REQUIRED_WHEELS = frozenset({"torch", "tokenizers"})


class Status(StrEnum):
    """Resolution outcomes, independent of resolver error prose."""

    RESOLVED = "resolved"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    UNSUPPORTED_SOURCE = "unsupported_source"


class Project(BaseModel):
    """The dependency input owned by one runtime."""

    dependencies: list[str] = Field(min_length=1)


class Manifest(BaseModel):
    """Relevant standard project metadata; tooling settings are intentionally ignored."""

    project: Project


class Artifact(BaseModel):
    """Resolver-provided immutable distribution identity."""

    url: str
    hashes: dict[str, str]


class Package(BaseModel):
    """Artifacts already filtered by uv for the requested Python and platform."""

    name: str
    version: str
    wheels: list[Artifact] = Field(default_factory=list)
    sdist: Artifact | None = None


class PyLock(BaseModel):
    """The package portion of uv's PEP 751 output."""

    packages: list[Package] = Field(min_length=1)


class Cell(BaseModel):
    """One executed resolution and its artifact audit."""

    service: str
    python: str
    command: list[str]
    input_sha256: str
    status: Status
    exit_code: int
    log: str
    lock: str | None = None
    package_count: int = 0
    wheel_count: int = 0
    source_packages: list[str] = Field(default_factory=list)


class Report(BaseModel):
    """Machine-readable evidence; successful resolution never asserts CUDA readiness."""

    schema_version: int = 1
    started_at: datetime
    uv_version: str
    platform: str = PLATFORM
    cuda_verified: bool = False
    cells: list[Cell]


class RunConfig(BaseModel):
    """Explicit paths and execution deadline for a matrix run."""

    uv: str
    root: Path
    run: Path
    timeout: int = Field(ge=1, le=1800)


def isolated_environment(root: Path) -> dict[str, str]:
    """Discard foreign interpreter, resolver and package-index selectors."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("UV_", "PYTHON", "PIP_")) and key != "VIRTUAL_ENV"
    }
    env.update(
        UV_CACHE_DIR=str(root / ".uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(root / ".uv-cache/python"),
        UV_HTTP_TIMEOUT="30",
        UV_HTTP_RETRIES="2",
        UV_NO_PROGRESS="1",
    )
    return env


def compile_cell(config: RunConfig, service: str, python: str) -> Cell:
    """Resolve one input, retaining failures and never piping away its exit code."""
    manifest_bytes = (config.root / PROJECTS[service] / "pyproject.toml").read_bytes()
    manifest = Manifest.model_validate(tomllib.loads(manifest_bytes.decode()))
    source = config.run / f"{service}-{python}.in"
    source.write_text("\n".join(manifest.project.dependencies) + "\n")
    # Explicit pyproject inputs preserve per-package uv sources (notably CUDA wheels).
    snapshot = config.run / f"{service}-{python}" / "pyproject.toml"
    snapshot.parent.mkdir()
    snapshot.write_bytes(manifest_bytes)
    output = config.run / f"pylock.{service}-{python}.toml"
    log = config.run / f"{service}-{python}.log"
    command = [
        config.uv,
        "--no-config",
        "pip",
        "compile",
        str(snapshot),
        "--python-version",
        python,
        "--python",
        python,
        "--python-platform",
        PLATFORM,
        "--default-index",
        "https://pypi.org/simple",
        "--format",
        "pylock.toml",
        "--output-file",
        str(output),
        "--color",
        "never",
    ]
    status = Status.FAILED
    with log.open("w") as stream:
        try:
            result = subprocess.run(  # noqa: S603 -- fixed executable and structured arguments.
                command,
                cwd=config.root,
                env=isolated_environment(config.root),
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=config.timeout,
                check=False,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            status, code = Status.TIMED_OUT, 124
            stream.write("\nCompatibility runner deadline exceeded.\n")
    cell = Cell(
        service=service,
        python=python,
        command=command,
        input_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        status=status,
        exit_code=code,
        log=log.name,
    )
    if code != 0:
        return cell
    return audit_artifacts(cell, output)


def audit_artifacts(cell: Cell, output: Path) -> Cell:
    """Reject source-only torch/tokenizers even if dependency resolution succeeded."""
    lock = PyLock.model_validate(tomllib.loads(output.read_text()))
    cell.lock = output.name
    cell.package_count = len(lock.packages)
    cell.wheel_count = sum(bool(package.wheels) for package in lock.packages)
    cell.source_packages = [package.name for package in lock.packages if not package.wheels]
    cell.status = Status.RESOLVED
    if cell.service == "model-runtime" and REQUIRED_WHEELS.intersection(cell.source_packages):
        cell.status = Status.UNSUPPORTED_SOURCE
    return cell


def run_matrix(uv: str, root: Path, run: Path, timeout: int) -> Report:
    """Collect the entire matrix, including rows after an earlier failure."""
    started = datetime.now(UTC)
    version = subprocess.run(  # noqa: S603 -- caller supplies the discovered uv executable.
        [uv, "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        env=isolated_environment(root),
    ).stdout.strip()
    cells = []
    config = RunConfig(uv=uv, root=root, run=run, timeout=timeout)
    for python in PYTHONS:
        for service in PROJECTS:
            print(f"Resolving {service} / Python {python}", flush=True)
            cell = compile_cell(config, service, python)
            cells.append(cell)
            print(f"  {cell.status}: resolver exit {cell.exit_code}; log {cell.log}", flush=True)
    report = Report(started_at=started, uv_version=version, cells=cells)
    (run / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    return report


def main() -> int:
    """Write evidence into a fresh directory; never overwrite an earlier run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / ".uv-cache/compat")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 1800:  # noqa: PLR2004 -- CLI safety bounds.
        parser.error("--timeout-seconds must be between 1 and 1800")
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv 0.11.32 is required")
    args.out.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=args.out.resolve()))
    report = run_matrix(uv, ROOT, run, args.timeout_seconds)
    print(f"Evidence: {run}")
    return int(any(cell.status != Status.RESOLVED for cell in report.cells))


if __name__ == "__main__":
    raise SystemExit(main())
