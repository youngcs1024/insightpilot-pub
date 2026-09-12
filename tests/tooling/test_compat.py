"""Exercise resolver failures and isolation without network access or GPU imports."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_compat import (
    PROJECTS,
    Cell,
    RunConfig,
    Status,
    audit_artifacts,
    compile_cell,
    isolated_environment,
    run_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
SUCCESS_LOCK = """lock-version = "1.0"
[[packages]]
name = "example"
version = "1.0"
wheels = [{url = "https://example.invalid/example.whl", hashes = {sha256 = "abc"}}]
"""


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    """Build three minimal independent inputs in a disposable test directory."""
    for folder in PROJECTS.values():
        directory = tmp_path / folder
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "pyproject.toml").write_text('[project]\ndependencies = ["example==1.0"]\n')
    return tmp_path


@pytest.fixture
def fake_uv(tmp_path: Path) -> Path:
    """Provide a real subprocess that fails one cell and writes valid artifacts for the rest."""
    executable = tmp_path / "uv"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys, time\n"
        "args = sys.argv[1:]\n"
        "if '--version' in args:\n"
        "    print('uv 0.11.32')\n"
        "    sys.exit(0)\n"
        "source = pathlib.Path(args[args.index('compile') + 1])\n"
        "if 'hang==1' in source.read_text():\n"
        "    time.sleep(5)\n"
        "if source.parent.name == 'api-3.12':\n"
        "    print('Intentional resolver failure', file=sys.stderr)\n"
        "    sys.exit(7)\n"
        "out = pathlib.Path(args[args.index('--output-file') + 1])\n"
        f"out.write_text({SUCCESS_LOCK!r})\n"
    )
    executable.chmod(0o755)
    return executable


def test_failure_keeps_all_six_cells(fake_project: Path, fake_uv: Path) -> None:
    """A nonzero resolver exit is retained and later combinations still execute."""
    run = fake_project / "evidence"
    run.mkdir()
    report = run_matrix(str(fake_uv), fake_project, run, 10)
    assert len(report.cells) == 6  # noqa: PLR2004 -- the required matrix size.
    failed = report.cells[0]
    assert failed.status == Status.FAILED
    assert failed.exit_code == 7  # noqa: PLR2004 -- fake resolver's distinctive exit code.
    assert failed.lock is None
    assert "Intentional resolver failure" in (run / failed.log).read_text()
    assert all(cell.status == Status.RESOLVED for cell in report.cells[1:])
    assert not report.cuda_verified
    assert json.loads((run / "report.json").read_text())["cells"][0]["exit_code"] == 7  # noqa: PLR2004


def test_timeout_is_recorded(fake_project: Path, fake_uv: Path) -> None:
    """Kill a stalled resolver at the deadline and preserve an explicit timeout outcome."""
    (fake_project / "pyproject.toml").write_text('[project]\ndependencies = ["hang==1"]\n')
    config = RunConfig(uv=str(fake_uv), root=fake_project, run=fake_project, timeout=1)
    cell = compile_cell(config, "api", "3.12")
    assert cell.status == Status.TIMED_OUT
    assert cell.exit_code == 124  # noqa: PLR2004 -- conventional timeout exit.
    assert "deadline exceeded" in (fake_project / cell.log).read_text()


def test_named_cuda_source_is_preserved(fake_project: Path, fake_uv: Path) -> None:
    """Do not accidentally test a PyPI build after selecting a different CUDA index."""
    manifest = fake_project / "model_runtime/pyproject.toml"
    manifest.write_text(
        manifest.read_text()
        + '\n[tool.uv.sources]\ntorch = {index = "pytorch-cu126"}\n'
        + '[[tool.uv.index]]\nname = "pytorch-cu126"\n'
        + 'url = "https://download.pytorch.org/whl/cu126"\nexplicit = true\n'
    )
    config = RunConfig(uv=str(fake_uv), root=fake_project, run=fake_project, timeout=10)
    cell = compile_cell(config, "model-runtime", "3.13")
    source = Path(cell.command[cell.command.index("compile") + 1])
    assert source.name == "pyproject.toml"
    assert source.read_bytes() == manifest.read_bytes()
    assert cell.status == Status.RESOLVED


@pytest.mark.parametrize("name", ["torch", "tokenizers"])
@pytest.mark.parametrize("has_wheel", [True, False])
def test_required_model_wheels(tmp_path: Path, name: str, *, has_wheel: bool) -> None:
    """A source fallback fails only when no compatible wheel is available."""
    output = tmp_path / "pylock.toml"
    content = f'[[packages]]\nname = "{name}"\nversion = "1.0"\n'
    content += (
        'sdist = {url = "https://example.invalid/source.tar.gz", hashes = {sha256 = "abc"}}\n'
    )
    if has_wheel:
        content += (
            'wheels = [{url = "https://example.invalid/model.whl", hashes = {sha256 = "abc"}}]\n'
        )
    output.write_text(content)
    cell = Cell(
        service="model-runtime",
        python="3.12",
        command=[],
        input_sha256="abc",
        status=Status.FAILED,
        exit_code=0,
        log="resolver.log",
    )
    result = audit_artifacts(cell, output)
    assert result.status == (Status.RESOLVED if has_wheel else Status.UNSUPPORTED_SOURCE)
    assert result.source_packages == ([] if has_wheel else [name])


def test_foreign_resolver_settings_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Foreign indexes, overrides and interpreters cannot change the compatibility evidence."""
    for key in (
        "UV_PROJECT",
        "UV_WORKING_DIR",
        "UV_WORKING_DIRECTORY",
        "UV_CONFIG_FILE",
        "UV_INDEX",
        "UV_OVERRIDE",
        "UV_TORCH_BACKEND",
        "UV_PROJECT_ENVIRONMENT",
        "PIP_INDEX_URL",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    ):
        monkeypatch.setenv(key, "foreign")
    env = isolated_environment(tmp_path)
    assert "foreign" not in env.values()
    assert env["UV_CACHE_DIR"] == str(tmp_path / ".uv-cache")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / ".uv-cache/python")


def test_cli_failure_and_fresh_evidence(fake_project: Path, fake_uv: Path) -> None:
    """The shell entrypoint exits nonzero and repeat runs cannot reuse stale successful outputs."""
    scripts = fake_project / "scripts"
    scripts.mkdir()
    for name in ("check_compat.sh", "check_compat.py"):
        (scripts / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    binary = fake_project / ".venv/bin"
    binary.mkdir(parents=True)
    launcher = binary / "python"
    launcher.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    launcher.chmod(0o755)
    env = dict(os.environ, PATH=f"{fake_uv.parent}:{os.environ['PATH']}")
    for _ in range(2):
        result = subprocess.run(  # noqa: S603 -- project script and test-controlled paths.
            ["/bin/bash", str(scripts / "check_compat.sh")],
            cwd=fake_project.parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        assert result.returncode == 1, result.stderr
        assert "Evidence:" in result.stdout, result.stderr
    runs = list((fake_project / ".uv-cache/compat").glob("run-*"))
    assert len(runs) == 2  # noqa: PLR2004 -- one directory per invocation.
    assert all((run / "report.json").is_file() for run in runs)
