"""Exercise command guards without starting external services."""

import os
import shlex
from pathlib import Path

import pytest

from tests.tooling.support import ROOT, run_make


@pytest.fixture
def probe_makefile(tmp_path: Path) -> Path:
    """Exercise the real Make variables and uv wrapper without running a quality tool."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        """import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
assert Path.cwd().resolve() == root
assert Path(sys.prefix).resolve() == root / ".venv"
assert Path(sys.executable).parent == root / ".venv" / "bin"
expected = {
    "UV_PROJECT_ENVIRONMENT": root / ".venv",
    "UV_CACHE_DIR": root / ".uv-cache",
    "UV_PYTHON_INSTALL_DIR": root / ".uv-cache" / "python",
    "VIRTUAL_ENV": root / ".venv",
}
for key, value in expected.items():
    assert Path(os.environ[key]).resolve() == value, key
for key in (
    "UV_PROJECT", "UV_WORKING_DIRECTORY", "PYTHONPATH", "PYTHONHOME",
    "PYTHONUSERBASE", "PIP_CONFIG_FILE",
):
    assert key not in os.environ, key
assert os.environ["UV_HTTP_TIMEOUT"] == "30"
assert os.environ["UV_HTTP_RETRIES"] == "2"
print("isolated runtime confirmed")
""",
        encoding="utf-8",
    )
    makefile = tmp_path / "probe.mk"
    makefile.write_text(
        f"include {ROOT}/Makefile\n"
        ".PHONY: probe\n"
        "probe: check-env\n"
        f"\t$(IP_RUN) python {shlex.quote(str(probe))} {shlex.quote(str(ROOT))}\n",
        encoding="utf-8",
    )
    return makefile


def test_default_target_is_help() -> None:
    """The default invocation only displays project help."""
    result = run_make()
    assert result.returncode == 0
    assert "InsightPilot — Step 1.3" in result.stdout


@pytest.mark.parametrize("environment", ["development", "staging", "production", "test"])
def test_valid_environment(environment: str) -> None:
    """Accept only the documented environment names."""
    assert run_make("help", f"ENV={environment}").returncode == 0


def test_invalid_environment_rejected() -> None:
    """Reject an unsupported environment before running a tool."""
    result = run_make("install", "ENV=foreign")
    assert result.returncode != 0
    assert "Invalid ENV" in result.stderr
    assert "uv " not in result.stdout


@pytest.mark.parametrize("target", ["lint-check", "format-check", "lint"])
def test_quality_entrypoints_preserve_locked_scope(target: str) -> None:
    result = run_make("-n", target, "ENV=test")
    assert result.returncode == 0
    assert "run --locked --no-env-file ruff" in result.stdout
    assert (
        "app model_tunnel model_runtime tests scripts spikes alembic data mcp_server"
        in result.stdout
    )
    assert ("ruff check " in result.stdout) is (target != "format-check")
    assert ("ruff format --check " in result.stdout) is (target != "lint-check")
    assert run_make(target, "ENV=foreign").returncode != 0


def test_eval_selects_locked_live_entrypoint() -> None:
    """Dry-run the command without a live model or database."""
    result = run_make("-n", "eval", "s=nl2sql", "ENV=test")
    assert result.returncode == 0
    assert "run --locked --no-env-file python -m evals.cli run --suite 'nl2sql'" in result.stdout


def test_clean_preserves_files(tmp_path: Path) -> None:
    """Cleanup remains an informational command."""
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    result = run_make("clean", cwd=tmp_path)
    assert result.returncode == 0
    assert "No files deleted" in result.stdout
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_foreign_environment_cannot_redirect_tools(tmp_path: Path, probe_makefile: Path) -> None:
    """Run from elsewhere with foreign uv and Python settings safely."""
    foreign = tmp_path / "foreign"
    env = dict(os.environ)
    for key in (
        "UV_PROJECT_ENVIRONMENT",
        "UV_CACHE_DIR",
        "UV_PROJECT",
        "UV_WORKING_DIRECTORY",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PIP_CONFIG_FILE",
    ):
        env[key] = str(foreign)
    env["COMPOSE_PROJECT_NAME"] = "pathfinder"
    result = run_make("probe", cwd=tmp_path, env=env, makefile=probe_makefile)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'bash "{ROOT}/scripts/uv_project.sh" api' in result.stdout
    assert "isolated runtime confirmed" in result.stdout
    assert not foreign.exists()


def test_command_line_cannot_override_environment_path(
    tmp_path: Path, probe_makefile: Path
) -> None:
    """Make variable overrides cannot select another project's environment."""
    foreign = tmp_path / "foreign"
    result = run_make(
        "probe",
        f"UV_PROJECT_ENVIRONMENT={foreign}",
        f"UV_CACHE_DIR={foreign}",
        f"IP_ROOT={foreign}",
        f"IP_UV={foreign}",
        f"IP_RUN={foreign}",
        cwd=tmp_path,
        makefile=probe_makefile,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "isolated runtime confirmed" in result.stdout
    assert not foreign.exists()


def test_typecheck_entrypoint_preserves_full_locked_scope() -> None:
    """Full mypy acceptance belongs to quality, independent of isolation probes."""
    result = run_make("--dry-run", "typecheck", "ENV=test")
    assert result.returncode == 0
    assert result.stdout.splitlines()[-1] == (
        f'bash "{ROOT}/scripts/uv_project.sh" api run --locked --no-env-file mypy'
    )
    assert run_make("typecheck", "ENV=foreign").returncode != 0


def test_dev_uses_isolated_runtime_and_loopback() -> None:
    """Inspect the real recipe without starting a reload server in the test runner."""
    result = run_make("--dry-run", "dev", "ENV=test")
    assert result.returncode == 0
    assert 'IP_ENVIRONMENT="test"' in result.stdout
    assert 'scripts/uv_project.sh" api' in result.stdout
    assert "uvicorn app.main:app --host 127.0.0.1 --port 18000 --reload" in result.stdout
    assert "--timeout-graceful-shutdown 10" in result.stdout


def test_migration_commands_select_target_without_shell_evaluation() -> None:
    """Inspect shell quoting for both a selected history and a punctuation-heavy message."""
    result = run_make("--dry-run", "migration", "db=business", "m=quote ' and `literal` text")
    assert result.returncode == 0
    assert "scripts.migrate_all revision --database 'business'" in result.stdout
    assert "--message 'quote '" in result.stdout
    assert "`literal`" in result.stdout
    upgrade = run_make("--dry-run", "migrate")
    assert "scripts.migrate_all upgrade" in upgrade.stdout


def test_migration_requires_message_and_valid_target() -> None:
    """Invalid arguments fail before any migration connection or file write."""
    assert "requires a nonempty --message" in run_make("migration").stderr
    assert "invalid choice" in run_make("migration", "db=foreign", "m=test").stderr


def test_seed_recipe_uses_process_configuration() -> None:
    result = run_make("--dry-run", "seed", "ENV=test")
    assert "python -m data.seed.generate" in result.stdout
    assert 'IP_ENVIRONMENT="test"' in result.stdout
    assert "python -m scripts.seed" in result.stdout


def test_type_report_entrypoint_uses_same_locked_environment() -> None:
    result = run_make("--dry-run", "typecheck-report", "ENV=test", "CI_TYPES_OUTPUT=type-evidence")
    assert result.returncode == 0
    assert result.stdout.splitlines()[-1] == (
        f'bash "{ROOT}/scripts/uv_project.sh" api run --locked --no-env-file '
        'python -m scripts.ci_types --output-dir "type-evidence"'
    )
    assert run_make("typecheck-report", "ENV=foreign").returncode != 0
