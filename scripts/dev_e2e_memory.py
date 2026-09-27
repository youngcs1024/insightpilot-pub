"""Run the isolated deterministic two-conversation acceptance, including real MCP and SQL."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = "tests/integration/test_metric_override.py::test_override_changes_date_field_in_sql"


def main() -> int:
    """Use the same fixture-owned scenario as CI and preserve its exit status and report."""
    result = subprocess.run(  # noqa: S603 -- fixed interpreter/module/case; no shell.
        [sys.executable, "-m", "pytest", CASE, "-v", "-s", "--no-cov", "-m", "integration"],
        cwd=ROOT, check=False, timeout=600,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
