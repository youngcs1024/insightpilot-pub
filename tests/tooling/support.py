"""Existing tooling fixture helpers, independent of pytest test modules."""

import re
import shutil
import subprocess
from pathlib import Path

from scripts.ci_policy import Plan

ROOT = Path(__file__).resolve().parents[2]
MAKE = shutil.which("make")
SHA = "a" * 40


def run_make(
    *args: str,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    makefile: Path = ROOT / "Makefile",
) -> subprocess.CompletedProcess[str]:
    """Run the project Makefile with a bounded execution time."""
    assert MAKE is not None
    return subprocess.run(  # noqa: S603 -- fixed Makefile; test-controlled arguments, no shell.
        [MAKE, "--no-print-directory", "-f", str(makefile), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def git_executable() -> str:
    """Resolve the real fixture tool rather than suppressing partial-path diagnostics."""
    git = shutil.which("git")
    assert git is not None, "Git is required for patch applicability evidence"
    return git


def needs_for(plan: Plan) -> dict[str, dict[str, object]]:
    needs: dict[str, dict[str, object]] = {
        name: {"result": "success" if selected else "skipped"}
        for name, selected in {
            "changes": True,
            "quality": plan.checks,
            "unit": plan.checks,
            "integration": plan.checks,
            "storage": plan.checks,
            "coverage": plan.checks,
            "build": bool(plan.images),
        }.items()
    }
    needs["quality"]["outputs"] = {"collection_outcome": "success" if plan.checks else ""}
    return needs


def condition_holds(expression: str, values: dict[str, str], *, cancelled: bool = False) -> bool:
    """Evaluate the workflow's conjunction subset with explicit Actions status semantics.

    Reject unsupported syntax instead of evaluating arbitrary Python or silently
    approximating a new Actions condition. Explicit status checks remove the
    implicit success() that would otherwise skip jobs after a failed dependency.
    """
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    terms = [term.strip() for term in expression.split("&&")]
    assert any(term in {"!cancelled()", "always()"} for term in terms)
    outcomes = []
    for term in terms:
        if term in {"!cancelled()", "always()"}:
            outcomes.append(term == "always()" or not cancelled)
            continue
        match = re.fullmatch(r"([\w.]+)\s*(==|!=)\s*'([^']*)'", term)
        assert match is not None, f"Unsupported condition: {term}"
        field, operator, expected = match.groups()
        equal = values.get(field, "") == expected
        outcomes.append(equal if operator == "==" else not equal)
    return all(outcomes)
