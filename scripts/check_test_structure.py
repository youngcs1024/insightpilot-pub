"""Statically enforce test-module boundaries without importing test code."""

import argparse
import ast
from enum import StrEnum
from importlib.util import resolve_name
from pathlib import Path, PurePosixPath

from pydantic import BaseModel


class Problem(StrEnum):
    """Stable structural failure categories."""

    TEST_IMPORT = "test_import"
    SYNTAX = "syntax_error"
    UNREADABLE = "unreadable_source"
    RELATIVE_IMPORT = "invalid_relative_import"


class Violation(BaseModel):
    """Boundaries are reported without copying source or exception prose."""

    path: str
    line: int
    problem: Problem
    target: str = ""


def import_targets(node: ast.AST, package: str) -> list[str]:
    """Resolve absolute and relative imports without loading any modules."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    module = node.module or ""
    if node.level:
        module = resolve_name("." * node.level + module, package)
    return [module, *(f"{module}.{alias.name}" for alias in node.names)]


def inspect_source(source: str, path: PurePosixPath, modules: set[str]) -> list[Violation]:
    """Inspect nested imports and report malformed source as a blocking defect."""
    name = path.as_posix()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Violation(path=name, line=exc.lineno or 1, problem=Problem.SYNTAX)]
    violations = []
    package = ".".join(path.parent.parts)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        try:
            targets = import_targets(node, package)
        except (ImportError, ValueError):
            violations.append(
                Violation(path=name, line=node.lineno, problem=Problem.RELATIVE_IMPORT)
            )
            continue
        matches = {
            module
            for target in targets
            for module in modules
            if target == module or target.startswith(module + ".")
        }
        violations.extend(
            Violation(path=name, line=node.lineno, problem=Problem.TEST_IMPORT, target=module)
            for module in sorted(matches)
        )
    return violations


def inspect_file(path: Path, root: Path, modules: set[str]) -> list[Violation]:
    """Unreadable or invalidly encoded files cannot silently disappear from the audit."""
    relative = PurePosixPath(path.relative_to(root).as_posix())
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return [Violation(path=str(relative), line=1, problem=Problem.UNREADABLE)]
    return inspect_source(source, relative, modules)


def inspect_repository(root: Path) -> list[Violation]:
    """Scan test and support source, including dedicated modules, without imports."""
    directory = root / "tests"
    files = sorted(directory.rglob("*.py"))
    if not files:
        return [Violation(path="tests", line=1, problem=Problem.UNREADABLE)]
    modules = {
        ".".join(path.relative_to(root).with_suffix("").parts)
        for path in files
        if path.name.startswith("test_")
    }
    return [issue for path in files for issue in inspect_file(path, root, modules)]


def main() -> None:
    """Report all violations with a failing process exit, without running pytest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    issues = inspect_repository(args.root)
    for issue in issues:
        print(f"{issue.path}:{issue.line}: {issue.problem.value} {issue.target}".rstrip())
    print(f"Test structure: {'FAIL' if issues else 'PASS'}; violations={len(issues)}")
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
