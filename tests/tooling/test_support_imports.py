"""Shared helpers must not import collected test modules, even through relative imports."""

import ast
from importlib.util import resolve_name
from pathlib import Path, PurePosixPath

import pytest


def _import_targets(node: ast.AST, package: str) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    module = node.module or ""
    if node.level:
        module = resolve_name("." * node.level + module, package)
    return [module, *(f"{module}.{alias.name}" for alias in node.names)]


def _test_import_lines(source: str, path: PurePosixPath, modules: set[str]) -> list[int]:
    package = ".".join(path.parent.parts)
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if any(
            target == module or target.startswith(module + ".")
            for target in _import_targets(node, package)
            for module in modules
        )
    ]


@pytest.mark.parametrize(
    "source",
    [
        "from tests.unit.test_example import factory",
        "import tests.unit.test_example as example",
        "from .test_example import factory",
        "from . import test_example as example",
        "from ..unit import test_example",
        "def build():\n    from .test_example import factory\n    return factory()",
    ],
)
def test_import_guard_resolves_absolute_relative_and_nested_imports(source: str) -> None:
    assert _test_import_lines(
        source, PurePosixPath("tests/unit/test_consumer.py"), {"tests.unit.test_example"}
    )


def test_import_guard_allows_support_symbols_and_ignores_prose() -> None:
    source = (
        "from tests.unit.support import test_factory\n"
        "from .support import factory\n"
        "# from tests.unit.test_example import factory\n"
        "value = 'tests.unit.test_example'\n"
    )
    assert not _test_import_lines(
        source, PurePosixPath("tests/unit/test_consumer.py"), {"tests.unit.test_example"}
    )


def test_repository_helpers_never_import_test_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "tests").rglob("*.py"))
    modules = {
        ".".join(path.relative_to(root).with_suffix("").parts)
        for path in files
        if path.name.startswith("test_")
    }
    violations = {
        path.relative_to(root).as_posix(): lines
        for path in files
        if (
            lines := _test_import_lines(
                path.read_text(), PurePosixPath(path.relative_to(root).as_posix()), modules
            )
        )
    }
    assert not violations, violations
