"""Shared helpers must not import collected test modules, even through relative imports."""

from pathlib import Path, PurePosixPath

import pytest

from scripts.check_test_structure import inspect_file, inspect_repository, inspect_source


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
    assert inspect_source(
        source, PurePosixPath("tests/unit/test_consumer.py"), {"tests.unit.test_example"}
    )


def test_import_guard_allows_support_symbols_and_ignores_prose() -> None:
    source = (
        "from tests.unit.support import test_factory\n"
        "from .support import factory\n"
        "# from tests.unit.test_example import factory\n"
        "value = 'tests.unit.test_example'\n"
    )
    assert not inspect_source(
        source, PurePosixPath("tests/unit/test_consumer.py"), {"tests.unit.test_example"}
    )


def test_repository_helpers_never_import_test_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    assert not inspect_repository(root)


@pytest.mark.parametrize(
    ("source", "problem"),
    [("def broken(:", "syntax_error"), ("from .... import thing", "invalid_relative_import")],
)
def test_invalid_source_is_blocking(source: str, problem: str) -> None:
    issues = inspect_source(source, PurePosixPath("tests/unit/support.py"), set())
    assert len(issues) == 1
    assert issues[0].problem.value == problem
    assert issues[0].line == 1


def test_unreadable_source_is_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(self: Path, **kwargs: object) -> str:
        raise PermissionError

    monkeypatch.setattr(Path, "read_text", denied)
    issues = inspect_file(tmp_path / "tests/support.py", tmp_path, set())
    assert issues[0].problem.value == "unreadable_source"


def test_static_audit_never_executes_source_and_reports_targets(tmp_path: Path) -> None:
    directory = tmp_path / "tests"
    directory.mkdir()
    (directory / "test_origin.py").write_text("raise RuntimeError('must not execute')\n")
    (directory / "support.py").write_text("from .test_origin import factory\n")
    issues = inspect_repository(tmp_path)
    assert len(issues) == 1
    assert issues[0].path == "tests/support.py"
    assert issues[0].line == 1
    assert issues[0].target == "tests.test_origin"


def test_absent_source_and_invalid_encoding_cannot_pass(tmp_path: Path) -> None:
    assert inspect_repository(tmp_path)
    directory = tmp_path / "tests"
    directory.mkdir()
    (directory / "support.py").write_bytes(b"\xff")
    assert inspect_repository(tmp_path)[0].problem.value == "unreadable_source"
