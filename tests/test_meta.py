"""Executable architecture contracts across all application Python modules."""

from pathlib import Path

from tests.architecture_rules import violations

ROOT = Path(__file__).resolve().parents[1]


def assert_rule(rule: str) -> None:
    failures = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in sorted((ROOT / "app").rglob("*.py"))
        for line in violations(path.read_text(), path.relative_to(ROOT).as_posix())[rule]
    ]
    assert not failures, f"{rule}: {', '.join(failures)}"


def test_no_bare_create_task() -> None:
    assert_rule("no_bare_create_task")


def test_no_detail_str_e() -> None:
    assert_rule("no_detail_str_e")


def test_no_sync_engine_in_app() -> None:
    assert_rule("no_sync_engine_in_app")


def test_nodes_do_no_io() -> None:
    assert_rule("nodes_do_no_io")
