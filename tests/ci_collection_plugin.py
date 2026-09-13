"""Explicit CI-only collection reporter; never registered during ordinary test runs."""

from collections.abc import Generator
from pathlib import Path

import pytest

from scripts.ci_collection import MAX_COLLECTORS, MAX_NODE_LENGTH, CollectionReport

REPORT_KEY = pytest.StashKey[CollectionReport]()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Require an explicit output path when the workflow loads this plugin."""
    parser.addoption("--collection-report", type=Path, default=None)


def pytest_configure(config: pytest.Config) -> None:
    """Keep collection state scoped to this pytest invocation."""
    config.stash[REPORT_KEY] = CollectionReport(raw_exit=0, collected=0, failed_count=0)


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    """Read structured outcomes without interpreting exception prose."""
    report = yield
    if report.failed:
        value = collector.config.stash[REPORT_KEY]
        value.failed_count += 1
        if len(value.failed_nodes) < MAX_COLLECTORS:
            value.failed_nodes.append(report.nodeid[:MAX_NODE_LENGTH])
    return report


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write after collection; pytest retains its original exit status."""
    output = session.config.getoption("collection_report")
    if output is None:
        return
    value = session.config.stash[REPORT_KEY]
    value.raw_exit = int(exitstatus)
    value.collected = session.testscollected
    Path(output).write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")
