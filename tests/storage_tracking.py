"""Remember raw testcase failures for resource finalizers without copying exception text."""

from collections.abc import Generator

import pytest

FAILED = pytest.StashKey[bool]()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Setup/call/teardown outcomes remain authoritative pytest outcomes."""
    report = yield
    if report.failed:
        item.stash[FAILED] = True
    return report
