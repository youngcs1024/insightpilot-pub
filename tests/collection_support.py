"""Reject invalid partition markers before pytest deselects ordinary tests."""

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """A storage test cannot accidentally run as a pure test or disappear."""
    invalid = [
        item.nodeid
        for item in items
        if item.get_closest_marker("storage") is not None
        and item.get_closest_marker("integration") is None
    ]
    if invalid:
        raise pytest.UsageError("storage requires integration: " + ", ".join(invalid))

    invalid_e2e = [
        item.nodeid
        for item in items
        if item.get_closest_marker("e2e") is not None
        and item.get_closest_marker("external") is None
        and item.get_closest_marker("gpu") is None
        and (
            item.get_closest_marker("integration") is None
            or item.get_closest_marker("storage") is not None
        )
    ]
    if invalid_e2e:
        raise pytest.UsageError(
            "ordinary e2e requires integration without storage: " + ", ".join(invalid_e2e)
        )
