"""Explicit evidence selection for dedicated tests, with no import-time I/O."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--rerank-evidence", default=None)
    parser.addoption("--client-sha", default=None)
    parser.addoption("--retrieval-evidence", default=None)
