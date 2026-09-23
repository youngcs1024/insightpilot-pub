"""All stack setup is fixture-owned, never triggered during collection."""

from collections.abc import AsyncIterator

import pytest

from tests.e2e.client import Session, open_session
from tests.e2e.stack import E2EStack, e2e_stack

__all__ = ["e2e_stack"]


@pytest.fixture
async def session(e2e_stack: E2EStack, request: pytest.FixtureRequest) -> AsyncIterator[Session]:
    directory = e2e_stack.directory / request.node.name
    async with open_session(e2e_stack, directory) as active:
        yield active
