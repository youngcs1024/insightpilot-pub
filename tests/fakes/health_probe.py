"""Controllable health probe shared by API lifecycle tests."""

import asyncio

from app.core.errors import HealthProbeError


class FakeProbe:
    """Record lifecycle and allow exact control over outage, delay, and cancellation."""

    def __init__(self) -> None:
        self.failure = False
        self.hang = False
        self.close_hang = False
        self.close_failure = False
        self.calls = 0
        self.closed = False
        self.cancelled = False
        self.entered = asyncio.Event()
        self.peer: FakeProbe | None = None

    async def check(self) -> None:
        self.calls += 1
        self.entered.set()
        if self.peer:
            await self.peer.entered.wait()
        if self.failure:
            raise HealthProbeError("secret upstream diagnostic must not escape")
        try:
            if self.hang:
                await asyncio.Event().wait()
        finally:
            self.cancelled = self.hang

    async def aclose(self) -> None:
        self.closed = True
        if self.close_failure:
            raise HealthProbeError("secret cleanup diagnostic")
        if self.close_hang:
            await asyncio.Event().wait()
