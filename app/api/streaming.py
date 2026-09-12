"""Close the producer even when the socket fails between generator iterations."""

from collections.abc import AsyncGenerator

import anyio
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send


class ChatStreamingResponse(StreamingResponse):
    """Listen for disconnects on every ASGI version and always close the generator."""

    def __init__(self, content: AsyncGenerator[str, None], headers: dict[str, str]) -> None:
        super().__init__(content, media_type="text/event-stream", headers=headers)
        self.generator = content

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(self._send, send, group.cancel_scope)
                await self.listen_for_disconnect(receive)
                group.cancel_scope.cancel()
        finally:
            with anyio.CancelScope(shield=True):
                await self.generator.aclose()

    async def _send(self, send: Send, cancel_scope: anyio.CancelScope) -> None:
        try:
            await self.stream_response(send)
        except OSError:
            # Socket delivery failure is the same cancellation path as http.disconnect.
            pass
        finally:
            cancel_scope.cancel()
