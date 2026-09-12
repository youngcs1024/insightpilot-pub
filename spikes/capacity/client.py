"""Bounded HTTP client for the disposable probe; no automatic inference retries."""

import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from spikes.capacity.contracts import (
    EmbedResponse,
    ErrorResponse,
    FailureKind,
    ModelIdentity,
    PairBatch,
    ProbeError,
    ReadyResponse,
    RerankResponse,
    TextBatch,
)

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ProbeClient:
    """Retain transport timing separately from the server's synchronized computation."""

    def __init__(self, url: str, token: str) -> None:
        self.http = httpx.AsyncClient(
            base_url=url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(30, connect=5),
            trust_env=False,
        )
        self.embed_seconds: list[float] = []
        self.rerank_seconds: list[float] = []
        self.identity: ModelIdentity | None = None

    async def request(
        self, path: str, result_type: type[ResponseT], body: BaseModel | None = None
    ) -> ResponseT:
        """A failed non-idempotent workload is not replayed; HTTP error types are explicit."""
        try:
            response = await self.http.request(
                "GET" if body is None else "POST",
                path,
                content=None if body is None else body.model_dump_json(),
                headers={} if body is None else {"Content-Type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise ProbeError(
                FailureKind.TIMEOUT, "Model request exceeded its client deadline."
            ) from exc
        except httpx.TransportError as exc:
            raise ProbeError(FailureKind.TRANSPORT, "Model transport failed.") from exc
        if response.is_error:
            try:
                error = ErrorResponse.model_validate_json(response.content)
            except ValidationError as exc:
                raise ProbeError(
                    FailureKind.OUTPUT, "Unrecognized model failure response."
                ) from exc
            raise ProbeError(error.kind, "Model probe returned a typed failure.")
        try:
            return result_type.model_validate_json(response.content)
        except ValidationError as exc:
            raise ProbeError(FailureKind.OUTPUT, "Malformed model output.") from exc

    async def ready(self) -> ReadyResponse:
        """Check authenticated readiness and record actual model identity."""
        response = await self.request("/ready", ReadyResponse)
        self.identity = response.identity
        return response

    async def embed(self, batch: TextBatch) -> EmbedResponse:
        """Measure full-path wall time, including the SSH transport."""
        start = time.monotonic()
        response = await self.request("/v1/embed", EmbedResponse, batch)
        self.check_identity(response.identity)
        self.embed_seconds.append(time.monotonic() - start)
        if len(response.dense) != len(batch.texts) or len(response.sparse) != len(batch.texts):
            raise ProbeError(FailureKind.OUTPUT, "Embedding count mismatch.")
        return response

    async def rerank(self, batch: PairBatch) -> RerankResponse:
        """Measure full-path reranking for one bounded request."""
        start = time.monotonic()
        response = await self.request("/v1/rerank", RerankResponse, batch)
        self.check_identity(response.identity)
        self.rerank_seconds.append(time.monotonic() - start)
        if len(response.scores) != len(batch.pairs):
            raise ProbeError(FailureKind.OUTPUT, "Rerank count mismatch.")
        return response

    def check_identity(self, identity: ModelIdentity) -> None:
        """An endpoint switched during measurement must not mix model settings silently."""
        if self.identity is not None and self.identity != identity:
            raise ProbeError(FailureKind.OUTPUT, "Model identity changed during the run.")
