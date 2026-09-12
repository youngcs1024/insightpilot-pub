"""Deterministic synchronous model substitute, with observable admission behavior."""

import threading

from app.schemas.model_runtime import (
    EMBED_REVISION,
    RERANK_REVISION,
    EmbedRequest,
    ModelMetadata,
    RerankRequest,
)
from model_runtime.embedder import Embeddings
from model_runtime.reranker import Scores


class FakeModels:
    """No GPU imports, network access or timers needed to control inference."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls: list[str] = []
        self.failure: Exception | None = None

    def load(self) -> None:
        self.calls.append("load")
        if self.failure:
            raise self.failure

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            embed_revision=EMBED_REVISION,
            rerank_revision=RERANK_REVISION,
            precision="fp16",
            embed_batch=16,
            rerank_batch=16,
            embed_max_length=512,
            rerank_max_length=320,
        )

    def embed(self, request: EmbedRequest, at: float) -> tuple[Embeddings, ModelMetadata]:
        self.calls.append("embed")
        self.entered.set()
        if not self.release.wait(3):
            raise AssertionError("Test did not release model work")
        return Embeddings(
            dense=[[1.0] + [0.0] * 1023 for _ in request.texts],
            sparse=[{42: 0.5} for _ in request.texts],
        ), self.metadata()

    def rerank(self, request: RerankRequest, at: float) -> tuple[Scores, ModelMetadata]:
        self.calls.append("rerank")
        return Scores(
            scores=[1 / (index + 1) for index in range(len(request.passages))]
        ), self.metadata()
