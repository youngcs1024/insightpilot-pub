"""Offline CUDA adapter; imports ML packages only inside authorized startup."""

import importlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import structlog

from app.schemas.model_runtime import EmbedRequest, ModelMetadata, RerankRequest
from model_runtime.config import ModelServerSettings
from model_runtime.embedder import Embeddings, lexical_weights
from model_runtime.errors import ModelContractError, ModelDeadlineError, ModelError, ModelOOMError
from model_runtime.reranker import Scores, sigmoid

logger = structlog.get_logger(__name__)


class ModelBackend(Protocol):
    """Synchronous boundary always invoked by the single inference executor."""

    def load(self) -> None: ...

    def metadata(self) -> ModelMetadata: ...

    def embed(self, request: EmbedRequest, at: float) -> tuple[Embeddings, ModelMetadata]: ...

    def rerank(self, request: RerankRequest, at: float) -> tuple[Scores, ModelMetadata]: ...


class CudaModels:
    """Both resident models share one GPU, precision and bounded recovery policy."""

    def __init__(self, settings: ModelServerSettings) -> None:
        self.settings = settings

    def metadata(self) -> ModelMetadata:
        """Report configured settings; operations override recovered microbatch sizes."""
        return ModelMetadata.model_validate(
            self.settings.model_dump(
                exclude={"auth_token", "device", "workers", "max_concurrency", "queue_capacity"}
            )
        )

    def load(self) -> None:
        """Fail closed on absent CUDA/cache; weights are never fetched by this service."""
        self.torch = importlib.import_module("torch")
        if not self.torch.cuda.is_available() or self.torch.cuda.device_count() != 1:
            raise ModelError("Exactly one visible CUDA device is required")
        paths = [
            Path("/models/models--BAAI--bge-m3/snapshots") / self.settings.embed_revision,
            Path("/models/models--BAAI--bge-reranker-v2-m3/snapshots")
            / self.settings.rerank_revision,
        ]
        if not all((path / "config.json").is_file() for path in paths):
            raise ModelContractError("Pinned offline model snapshots are missing")
        package = importlib.import_module("FlagEmbedding")
        self.encoder = package.BGEM3FlagModel(
            str(paths[0]), devices=["cuda:0"], use_fp16=self.settings.precision == "fp16"
        )
        self.ranker = package.FlagReranker(
            str(paths[1]), devices=["cuda:0"], use_fp16=self.settings.precision == "fp16"
        )
        dtype = self.torch.float16 if self.settings.precision == "fp16" else self.torch.float32
        for model in (self.encoder.model, self.ranker.model):
            model.to(device="cuda:0", dtype=dtype)
            model.eval()
            parameter = next(model.parameters())
            if parameter.device.type != "cuda" or parameter.dtype != dtype:
                raise ModelContractError("Actual CUDA precision differs from settings")

    def _recover[T](self, call: Callable[[int], T], batch: int, at: float) -> tuple[T, int]:
        self.torch.cuda.reset_peak_memory_stats()
        for attempt in range(2):
            if time.monotonic() >= at:
                raise ModelDeadlineError()
            try:
                with self.torch.inference_mode():
                    result = call(batch)
                self.torch.cuda.synchronize()
            except self.torch.cuda.OutOfMemoryError:
                # Leave the exception scope before clearing cache, so traceback-held
                # tensors from a failed forward do not survive into the next attempt.
                logger.exception(
                    "model_cuda_oom", attempt=attempt, microbatch=batch, exc_info=False
                )
            else:
                logger.info(
                    "model_inference_completed",
                    microbatch=batch,
                    vram_peak_bytes=self.torch.cuda.max_memory_allocated(),
                )
                return result, batch
            self.torch.cuda.empty_cache()
            if attempt == 1 or batch == 1:
                raise ModelOOMError()
            batch = max(1, batch // 2)
        raise ModelOOMError()

    def embed(self, request: EmbedRequest, at: float) -> tuple[Embeddings, ModelMetadata]:
        """Direct forward avoids FlagEmbedding's unbounded automatic batch recovery."""
        result, batch = self._recover(
            lambda size: self._embed(request, size, at), self.settings.embed_batch, at
        )
        return result, self.metadata().model_copy(update={"embed_batch": batch})

    def _embed(self, request: EmbedRequest, batch: int, at: float) -> Embeddings:
        dense: list[list[float]] = []
        sparse: list[dict[int, float]] = []
        excluded = {
            int(value)
            for name in ("cls_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
            if (value := getattr(self.encoder.tokenizer, name, None)) is not None
        }
        for start in range(0, len(request.texts), batch):
            if time.monotonic() >= at:
                raise ModelDeadlineError()
            inputs = self.encoder.tokenizer(
                request.texts[start : start + batch],
                padding=True,
                truncation=True,
                max_length=self.settings.embed_max_length,
                return_tensors="pt",
            ).to("cuda:0")
            output = self.encoder.model(
                inputs, return_dense=True, return_sparse=True, return_colbert_vecs=False
            )
            dense.extend(output["dense_vecs"].float().cpu().tolist())
            weights = output["sparse_vecs"].squeeze(-1).float().cpu().tolist()
            ids = inputs["input_ids"].cpu().tolist()
            sparse.extend(
                lexical_weights(row, tokens, excluded)
                for row, tokens in zip(weights, ids, strict=True)
            )
        return Embeddings(dense=dense, sparse=sparse)

    def rerank(self, request: RerankRequest, at: float) -> tuple[Scores, ModelMetadata]:
        """Apply the same explicit truncation and input order in both precisions."""
        length = min(request.max_length, self.settings.rerank_max_length)
        result, batch = self._recover(
            lambda size: self._rerank(request, size, length, at), self.settings.rerank_batch, at
        )
        return result, self.metadata().model_copy(
            update={"rerank_batch": batch, "rerank_max_length": length}
        )

    def _rerank(self, request: RerankRequest, batch: int, length: int, at: float) -> Scores:
        scores: list[float] = []
        for start in range(0, len(request.passages), batch):
            if time.monotonic() >= at:
                raise ModelDeadlineError()
            passages = request.passages[start : start + batch]
            inputs = self.ranker.tokenizer(
                [request.query] * len(passages),
                passages,
                padding=True,
                truncation=True,
                max_length=length,
                return_tensors="pt",
            ).to("cuda:0")
            logits = self.ranker.model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(sigmoid(float(value)) for value in logits.cpu().tolist())
        return Scores(scores=scores)
