"""Temporary HTTP GPU probe. Production runtime/queue/recovery remains Phase 3."""

import hmac
import importlib
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import cast

import uvicorn
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from model_runtime.config import ModelServerSettings
from spikes.capacity.cgroup import memory_peak
from spikes.capacity.contracts import (
    EmbedResponse,
    ErrorResponse,
    FailureKind,
    ModelIdentity,
    PairBatch,
    ProbeError,
    ReadyResponse,
    RerankResponse,
    Stage,
    TextBatch,
)
from spikes.capacity.corpus import fixed_pairs, text_at
from spikes.capacity.settings import ProbeRuntimeSettings
from spikes.model_runtime.cuda_smoke import valid_snapshot


def cgroup_peak() -> int:
    """The kernel tracks short memory spikes that periodic Docker sampling can miss."""
    return memory_peak()


def guard_forward(model: object, torch_module: ModuleType, batches: list[int]) -> None:
    """Surface GPU failures before FlagEmbedding's implicit batch-shrinking retry loop."""
    original = cast("Callable[..., object]", getattr(model, "forward", None))

    def forward(*args: object, **kwargs: object) -> object:
        inputs = args[0] if args else kwargs
        ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
        shape = getattr(ids, "shape", ())
        if shape:
            batches.append(int(shape[0]))
        try:
            return original(*args, **kwargs)
        except torch_module.cuda.OutOfMemoryError as exc:
            raise ProbeError(
                FailureKind.OOM, "CUDA probe exhausted memory at the requested batch size."
            ) from exc
        except RuntimeError as exc:
            raise ProbeError(
                FailureKind.CUDA, "CUDA probe forward failed without a batch-size retry."
            ) from exc

    model.forward = forward  # type: ignore[attr-defined]  # Confined torch adapter.


class Models:
    """Confine the untyped upstream ML adapter to an opt-in GPU process."""

    def __init__(self, settings: ModelServerSettings, headroom_bytes: int = 4 * 1024**3) -> None:
        self.headroom_bytes = headroom_bytes
        self.forward_batches: list[int] = []
        self.identity = ModelIdentity(
            embed_revision=settings.embed_revision,
            rerank_revision=settings.rerank_revision,
            precision=settings.precision,
        )
        self.lock = threading.Lock()
        self.torch = importlib.import_module("torch")
        if not self.torch.cuda.is_available() or self.torch.cuda.device_count() != 1:
            raise ProbeError(FailureKind.CUDA, "Exactly one authorized CUDA device is required.")
        self.check_headroom()
        embed = Path("/models/models--BAAI--bge-m3/snapshots") / settings.embed_revision
        rerank = (
            Path("/models/models--BAAI--bge-reranker-v2-m3/snapshots") / settings.rerank_revision
        )
        if not all(valid_snapshot(path) for path in (embed, rerank)):
            raise ProbeError(FailureKind.PREREQUISITE, "Pinned offline snapshots are missing.")
        package = importlib.import_module("FlagEmbedding")
        start = self.start_stage()
        self.embedder = package.BGEM3FlagModel(
            str(embed), devices=["cuda:0"], use_fp16=settings.precision == "fp16"
        )
        self.reranker = package.FlagReranker(
            str(rerank), devices=["cuda:0"], use_fp16=settings.precision == "fp16"
        )
        dtype = self.torch.float16 if settings.precision == "fp16" else self.torch.float32
        for model in (self.embedder.model, self.reranker.model):
            # FlagEmbedding constructors keep CPU weights until the first call.
            # Place both models explicitly before measuring warmup with both resident.
            try:
                model.to(device="cuda:0", dtype=dtype)
            except self.torch.cuda.OutOfMemoryError as exc:
                raise ProbeError(
                    FailureKind.OOM, "Both model weights do not fit CUDA memory."
                ) from exc
            except RuntimeError as exc:
                raise ProbeError(FailureKind.CUDA, "CUDA model placement failed.") from exc
            model.eval()
            guard_forward(model, self.torch, self.forward_batches)
            parameter = next(model.parameters())
            if parameter.device.type != "cuda" or parameter.dtype != dtype:
                raise ProbeError(FailureKind.CUDA, "Actual model precision/device mismatch.")
        self.stages = [self.finish_stage("load", start)]
        self.stages.append(self.embed(TextBatch(texts=[text_at(i) for i in range(16)])).stage)
        self.stages.append(self.rerank(PairBatch(pairs=fixed_pairs()[:20])).stage)

    def check_headroom(self) -> None:
        """Do not continue allocating once the agreed shared headroom is exhausted."""
        free, _ = self.torch.cuda.mem_get_info()
        if free < self.headroom_bytes:
            raise ProbeError(FailureKind.HEADROOM, "Shared GPU is below the configured reserve.")

    def start_stage(self) -> float:
        """Synchronize before timing CUDA work and reset per-stage high water marks."""
        self.check_headroom()
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        self.forward_batches.clear()
        return time.monotonic()

    def finish_stage(self, name: str, start: float) -> Stage:
        """Wall time includes all completed CUDA work, not just kernel submission."""
        self.torch.cuda.synchronize()
        self.check_headroom()
        return Stage(
            name=name,
            seconds=time.monotonic() - start,
            allocated_peak_bytes=self.torch.cuda.max_memory_allocated(),
            reserved_peak_bytes=self.torch.cuda.max_memory_reserved(),
            cgroup_peak_bytes=cgroup_peak(),
            forward_batch_sizes=list(self.forward_batches),
        )

    @contextmanager
    def operation(self) -> Iterator[None]:
        """One active inference; retain its slot until CUDA completes even on disconnect."""
        if not self.lock.acquire(timeout=1):
            raise ProbeError(FailureKind.TIMEOUT, "The single probe inference slot is occupied.")
        try:
            yield
        except self.torch.cuda.OutOfMemoryError as exc:
            raise ProbeError(FailureKind.OOM, "CUDA capacity probe exhausted memory.") from exc
        finally:
            self.lock.release()

    def embed(self, batch: TextBatch) -> EmbedResponse:
        """Execute the exact 16-by-512 dense+sparse feasibility workload."""
        with self.operation():
            start = self.start_stage()
            result = self.embedder.encode(
                batch.texts,
                batch_size=16,
                max_length=512,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            response = EmbedResponse(
                identity=self.identity,
                dense=result["dense_vecs"],
                sparse=result["lexical_weights"],
                stage=self.finish_stage("embed", start),
            )
            if len(response.dense) != len(batch.texts) or len(response.sparse) != len(batch.texts):
                raise ProbeError(FailureKind.OUTPUT, "Embedding count mismatch.")
            return response

    def rerank(self, batch: PairBatch) -> RerankResponse:
        """Return normalized scores in input order with a microbatch of sixteen."""
        with self.operation():
            start = self.start_stage()
            scores = self.reranker.compute_score(
                [[p.query, p.passage] for p in batch.pairs],
                batch_size=16,
                max_length=320,
                normalize=True,
            )
            response = RerankResponse(
                identity=self.identity,
                scores=scores if isinstance(scores, list) else [scores],
                stage=self.finish_stage("rerank", start),
            )
            if len(response.scores) != len(batch.pairs):
                raise ProbeError(FailureKind.OUTPUT, "Rerank count mismatch.")
            return response


def create_app(models: Models, token: str) -> FastAPI:
    """Only this disposable probe's four routes; no production runtime is introduced."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def authenticate(authorization: str = Header(default="")) -> None:
        if not hmac.compare_digest(authorization, f"Bearer {token}"):
            raise ProbeError(FailureKind.AUTHENTICATION, "Probe authentication failed.")

    @app.exception_handler(ProbeError)
    async def probe_error(_request: Request, error: ProbeError) -> JSONResponse:
        return JSONResponse(
            status_code=401 if error.kind is FailureKind.AUTHENTICATION else 503,
            content=ErrorResponse(kind=error.kind).model_dump(mode="json"),
        )

    @app.exception_handler(ValidationError)
    async def output_error(_request: Request, _error: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=503, content=ErrorResponse(kind=FailureKind.OUTPUT).model_dump(mode="json")
        )

    @app.get("/health")
    def health() -> bool:
        return True

    @app.get("/ready", dependencies=[Depends(authenticate)])
    def ready() -> ReadyResponse:
        return ReadyResponse(
            identity=models.identity,
            stages=models.stages,
            device=str(models.torch.cuda.get_device_name(0)),
        )

    @app.post("/v1/embed", dependencies=[Depends(authenticate)])
    def embed(batch: TextBatch) -> EmbedResponse:
        return models.embed(batch)

    @app.post("/v1/rerank", dependencies=[Depends(authenticate)])
    def rerank(batch: PairBatch) -> RerankResponse:
        return models.rerank(batch)

    return app


def main() -> None:
    """Warm up before binding the port; missing CUDA/cache never produces readiness."""
    settings = ProbeRuntimeSettings.load()
    config = settings.model_server
    try:
        models = Models(config, settings.probe.gpu_headroom_bytes)
    except ProbeError as exc:
        print(ErrorResponse(kind=exc.kind).model_dump_json(), flush=True)
        raise SystemExit(1) from exc
    uvicorn.run(
        create_app(models, config.auth_token.get_secret_value()),
        host="0.0.0.0",  # noqa: S104 -- published on server loopback only.
        port=8100,
        access_log=False,
    )


if __name__ == "__main__":
    main()
