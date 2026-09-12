"""Synthetic storage/inference overlap, deliberately separate from business ingestion."""

import asyncio
import hashlib
import importlib
import time
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]  # Third-party DB adapter.
from pydantic import BaseModel, Field, ValidationError

from spikes.capacity.client import ProbeClient
from spikes.capacity.contracts import (
    EmbedResponse,
    FailureKind,
    Pair,
    PairBatch,
    ProbeError,
    ReadyResponse,
    Stage,
    Status,
    TextBatch,
    percentile,
)
from spikes.capacity.corpus import corpus_hash, fixed_pairs, text_at
from spikes.capacity.settings import WorkloadFields, WorkloadSettings

MIN_SEARCH_ROWS = 32


class WorkloadResult(BaseModel):
    """Scope and timings stay explicit even when only some stages finish."""

    status: Status = Status.PENDING
    failure: FailureKind | None = None
    readiness: ReadyResponse | None = None
    corpus_size: int
    corpus_sha256: str
    fixed_pairs_sha256: str
    inserted_vectors: int = 0
    ddl_rows: int = 0
    retrieval_seconds: list[float] = Field(default_factory=list)
    embed_seconds: list[float] = Field(default_factory=list)
    rerank_seconds: list[float] = Field(default_factory=list)
    comparison_scores: list[float] = Field(default_factory=list)
    server_stages: list[Stage] = Field(default_factory=list)
    retrieval_p50_s: float | None = None
    retrieval_p95_s: float | None = None
    rerank_p50_s: float | None = None
    rerank_p95_s: float | None = None
    elapsed_s: float = 0
    cgroup_peak_bytes: int | None = None


def classify_failure(error: BaseException) -> FailureKind:
    """Unwrap structured task failures without inspecting any exception prose."""
    if isinstance(error, BaseExceptionGroup):
        return classify_failure(error.exceptions[0])
    if isinstance(error, ProbeError):
        return error.kind
    if isinstance(error, TimeoutError):
        return FailureKind.TIMEOUT
    if isinstance(error, ValidationError):
        return FailureKind.OUTPUT
    return FailureKind.PREREQUISITE


class VectorStore:
    """Synchronous third-party Milvus calls run in a worker thread, with RPC timeouts."""

    def __init__(self, uri: str) -> None:
        library = importlib.import_module("pymilvus")
        self.client = library.MilvusClient(uri=uri, timeout=10)
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", library.DataType.INT64, is_primary=True)
        schema.add_field("dense", library.DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse", library.DataType.SPARSE_FLOAT_VECTOR)
        indexes = self.client.prepare_index_params()
        indexes.add_index(
            field_name="dense",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 200},
        )
        indexes.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )
        # Fresh collection only: rerunning against a populated project is an operator error.
        if self.client.has_collection("capacity_chunks", timeout=10):
            raise ProbeError(
                FailureKind.PREREQUISITE, "Use a fresh isolated project for every run."
            )
        self.client.create_collection(
            "capacity_chunks",
            schema=schema,
            index_params=indexes,
            consistency_level="Strong",
            timeout=60,
        )

    def insert(self, offset: int, encoded_json: str) -> int:
        """Third-party dictionaries exist only inside this adapter."""
        encoded = EmbedResponse.model_validate_json(encoded_json)
        result = self.client.insert(
            "capacity_chunks",
            data=[
                {"id": offset + i, "dense": dense, "sparse": sparse}
                for i, (dense, sparse) in enumerate(zip(encoded.dense, encoded.sparse, strict=True))
            ],
            timeout=30,
        )
        return int(result["insert_count"])

    def search(self, vector: list[float]) -> list[int]:
        """Search real indexed vectors, returning scalar IDs across the adapter boundary."""
        results = self.client.search(
            "capacity_chunks",
            data=[vector],
            anns_field="dense",
            limit=20,
            search_params={"metric_type": "IP", "params": {"ef": 64}},
            timeout=20,
        )
        return [int(hit["id"]) for hit in results[0]]

    def flush(self) -> None:
        """Include persistent storage flush/index work in the capacity observation window."""
        self.client.flush("capacity_chunks", timeout=60)

    def close(self) -> None:
        """Close the RPC transport without dropping collections or their data."""
        self.client.close()


async def database_load(config: WorkloadFields, result: WorkloadResult) -> None:
    """Disposable DDL/bulk-write workload; no business schema or existing volume is touched."""
    connection = await asyncpg.connect(
        host=config.postgres_host,
        user="postgres",
        database="postgres",
        password=config.postgres_password.get_secret_value(),
        timeout=10,
        command_timeout=20,
    )
    try:
        await connection.execute(
            "CREATE TABLE capacity_rows(id bigint PRIMARY KEY, payload text NOT NULL)"
        )
        for offset in range(0, 50000, 1000):
            await connection.executemany(
                "INSERT INTO capacity_rows VALUES ($1, $2)",
                [(i, text_at(i)) for i in range(offset, offset + 1000)],
                timeout=20,
            )
            result.ddl_rows += 1000
            await asyncio.sleep(0.2)
        await connection.execute(
            "CREATE INDEX capacity_payload_idx ON capacity_rows ((left(payload, 24)))"
        )
        await connection.execute("ANALYZE capacity_rows")
    finally:
        await connection.close(timeout=5)


async def vector_load(
    config: WorkloadFields, client: ProbeClient, store: VectorStore, result: WorkloadResult
) -> None:
    """Encode distinct deterministic batches and persist both dense and sparse vectors."""
    for offset in range(0, config.corpus_size, 16):
        encoded = await client.embed(
            TextBatch(
                texts=[text_at(i) for i in range(offset, min(offset + 16, config.corpus_size))]
            )
        )
        result.server_stages.append(encoded.stage)
        result.inserted_vectors += await asyncio.to_thread(
            store.insert, offset, encoded.model_dump_json()
        )
    await asyncio.to_thread(store.flush)


async def turns(
    config: WorkloadFields, client: ProbeClient, store: VectorStore, result: WorkloadResult
) -> None:
    """Query/real search/rerank while ingestion is active, then finish a fixed turn count."""
    while result.inserted_vectors < MIN_SEARCH_ROWS:
        await asyncio.sleep(0.1)
    for i in range(config.turns):
        start = time.monotonic()
        encoded = await client.embed(TextBatch(texts=[text_at(i)]))
        ids = await asyncio.to_thread(store.search, encoded.dense[0])
        if len(ids) != 20:  # noqa: PLR2004 -- documented candidate count.
            raise ProbeError(
                FailureKind.OUTPUT, "Search must supply twenty real stored candidates."
            )
        reranked = await client.rerank(
            PairBatch(pairs=[Pair(query="退款政策是什么?", passage=text_at(key)) for key in ids])
        )
        result.server_stages.extend([encoded.stage, reranked.stage])
        result.retrieval_seconds.append(time.monotonic() - start)


async def run(config: WorkloadFields) -> WorkloadResult:
    """All concurrent jobs are strongly retained and cancelled together on failure."""
    result = WorkloadResult(
        corpus_size=config.corpus_size,
        corpus_sha256=corpus_hash(config.corpus_size),
        fixed_pairs_sha256=hashlib.sha256(
            "\n".join(p.model_dump_json() for p in fixed_pairs()).encode()
        ).hexdigest(),
    )
    client = ProbeClient(config.model_url, config.auth_token.get_secret_value())
    store: VectorStore | None = None
    start = time.monotonic()
    try:
        result.readiness = await client.ready()
        for offset in range(0, 50, 20):
            scored = await client.rerank(PairBatch(pairs=fixed_pairs()[offset : offset + 20]))
            result.comparison_scores.extend(scored.scores)
        store = await asyncio.to_thread(VectorStore, config.milvus_uri)
        async with asyncio.timeout(1200), asyncio.TaskGroup() as group:
            group.create_task(database_load(config, result))
            group.create_task(vector_load(config, client, store, result))
            group.create_task(turns(config, client, store, result))
        result.status = Status.PASSED
    except ProbeError as exc:
        result.status, result.failure = Status.FAILED, exc.kind
    except Exception as exc:
        # CLI boundary: preserve partial evidence, never send raw adapter errors over HTTP.
        result.status, result.failure = Status.FAILED, classify_failure(exc)
    finally:
        result.elapsed_s = time.monotonic() - start
        result.cgroup_peak_bytes = await asyncio.to_thread(read_job_peak)
        result.embed_seconds, result.rerank_seconds = client.embed_seconds, client.rerank_seconds
        if result.retrieval_seconds:
            result.retrieval_p50_s = percentile(result.retrieval_seconds, 0.5)
            result.retrieval_p95_s = percentile(result.retrieval_seconds, 0.95)
        if result.rerank_seconds:
            result.rerank_p50_s = percentile(result.rerank_seconds, 0.5)
            result.rerank_p95_s = percentile(result.rerank_seconds, 0.95)
        if store:
            await asyncio.to_thread(store.close)
        await client.http.aclose()
        config.output.write_text(result.model_dump_json(indent=2) + "\n")
    return result


def read_job_peak() -> int | None:
    """A one-shot job records its kernel peak before the container's cgroup disappears."""
    path = Path("/sys/fs/cgroup/memory.peak")
    return int(path.read_text()) if path.is_file() else None


def main() -> int:
    """One fresh run directory and disposable database/collection per invocation."""
    config = WorkloadSettings.load().workload
    if config.output.exists():
        raise ProbeError(FailureKind.PREREQUISITE, "Evidence output already exists.")
    result = asyncio.run(run(config))
    print(result.status.value)
    return 0 if result.status is Status.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
