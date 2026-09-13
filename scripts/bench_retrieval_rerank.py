"""Dedicated fixed-workload acceptance through the production post-search rerank stage."""

import argparse
import asyncio
import statistics
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field

from app.clients.model_runtime import ModelRuntimeClient
from app.core.deadline import Deadline
from app.retrieval.config import RetrievalConfig
from app.retrieval.reranking import rerank_candidates
from app.schemas.corpus import DocumentType
from app.schemas.ingestion import digest
from app.schemas.model_runtime import ModelMetadata
from app.schemas.retrieval import Candidate, RankingResult
from scripts.model_diagnostics_settings import ModelDiagnosticsSettings
from scripts.model_evidence import (
    CANDIDATE_COUNT,
    DEFAULT_BATCH,
    PAIR_COUNT,
    RERANK_LENGTH,
    RERANK_P50_SECONDS,
    Evidence,
    Provenance,
    Seconds,
)
from scripts.model_workload import input_hash, pairs


class StageMeasurement(Evidence):
    """Safe raw timing and results; excludes source text and credentials."""

    client_seconds: Seconds
    rerank_seconds: Seconds
    result: RankingResult


class RerankBenchmark(Evidence):
    """Separate client commit and deployed model provenance, bound to the fixed workload."""

    client_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    server: Provenance
    workload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: RetrievalConfig
    metadata: ModelMetadata
    measurements: list[StageMeasurement] = Field(max_length=50)

    @property
    def p50_s(self) -> float | None:
        """Full rerank/filter client wall time, including transport and queueing."""
        return (
            statistics.median(item.client_seconds for item in self.measurements)
            if self.measurements
            else None
        )

    @property
    def p95_s(self) -> float | None:
        """Nearest-rank percentile; incomplete measurements do not pass acceptance."""
        if len(self.measurements) != PAIR_COUNT:
            return None
        return sorted(item.client_seconds for item in self.measurements)[47]

    @property
    def accepted(self) -> bool:
        """Require fifty successful, identity-consistent 20-by-320 FP16 stage calls."""
        return (
            self.config == RetrievalConfig()
            and self.workload_sha256 == input_hash()
            and self.metadata.precision == "fp16"
            and self.metadata.rerank_max_length == RERANK_LENGTH
            and self.metadata.rerank_batch == DEFAULT_BATCH
            and len(self.measurements) == PAIR_COUNT
            and self.p50_s is not None
            and self.p50_s <= RERANK_P50_SECONDS
            and all(
                item.result.reranked
                and item.result.degradation is None
                and item.result.response is not None
                and item.result.response.metadata == self.metadata
                and len(item.result.response.scores) == CANDIDATE_COUNT
                and item.result.stages[0].input_count == CANDIDATE_COUNT
                for item in self.measurements
            )
        )


def workload() -> list[Candidate]:
    """Deterministic detached post-admission workload; this is not a retrieval quality set."""
    return [
        Candidate(
            chunk_uuid=uuid5(NAMESPACE_URL, f"rerank-benchmark/chunk/{index}"),
            document_id=uuid5(NAMESPACE_URL, f"rerank-benchmark/document/{index}"),
            document_version="a" * 64,
            chunking_version="b" * 64,
            content_sha256=digest(pair.passage),
            milvus_pk=index,
            content=pair.passage,
            parent_content="父章节。" + pair.passage,
            heading_path="固定业务记录",
            doc_type=DocumentType.POLICY,
            effective_from=None,
            effective_to=None,
            source_path=f"benchmark/{index}.md",
        )
        for index, pair in enumerate(pairs()[:CANDIDATE_COUNT])
    ]


async def benchmark(client_sha: str, server: Provenance) -> RerankBenchmark:
    """Use the existing authenticated tunnel without deployment or precision changes."""
    client = ModelRuntimeClient(ModelDiagnosticsSettings.load().model_runtime)
    measurements = []
    config = RetrievalConfig()
    try:
        metadata = (await client.ready(deadline=Deadline(time.monotonic() + 2))).metadata
        candidates = workload()
        for _ in range(PAIR_COUNT):
            started = time.monotonic()
            result = await rerank_candidates(
                pairs()[0].query, candidates, client, config, deadline=Deadline(started + 30)
            )
            elapsed = time.monotonic() - started
            # Candidates contain source text. Raw diagnostic artifacts retain only
            # per-stage counts/scores and model responses, never those candidates.
            safe = result.model_copy(deep=True)
            safe.candidates = []
            measurements.append(
                StageMeasurement(
                    client_seconds=elapsed,
                    rerank_seconds=result.stages[0].elapsed_ms / 1000,
                    result=safe,
                )
            )
            if not result.reranked:
                break
        return RerankBenchmark(
            client_sha=client_sha,
            server=server,
            workload_sha256=input_hash(),
            config=config,
            metadata=metadata,
            measurements=measurements,
        )
    finally:
        await client.aclose()


def main() -> None:
    """Write complete raw evidence and a concise summary, with failing gates reflected in exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-sha", required=True)
    parser.add_argument("--server-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = Provenance.model_validate_json(args.server_provenance.read_text())
    result = asyncio.run(benchmark(args.client_sha, provenance))
    args.output.write_text(result.model_dump_json(indent=2) + "\n")
    print(
        f"Rerank stage: accepted={result.accepted}; calls={len(result.measurements)}; "
        f"client p50={result.p50_s}s; p95={result.p95_s}s"
    )
    raise SystemExit(0 if result.accepted else 1)


if __name__ == "__main__":
    main()
