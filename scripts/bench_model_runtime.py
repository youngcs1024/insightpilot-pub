"""Measure the production HTTP route with a fixed fifty-pair workload."""

import argparse
import asyncio
import statistics
import time
from pathlib import Path

from pydantic import BaseModel, Field

from app.clients.model_runtime import ModelRuntimeClient
from app.core.deadline import Deadline
from app.schemas.model_runtime import EmbedMode, ModelMetadata, Score
from scripts.model_diagnostics_settings import ModelDiagnosticsSettings
from scripts.model_workload import input_hash, pairs

RERANK_P50_SECONDS = 2.0


class Benchmark(BaseModel):
    """Client wall times and server elapsed times remain separately attributable."""

    schema_version: int = 1
    input_sha256: str
    metadata: ModelMetadata
    scores: list[Score] = Field(min_length=50, max_length=50)
    client_seconds: list[float] = Field(min_length=50, max_length=50)
    server_ms: list[int] = Field(min_length=50, max_length=50)
    queue_ms: list[int] = Field(min_length=50, max_length=50)
    inference_ms: list[int] = Field(min_length=50, max_length=50)
    p50_s: float
    p95_s: float
    relevant_above_control: bool
    accepted: bool


async def benchmark() -> Benchmark:
    """No downloads or host management; only the configured typed model client."""
    client = ModelRuntimeClient(ModelDiagnosticsSettings.load().model_runtime)
    try:
        identity = (await client.ready(deadline=Deadline(time.monotonic() + 2))).metadata
        workload = pairs()
        await client.embed(
            [pair.passage for pair in workload[:16]],
            EmbedMode.DOCUMENT,
            deadline=Deadline(time.monotonic() + 20),
        )
        relevant = await client.rerank(
            "七天无理由退货的条件是什么?",
            [
                "商品签收后七天内未使用且保留包装，可以申请无理由退货。",
                "明天天气晴朗，最高气温二十八摄氏度。",
            ],
            deadline=Deadline(time.monotonic() + 30),
        )
        scores = [0.0] * 50
        for group in range(5):
            indices = list(range(group, 50, 5))
            result = await client.rerank(
                workload[group].query,
                [workload[i].passage for i in indices],
                deadline=Deadline(time.monotonic() + 30),
            )
            for index, score in zip(indices, result.scores, strict=True):
                scores[index] = score
        elapsed, server, queue, inference = [], [], [], []
        for _ in range(50):
            start = time.monotonic()
            result = await client.rerank(
                workload[0].query,
                [pair.passage for pair in workload[:20]],
                deadline=Deadline(start + 30),
            )
            elapsed.append(time.monotonic() - start)
            server.append(result.ms)
            queue.append(result.queue_ms)
            inference.append(result.inference_ms)
        p50 = statistics.median(elapsed)
        ordered = sorted(elapsed)
        correct = relevant.scores[0] > relevant.scores[1]
        return Benchmark(
            input_sha256=input_hash(),
            metadata=identity,
            scores=scores,
            client_seconds=elapsed,
            server_ms=server,
            queue_ms=queue,
            inference_ms=inference,
            p50_s=p50,
            p95_s=ordered[47],
            relevant_above_control=correct,
            accepted=correct and (identity.precision == "fp32" or p50 <= RERANK_P50_SECONDS),
        )
    finally:
        await client.aclose()


def main() -> None:
    """Retain measured evidence even if an acceptance threshold fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(benchmark())
    output = result.model_dump_json(indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output)
    raise SystemExit(0 if result.accepted else 1)


if __name__ == "__main__":
    main()
