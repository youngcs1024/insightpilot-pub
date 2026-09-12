"""Measure the production HTTP route with a fixed fifty-pair workload."""

import argparse
import asyncio
import time
from pathlib import Path

from app.clients.model_runtime import ModelRuntimeClient
from app.core.deadline import Deadline
from app.schemas.model_runtime import EmbedMode
from scripts.model_diagnostics_settings import ModelDiagnosticsSettings
from scripts.model_evidence import Benchmark, CallMeasurement, Provenance, RerankMeasurement
from scripts.model_workload import input_hash, pairs


async def benchmark(provenance: Provenance) -> Benchmark:
    """No downloads or host management; only the configured typed model client."""
    client = ModelRuntimeClient(ModelDiagnosticsSettings.load().model_runtime)
    try:
        identity = (await client.ready(deadline=Deadline(time.monotonic() + 2))).metadata
        workload = pairs()
        start = time.monotonic()
        embedded = await client.embed(
            [pair.passage for pair in workload[:16]],
            EmbedMode.DOCUMENT,
            deadline=Deadline(time.monotonic() + 20),
        )
        embedding = CallMeasurement.from_response(embedded, time.monotonic() - start)
        start = time.monotonic()
        relevant = await client.rerank(
            "七天无理由退货的条件是什么?",
            [
                "商品签收后七天内未使用且保留包装，可以申请无理由退货。",
                "明天天气晴朗，最高气温二十八摄氏度。",
            ],
            deadline=Deadline(time.monotonic() + 30),
        )
        relevance = RerankMeasurement.from_response(relevant, time.monotonic() - start)
        precision_calls = []
        for group in range(5):
            indices = list(range(group, 50, 5))
            start = time.monotonic()
            result = await client.rerank(
                workload[group].query,
                [workload[i].passage for i in indices],
                deadline=Deadline(time.monotonic() + 30),
            )
            precision_calls.append(
                RerankMeasurement.from_response(result, time.monotonic() - start)
            )
        latency_calls = []
        for _ in range(50):
            start = time.monotonic()
            result = await client.rerank(
                workload[0].query,
                [pair.passage for pair in workload[:20]],
                deadline=Deadline(start + 30),
            )
            latency_calls.append(
                RerankMeasurement.from_response(result, time.monotonic() - start)
            )
        return Benchmark(
            provenance=provenance,
            input_sha256=input_hash(),
            metadata=identity,
            embedding=embedding,
            relevance=relevance,
            precision_calls=precision_calls,
            latency_calls=latency_calls,
        )
    finally:
        await client.aclose()


def main() -> None:
    """Retain measured evidence even if an acceptance threshold fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--provenance", type=Path)
    source.add_argument("--provenance-json")
    args = parser.parse_args()
    provenance = Provenance.model_validate_json(
        args.provenance.read_text() if args.provenance else args.provenance_json
    )
    result = asyncio.run(benchmark(provenance))
    output = result.model_dump_json(indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output)
    raise SystemExit(0 if result.accepted else 1)


if __name__ == "__main__":
    main()
