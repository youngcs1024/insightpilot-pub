"""Run the same fixed inputs sequentially against FP16 and FP32 probe instances."""

import asyncio
import hashlib

from pydantic import BaseModel, Field

from spikes.capacity.client import ProbeClient
from spikes.capacity.contracts import (
    FailureKind,
    PairBatch,
    ProbeError,
    ReadyResponse,
    RerankResponse,
    Status,
)
from spikes.capacity.corpus import fixed_pairs
from spikes.capacity.settings import WorkloadSettings


class BenchmarkResult(BaseModel):
    """Keep complete score vectors and client/server latency observations."""

    status: Status = Status.PENDING
    failure: FailureKind | None = None
    readiness: ReadyResponse | None = None
    input_sha256: str
    scores: list[float] = Field(default_factory=list)
    timings_s: list[float] = Field(default_factory=list)
    responses: list[RerankResponse] = Field(default_factory=list)


async def benchmark(client: ProbeClient) -> BenchmarkResult:
    """No FP32 process is started until the FP16 process has been stopped by its operator."""
    pairs = fixed_pairs()
    result = BenchmarkResult(
        input_sha256=hashlib.sha256(
            "\n".join(p.model_dump_json() for p in pairs).encode()
        ).hexdigest()
    )
    try:
        result.readiness = await client.ready()
        for offset in range(0, len(pairs), 20):
            response = await client.rerank(PairBatch(pairs=pairs[offset : offset + 20]))
            result.scores.extend(response.scores)
            result.responses.append(response)
        # Percentiles must not mix the final ten-pair comparison call with twenty-pair calls.
        for _ in range(50):
            result.responses.append(await client.rerank(PairBatch(pairs=pairs[:20])))
        result.timings_s = client.rerank_seconds[-50:]
        result.status = Status.PASSED
    except ProbeError as exc:
        result.status, result.failure = Status.FAILED, exc.kind
    return result


async def main_async() -> int:
    """Reuse the client-only process settings, never a server SSH credential."""
    config = WorkloadSettings.load().workload
    if config.output.exists():
        raise ProbeError(FailureKind.PREREQUISITE, "Choose a fresh benchmark output.")
    client = ProbeClient(config.model_url, config.auth_token.get_secret_value())
    try:
        result = await benchmark(client)
        config.output.write_text(result.model_dump_json(indent=2) + "\n")
        return 0 if result.status is Status.PASSED else 1
    finally:
        await client.http.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
