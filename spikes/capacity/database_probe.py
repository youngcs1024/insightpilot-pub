"""Measure the isolated database workload even when GPU admission is unavailable."""

import asyncio

from spikes.capacity.contracts import FailureKind, ProbeError, Status
from spikes.capacity.corpus import corpus_hash
from spikes.capacity.settings import WorkloadSettings
from spikes.capacity.workload import WorkloadResult, database_load, read_job_peak


async def main_async() -> int:
    """Retain explicitly partial evidence; this never creates workload.json."""
    config = WorkloadSettings.load().workload
    output = config.output.with_name("database-workload.json")
    if output.exists():
        raise ProbeError(FailureKind.PREREQUISITE, "Choose a fresh database evidence output.")
    result = WorkloadResult(
        corpus_size=50000, corpus_sha256=corpus_hash(50000), fixed_pairs_sha256=""
    )
    await database_load(config, result)
    result.cgroup_peak_bytes = await asyncio.to_thread(read_job_peak)
    # Full workload remains pending: these are only the actual database row counters.
    result.status = Status.PENDING
    output.write_text(result.model_dump_json(indent=2) + "\n")
    print("database_probe_complete rows=50000 full_workload=pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
