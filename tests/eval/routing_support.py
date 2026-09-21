"""Shared offline routing evidence constructors; no service initialization."""

from datetime import UTC, datetime

from app.core.llm_config import ModelRoleSettings
from app.core.routing import RoutingStrategy
from evals.harness.routing_contracts import Attempt, Measurements, Observation, Snapshot, Split
from evals.harness.routing_dataset import load_cases
from tests.router_support import decision


def snapshot() -> Snapshot:
    return Snapshot(
        git_sha="a" * 40,
        source_dirty=False,
        dataset_hash="dataset",
        source_hash="source",
        prompt_hashes={"router.md": "prompt"},
        capability_hashes=["capabilities"],
        model=ModelRoleSettings(model="offline-model", timeout_s=45),
        min_confidence=0.6,
        timeout_s=90,
        production_strategy=RoutingStrategy.HYBRID,
    )


def measurements(split: Split = Split.DEVELOPMENT) -> Measurements:
    cases = [c for c in load_cases() if c.split is split]
    return Measurements(
        run_id="offline-routing",
        created_at=datetime.now(UTC),
        split=split,
        repeats=3,
        config=snapshot(),
        case_ids=[c.id for c in cases],
        attempts=[
            Attempt(
                case_id=c.id,
                arm=arm,
                repeat=n,
                observation=Observation(
                    decision=decision(c.expected)
                    if arm is not RoutingStrategy.PREFILTER_ONLY
                    else None,
                    tokens=20 if arm is not RoutingStrategy.PREFILTER_ONLY else 0,
                    latency_ms=1,
                ),
            )
            for c in cases
            for arm in RoutingStrategy
            for n in range(1, 4)
        ],
    )
