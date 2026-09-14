"""Real storage observations and formal judgment rebuild; deterministic HTTP models."""

from pathlib import Path

import pytest

from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from evals.harness.retrieval_contracts import Arm, Split, arm_config
from evals.harness.retrieval_dataset import check_manifest, load_dataset
from tests.consistency_support import ConsistencyHarness, consistency_harness
from tests.retrieval_support import RetrievalHarness, deadline, harness, query

pytestmark = [pytest.mark.integration, pytest.mark.storage]
__all__ = ["consistency_harness", "harness"]


@pytest.mark.parametrize("arm", [arm for arm in Arm if arm is not Arm.D_FP32])
async def test_observed_eight_arms_use_registered_production_path(
    harness: RetrievalHarness, arm: Arm
) -> None:
    pipeline = harness.pipeline(arm_config(arm))
    observed = await pipeline.retrieve_observed(query(), deadline=deadline())
    assert observed.trace.admitted
    assert {item.chunk_uuid for item in observed.trace.ranked} == {
        item.chunk_uuid for item in observed.trace.admitted
    }
    assert observed.result.degradation is None
    assert observed.result.retrieval_config == arm_config(arm)
    assert all(item.source_path for item in observed.trace.admitted)
    if arm_config(arm).use_rerank:
        response = observed.trace.rerank_response
        assert response is not None
        assert len(response.scores) == len(observed.trace.admitted)
        assert len(harness.embeddings.rerank_calls) == 1
    else:
        assert observed.trace.ranked == observed.result.candidates
    original = observed.trace.admitted[0].content
    observed.result.candidates.clear()
    assert observed.trace.admitted[0].content == original


async def test_formal_frozen_judgments_resolve_after_clean_rebuild(
    consistency_harness: ConsistencyHarness,
) -> None:
    harness = consistency_harness
    root = Path(__file__).resolve().parents[2] / "data/corpus"
    source = load_dataset(Split.FROZEN)
    first = await harness.service().ingest(root)
    assert first.successful
    before = (await harness.store.scan()).rows
    await harness.store._rpc(
        "test_drop_isolated_collection",
        lambda: harness.store._client.drop_collection(
            harness.store.settings.collection,
            timeout=None,
            retry_times=0,
            retry_on_rate_limit=False,
        ),
    )
    repaired = await harness.checker().check(root=root)
    assert repaired.successful
    after = (await harness.store.scan()).rows
    assert {item.chunk_uuid for item in before} == {item.chunk_uuid for item in after}
    assert {item.milvus_pk for item in before}.isdisjoint(item.milvus_pk for item in after)
    async with harness.database.session() as session:
        active = await DocumentRepository(session).manifest()
        registered = await ChunkRepository(session).list_all()
        admitted = await ChunkRepository(session).admit(after)
    check_manifest(source, active, {item.chunk_uuid for item in registered})
    assert set(source.manifest.chunk_ids) == {item.chunk_uuid for item in admitted}
    assert all(
        label.chunk_id in {item.chunk_uuid for item in after}
        for row in source.labels
        for label in row.judgments
    )
