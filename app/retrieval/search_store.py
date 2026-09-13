"""Bounded asynchronous Milvus hybrid search and optional per-arm diagnostics."""

import asyncio

from pymilvus import RRFRanker  # type: ignore[import-untyped]

from app.core.deadline import Deadline
from app.core.errors import RetrievalUnavailableError
from app.core.observability import TraceMetadata, observe
from app.retrieval.config import RetrievalConfig
from app.retrieval.fusion import (
    OUTPUT_FIELDS,
    SearchArm,
    arm_request,
    attach_scores,
    candidates,
    enabled_arms,
    time_filter,
)
from app.retrieval.milvus_repo import MilvusRepository
from app.schemas.retrieval import Candidate, EncodedQuery, KnowledgeTimeScope


class HybridSearchStore(MilvusRepository):
    """One outer search budget covers fusion and every diagnostic request."""

    async def hybrid_search(
        self,
        query: EncodedQuery,
        config: RetrievalConfig,
        scope: KnowledgeTimeScope,
        *,
        deadline: Deadline,
        timeout_s: float,
    ) -> list[Candidate]:
        """Keep existing RPC error typing and explicitly disabled vendor retries."""
        try:
            async with asyncio.timeout(deadline.budget(timeout_s)):
                if not self._validated:
                    await self.ensure_collection()
                return await self._search(query, config, time_filter(scope))
        except TimeoutError as exc:
            raise RetrievalUnavailableError(operation="hybrid_search") from exc

    async def _single(
        self,
        arm: SearchArm,
        query: EncodedQuery,
        config: RetrievalConfig,
        expression: str,
    ) -> list[Candidate]:
        request = arm_request(arm, query, config, expression)
        with observe("retrieval_arm", TraceMetadata(tool=arm.value)):
            raw = await self._rpc(
                "search_" + arm.value,
                lambda: self._client.search(
                    self.settings.collection,
                    data=request.data,
                    anns_field=request.anns_field,
                    search_params=request.param,
                    filter=expression,
                    limit=config.pool,
                    output_fields=OUTPUT_FIELDS,
                    consistency_level="Strong",
                    timeout=None,
                    retry_times=0,
                    retry_on_rate_limit=False,
                ),
            )
        return candidates(raw, arm)

    async def _search(
        self,
        query: EncodedQuery,
        config: RetrievalConfig,
        expression: str,
    ) -> list[Candidate]:
        arms = enabled_arms(config)
        if len(arms) == 1:
            return await self._single(arms[0], query, config, expression)
        requests = [arm_request(arm, query, config, expression) for arm in arms]
        with observe("retrieval_fusion", TraceMetadata(tool="rrf")):
            raw = await self._rpc(
                "hybrid_search",
                lambda: self._client.hybrid_search(
                    self.settings.collection,
                    reqs=requests,
                    ranker=RRFRanker(k=config.rrf_k),
                    limit=config.pool,
                    output_fields=OUTPUT_FIELDS,
                    consistency_level="Strong",
                    timeout=None,
                    retry_times=0,
                    retry_on_rate_limit=False,
                ),
            )
        result = candidates(raw, None)
        if config.record_arm_scores:
            for arm in arms:
                diagnostic = await self._single(arm, query, config, expression)
                attach_scores(result, diagnostic, arm)
        return result
