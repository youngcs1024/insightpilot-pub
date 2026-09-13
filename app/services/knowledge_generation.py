"""One citation correction at the shared prompt boundary; no chat or storage integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.prompts import KNOWLEDGE_CITATION_REPAIR, KNOWLEDGE_SYSTEM
from app.core.errors import FabricatedCitation, KnowledgeEvidenceError
from app.core.llm_config import ModelRole
from app.retrieval.evidence import render_documents
from app.schemas.knowledge import (
    Citation,
    KnowledgeAbstention,
    KnowledgeDraft,
    KnowledgeGeneration,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from app.agents.runtime import LlmPort
    from app.core.deadline import Deadline
    from app.schemas.knowledge import KnowledgeEvidence

logger = structlog.get_logger(__name__)


def validate_citations(draft: KnowledgeDraft, evidence: KnowledgeEvidence) -> tuple[Citation, ...]:
    """Never resolve against the full candidate pool or today's registry."""
    allowed = {chunk.chunk_id: chunk for chunk in evidence.chunks}
    identifiers = dict.fromkeys(identifier for passage in draft.passages for identifier in passage.chunk_ids)
    if any(identifier not in allowed for identifier in identifiers):
        raise FabricatedCitation()
    return tuple(
        Citation.model_validate(allowed[identifier].model_dump(include=set(Citation.model_fields)))
        for identifier in identifiers
    )


def _messages(evidence: KnowledgeEvidence, *, repair: bool) -> list[BaseMessage]:
    # Keep the frozen document block as its own message: JSON-escaping it again would
    # change the evidence token accounting. Query metadata is a separate context slot.
    system = KNOWLEDGE_SYSTEM + ("\n" + KNOWLEDGE_CITATION_REPAIR if repair else "")
    return [
        SystemMessage(content=system),
        HumanMessage(content=json.dumps({
            "question": evidence.query_used,
            "time_scope": evidence.time_scope.model_dump(mode="json"),
            "assumptions": evidence.assumptions,
            "reranked": evidence.reranked,
            "degradation": evidence.degradation,
            "valid_chunk_ids": [str(chunk.chunk_id) for chunk in evidence.chunks],
        }, ensure_ascii=False)),
        HumanMessage(content=evidence.generation_block),
    ]


class KnowledgeGenerationService:
    """Owners inject the existing structured LLM boundary and per-request deadline."""

    def __init__(self, llm: LlmPort) -> None:
        self.llm = llm

    async def generate(
        self, evidence: KnowledgeEvidence, *, deadline: Deadline
    ) -> KnowledgeGeneration:
        """Return only validated passages; rejected drafts are never returned or logged."""
        deadline.check("knowledge_generation")
        if evidence.generation_block != render_documents(evidence.chunks):
            raise KnowledgeEvidenceError(reason="generation_block_mismatch")
        if not evidence.chunks:
            reason = (
                KnowledgeAbstention.BUDGET_EXHAUSTED if evidence.decisions
                else KnowledgeAbstention.NO_EVIDENCE
            )
            return KnowledgeGeneration(abstention=reason, attempts=0)
        for attempt in (1, 2):
            deadline.check("knowledge_generation")
            draft = await self.llm.generate_structured(
                ModelRole.SYNTHESIS, _messages(evidence, repair=attempt == 2),
                KnowledgeDraft, deadline=deadline,
            )
            if not draft.passages:
                return KnowledgeGeneration(abstention=KnowledgeAbstention.UNSUPPORTED, attempts=attempt)
            try:
                citations = validate_citations(draft, evidence)
            except FabricatedCitation:
                logger.exception("knowledge_citation_rejected", attempt=attempt)
                continue
            return KnowledgeGeneration(passages=draft.passages, citations=citations, attempts=attempt)
        return KnowledgeGeneration(abstention=KnowledgeAbstention.FABRICATED_CITATION, attempts=2)
