"""Select whole context blocks and freeze exactly what generation may cite."""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

import structlog
from pydantic import ValidationError

from app.core.errors import KnowledgeEvidenceError
from app.schemas.ingestion import ChunkIdentity, digest
from app.schemas.knowledge import (
    EvidenceChunk,
    EvidenceDecision,
    FrozenScores,
    InjectionFlag,
    KnowledgeEvidence,
    TextSelection,
)

if TYPE_CHECKING:
    from app.agents.runtime import SchemaTokenPort
    from app.retrieval.config import EvidenceConfig
    from app.schemas.retrieval import Candidate, CandidateProvenance, RetrievalResult

logger = structlog.get_logger(__name__)
_PATTERNS = (
    (InjectionFlag.IGNORE_CHINESE, re.compile(r"忽略\s*以上")),
    (InjectionFlag.IGNORE_PREVIOUS, re.compile(r"ignore\s+previous", re.IGNORECASE)),
    (InjectionFlag.SYSTEM_ROLE, re.compile(r"system\s*:", re.IGNORECASE)),
    (InjectionFlag.ROLE_OVERRIDE, re.compile(r"you\s+are\s+now", re.IGNORECASE)),
)


def injection_flags(value: str) -> tuple[InjectionFlag, ...]:
    """Inspect the original text without editing or suppressing a source."""
    return tuple(flag for flag, pattern in _PATTERNS if pattern.search(value))


def render_documents(chunks: tuple[EvidenceChunk, ...]) -> str:
    """Escape both text and attributes so document content cannot close our delimiters."""
    blocks = []
    for chunk in chunks:
        attributes = {
            "id": str(chunk.chunk_id),
            "source": chunk.source_path,
            "title": chunk.document_title,
            "section": chunk.heading_path,
            "page": str(chunk.page) if chunk.page is not None else "",
            "effective_from": chunk.effective_from.isoformat() if chunk.effective_from else "",
            "effective_to": chunk.effective_to.isoformat() if chunk.effective_to else "",
        }
        rendered = " ".join(
            f'{key}="{html.escape(value, quote=True)}"' for key, value in attributes.items()
        )
        blocks.append(
            f"<retrieved_document {rendered}>\n"
            f"{html.escape(chunk.generation_text, quote=False)}\n</retrieved_document>"
        )
    return "\n".join(blocks)


def _identity(item: ChunkIdentity) -> tuple[object, ...]:
    return tuple(getattr(item, name) for name in ChunkIdentity.model_fields)


def _registered(
    result: RetrievalResult,
) -> dict[tuple[object, ...], CandidateProvenance]:
    registered = {_identity(item): item for item in result.provenance}
    if len(registered) != len(result.provenance):
        raise KnowledgeEvidenceError(reason="duplicate_provenance")
    seen = set()
    for candidate in result.candidates:
        provenance = registered.get(_identity(candidate))
        if provenance is None or candidate.chunk_uuid in seen:
            raise KnowledgeEvidenceError(reason="missing_or_duplicate_provenance")
        seen.add(candidate.chunk_uuid)
        fields = ("source_path", "heading_path", "doc_type", "effective_from", "effective_to")
        if any(getattr(candidate, key) != getattr(provenance, key) for key in fields):
            raise KnowledgeEvidenceError(reason="provenance_mismatch")
        if digest(candidate.content) != candidate.content_sha256:
            raise KnowledgeEvidenceError(reason="content_mismatch")
    return registered


def _chunk(
    candidate: Candidate,
    provenance: CandidateProvenance,
    selection: TextSelection,
    flags: tuple[InjectionFlag, ...],
) -> EvidenceChunk:
    text = candidate.parent_content if selection is TextSelection.PARENT else candidate.content
    fields = provenance.model_dump(exclude={"chunk_uuid", "milvus_pk", "doc_type"})
    return EvidenceChunk.model_validate(
        {
            **fields,
            "chunk_id": candidate.chunk_uuid,
            "original_text": text,
            "generation_text": text,
            "text_selection": selection,
            "scores": FrozenScores.model_validate(candidate.scores.model_dump()),
            "injection_flags": flags,
        }
    )


def _select(
    result: RetrievalResult, config: EvidenceConfig, counter: SchemaTokenPort
) -> tuple[tuple[EvidenceChunk, ...], tuple[EvidenceDecision, ...]]:
    registered = _registered(result)
    chunks: tuple[EvidenceChunk, ...] = ()
    decisions = []
    for candidate in result.candidates:
        provenance = registered[_identity(candidate)]
        flags = injection_flags(
            "\n".join(
                (
                    candidate.content,
                    candidate.parent_content,
                    provenance.document_title,
                    provenance.source_path,
                    provenance.heading_path,
                )
            )
        )
        if flags:
            logger.warning(
                "knowledge_instruction_like_content",
                chunk_id=str(candidate.chunk_uuid),
                document_id=str(candidate.document_id),
                flags=[flag.value for flag in flags],
            )
        selection = TextSelection.OMITTED_BUDGET
        for choice in (TextSelection.PARENT, TextSelection.CHILD):
            proposed = (*chunks, _chunk(candidate, provenance, choice, flags))
            if counter.count(render_documents(proposed)) <= config.max_tokens:
                chunks = proposed
                selection = choice
                break
        decisions.append(
            EvidenceDecision(
                chunk_id=candidate.chunk_uuid,
                selection=selection,
                injection_flags=flags,
            )
        )
    return chunks, tuple(decisions)


def package_evidence(
    result: RetrievalResult, config: EvidenceConfig, counter: SchemaTokenPort
) -> KnowledgeEvidence:
    """Package an admitted result without querying any mutable source or invoking a model."""
    try:
        chunks, decisions = _select(result, config, counter)
        block = render_documents(chunks)
        fields = result.model_dump(
            exclude={
                "schema_version",
                "query",
                "candidates",
                "provenance",
                "stages",
            }
        )
        evidence = KnowledgeEvidence.model_validate(
            {
                **fields,
                "query_used": result.query.standalone,
                "original_question": result.query.original_question,
                "time_scope": result.query.time_scope.model_dump(),
                "assumptions": tuple(result.query.assumptions),
                "chunks": chunks,
                "packaging_config": config.model_dump(),
                "decisions": decisions,
                "generation_block": block,
                "generation_tokens": counter.count(block),
                "tokenizer": counter.name,
            }
        )
    except ValidationError as exc:
        raise KnowledgeEvidenceError() from exc
    logger.info(
        "knowledge_evidence_packaged",
        candidates=len(result.candidates),
        selected=len(chunks),
        omitted=len(result.candidates) - len(chunks),
        tokens=evidence.generation_tokens,
    )
    return evidence
