"""Bounded synthesis generation from committed views, with one reference repair."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import Field

from app.agents.contracts import EvidenceBundle, SynthesisInput, SynthesisResult
from app.agents.prompts import SYNTHESIS, SYNTHESIS_REPAIR
from app.agents.synthesis_validation import data_view, validate_output
from app.core.errors import FabricatedCitation, SynthesisValidationError
from app.core.llm_config import ModelRole
from app.schemas.mcp import Contract
from app.schemas.synthesis import ClaimKind, SynthesisAbstention, SynthesisOutput

if TYPE_CHECKING:
    from app.agents.runtime import RuntimeContext

logger = structlog.get_logger(__name__)


class SynthesisMetadata(Contract):
    """The only non-evidence inputs allowed into the synthesis model call."""

    schema_version: Literal[1] = 1
    question: str = Field(max_length=32_000)
    sql_scope: str = Field(max_length=32_000)
    assumptions: list[str]
    valid_chunk_ids: list[str]
    missing_components: list[Literal["data", "knowledge"]]


def missing_sources(bundle: EvidenceBundle) -> list[Literal["data", "knowledge"]]:
    """An empty query is valid data; a knowledge snapshot without chunks is not support."""
    missing: list[Literal["data", "knowledge"]] = []
    if bundle.data is None:
        missing.append("data")
    if bundle.knowledge is None or not bundle.knowledge.knowledge.chunks:
        missing.append("knowledge")
    return missing


def synthesis_input(question: str, bundle: EvidenceBundle) -> SynthesisInput:
    """Only immutable snapshot assumptions cross this projection."""
    return SynthesisInput(
        question=question,
        data=bundle.data.data if bundle.data else None,
        knowledge=bundle.knowledge.knowledge if bundle.knowledge else None,
        assumptions=list(
            dict.fromkeys(
                [
                    *(bundle.data.data.assumptions if bundle.data else []),
                    *(bundle.knowledge.knowledge.assumptions if bundle.knowledge else []),
                ]
            )
        ),
    )


def _messages(
    source: SynthesisInput, missing: list[Literal["data", "knowledge"]], *, repair: bool
) -> list[BaseMessage]:
    metadata = SynthesisMetadata(
        question=source.question,
        sql_scope=source.data.sql if source.data else "",
        assumptions=source.assumptions,
        valid_chunk_ids=[str(chunk.chunk_id) for chunk in source.knowledge.chunks]
        if source.knowledge
        else [],
        missing_components=missing,
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=SYNTHESIS + ("\n" + SYNTHESIS_REPAIR if repair else "")),
        HumanMessage(content=metadata.model_dump_json()),
    ]
    # Preserve the frozen generation blocks, never serialize the full audit payload.
    if source.data is not None:
        messages.append(HumanMessage(content=source.data.generation_block))
    if source.knowledge is not None and source.knowledge.chunks:
        messages.extend(
            [
                HumanMessage(content=source.knowledge.time_scope.model_dump_json()),
                HumanMessage(content=source.knowledge.generation_block),
            ]
        )
    return messages


def _abstention(
    bundle: EvidenceBundle, reason: SynthesisAbstention, attempts: int
) -> SynthesisResult:
    return SynthesisResult(
        evidence_refs=bundle.refs,
        missing_components=missing_sources(bundle),
        abstention=reason,
        attempts=attempts,
        unanswered=["请补充可核验的查询结果或适用政策证据。"],
    )


async def generate_synthesis(
    source: SynthesisInput, bundle: EvidenceBundle, ctx: RuntimeContext
) -> SynthesisResult:
    """Upstream errors propagate; only invalid generated references get one repair."""
    missing = missing_sources(bundle)
    data_view(source)
    if len(missing) == 2:  # noqa: PLR2004 -- the two fixed specialists.
        return _abstention(bundle, SynthesisAbstention.NO_EVIDENCE, 0)
    for attempt in (1, 2):
        ctx.deadline.check("synthesize")
        draft = await ctx.llm.generate_structured(
            ModelRole.SYNTHESIS,
            _messages(source, missing, repair=attempt > 1),
            SynthesisOutput,
            deadline=ctx.deadline,
        )
        try:
            validated = validate_output(draft, source)
        except (FabricatedCitation, SynthesisValidationError) as exc:
            logger.exception(
                "synthesis_reference_rejected", attempt=attempt, code=exc.code, exc_info=False
            )
            continue
        downgraded = sum(
            original.kind != checked.kind
            for original, checked in zip(draft.claims, validated.claims, strict=True)
        )
        if downgraded:
            logger.info("synthesis_causal_downgraded", count=downgraded)
        if not any(
            claim.kind is not ClaimKind.UNSUPPORTED and (claim.data_refs or claim.chunk_ids)
            for claim in validated.claims
        ):
            return _abstention(bundle, SynthesisAbstention.UNSUPPORTED, attempt)
        logger.info(
            "synthesis_validated",
            claims=len(validated.claims),
            conflicts=len(validated.conflicts),
            attempt=attempt,
            missing=len(missing),
        )
        return SynthesisResult(
            **validated.model_dump(),
            evidence_refs=bundle.refs,
            missing_components=missing,
            attempts=attempt,
        )
    return _abstention(bundle, SynthesisAbstention.INVALID_REFERENCES, 2)
