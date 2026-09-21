"""Deterministic presentation and commit validation of synthesis results."""

from app.agents.answer_rendering import finalize_answer
from app.agents.contracts import Answer, EvidenceBundle, SynthesisResult
from app.agents.synthesis_generation import missing_sources, synthesis_input
from app.agents.synthesis_validation import citations_for, validate_output
from app.core.errors import ConflictError
from app.schemas.knowledge import KnowledgePassage
from app.schemas.memory import FormatPreferenceContent
from app.schemas.synthesis import ClaimKind, SynthesisOutput


def validate_result(result: SynthesisResult, bundle: EvidenceBundle) -> None:
    """Recheck all durable content before committing or rendering an answer."""
    if result.evidence_refs != bundle.refs or result.missing_components != missing_sources(bundle):
        raise ConflictError("synthesis references differ from snapshots")
    if result.abstention is not None:
        if result.claims or result.conflicts or result.summary:
            raise ConflictError("abstention retains rejected synthesis")
        return
    draft = SynthesisOutput.model_validate(
        result.model_dump(include=set(SynthesisOutput.model_fields))
    )
    checked = validate_output(draft, synthesis_input("validate committed answer", bundle))
    if checked != draft or not any(
        claim.kind is not ClaimKind.UNSUPPORTED and (claim.data_refs or claim.chunk_ids)
        for claim in draft.claims
    ):
        raise ConflictError("synthesis was not validated")


def synthesis_answer(
    result: SynthesisResult,
    bundle: EvidenceBundle,
    degraded_components: list[str],
    preference: FormatPreferenceContent | None = None,
    *,
    trace_id: str,
) -> Answer:
    """Render validated synthesis through the same final envelope as every route."""
    validate_result(result, bundle)
    degraded = list(dict.fromkeys([*degraded_components, *result.missing_components]))
    answer = Answer(
        markdown="pending",
        confidence=0,
        trace_id=trace_id,
        format_preference=preference,
        claims=list(result.claims),
        attempted_sources=["data", "knowledge"],
        unanswered=list(result.unanswered),
        sql=bundle.data.data.sql if bundle.data else "",
        assumptions=synthesis_input("answer assumptions", bundle).assumptions,
        evidence_refs=bundle.refs,
        citations=citations_for(result, bundle),
        knowledge_passages=[
            KnowledgePassage(text=claim.text, chunk_ids=tuple(claim.chunk_ids))
            for claim in result.claims
            if claim.chunk_ids
        ],
        degraded_components=degraded,
        abstained=result.abstention is not None,
        synthesis=result,
    )
    return finalize_answer(answer, bundle)


def validate_synthesis_answer(answer: Answer, bundle: EvidenceBundle) -> None:
    """The committed prose, confidence and presentation must all be reproducible."""
    if answer.synthesis is None or answer.trace_id is None:
        raise ConflictError("missing synthesis provenance")
    expected = synthesis_answer(
        answer.synthesis,
        bundle,
        answer.degraded_components,
        answer.format_preference,
        trace_id=answer.trace_id,
    )
    if answer != expected:
        raise ConflictError("answer differs from validated synthesis")
