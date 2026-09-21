"""Revalidate the v3 envelope against the committed snapshots before publication."""

from app.agents.answer_rendering import finalize_answer
from app.agents.contracts import Answer, EvidenceBundle
from app.agents.synthesis_answer import validate_synthesis_answer
from app.agents.synthesis_generation import synthesis_input
from app.agents.synthesis_validation import CAUSAL_MARKERS, citations_for, validate_output
from app.core.errors import ConflictError
from app.schemas.metric_resolution import MetricClarification
from app.schemas.synthesis import Claim, ClaimKind, SynthesisOutput

MISSING_EVIDENCE = "请补充可核验的查询结果、适用政策或更具体的时间与问题范围。"
INFERENCE_EVIDENCE = "需要能够核实上述推断的归因数据与适用政策依据。"


def validate_formatted_answer(
    answer: Answer, bundle: EvidenceBundle, clarification: MetricClarification | None = None
) -> None:
    """Historical answers are read unchanged; new commits must satisfy v3 invariants."""
    if answer.schema_version != 3 or answer.trace_id is None:
        raise ConflictError("new answers require current provenance")
    if answer.synthesis is not None:
        validate_synthesis_answer(answer, bundle)
        return
    if answer.abstained:
        if answer.claims or answer.citations or answer.knowledge_passages:
            raise ConflictError("abstention contains claims")
        expected_unanswered = [clarification.message if clarification else MISSING_EVIDENCE]
    else:
        if clarification or not answer.claims:
            raise ConflictError("answer contains no validated claims")
        expected_unanswered = [INFERENCE_EVIDENCE] if any(
            claim.kind is ClaimKind.INFERENCE for claim in answer.claims
        ) else []
    if answer.unanswered != expected_unanswered:
        raise ConflictError("answer missing-information text differs")
    if bundle.data:
        checked = validate_output(
            SynthesisOutput(claims=answer.claims), synthesis_input("validate answer", bundle)
        )
        if checked.claims != answer.claims or answer.knowledge_passages:
            raise ConflictError("data answer claims differ")
    elif bundle.knowledge:
        score = bundle.knowledge.knowledge.top_rerank_score
        expected = [Claim(
            text=passage.text,
            kind=ClaimKind.INFERENCE if CAUSAL_MARKERS.search(passage.text)
            else ClaimKind.FACT_DOCUMENT,
            chunk_ids=list(passage.chunk_ids), confidence=score if score is not None else 0.5,
        ) for passage in answer.knowledge_passages]
        if answer.claims != expected:
            raise ConflictError("knowledge claims differ from passages")
    elif not answer.abstained:
        raise ConflictError("answer without evidence")
    if answer.citations != citations_for(SynthesisOutput(claims=answer.claims), bundle):
        raise ConflictError("answer citations differ from snapshots")
    if finalize_answer(answer, bundle) != answer:
        raise ConflictError("answer presentation differs from validated fields")
