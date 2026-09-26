"""All routes publish one envelope derived only from committed evidence."""

from typing import Literal

from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.answer_generation import data_claims
from app.agents.answer_rendering import finalize_answer
from app.agents.answer_validation import INFERENCE_EVIDENCE, MISSING_EVIDENCE
from app.agents.contracts import Answer, EvidenceBundle, EvidenceRefs, Route
from app.agents.nodes.common import failed
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.agents.synthesis_answer import synthesis_answer
from app.agents.synthesis_generation import synthesis_input
from app.agents.synthesis_validation import CAUSAL_MARKERS, citations_for
from app.core.errors import ConflictError, InsightPilotError
from app.schemas.knowledge import KnowledgeDraft, KnowledgePassage
from app.schemas.memory import FormatPreferenceContent
from app.schemas.metric_resolution import ClarificationKind, MetricClarification
from app.schemas.synthesis import Claim, ClaimKind, SynthesisOutput
from app.services.knowledge_generation import validate_citations


def format_preference(state: AgentState) -> FormatPreferenceContent | None:
    """Read the finalized presentation projection without modifying shared state."""
    if state.context is None or state.context.format_preference is None:
        return None
    return state.context.format_preference.model_copy(deep=True)


def attempted_sources(route: Route | None) -> list[Literal["data", "knowledge"]]:
    """Expected analytical sources, independent of whether they returned evidence."""
    if route is Route.BOTH:
        return ["data", "knowledge"]
    if route is Route.KNOWLEDGE_ONLY:
        return ["knowledge"]
    return [] if route is Route.CLARIFY or route is None else ["data"]


def _no_evidence(state: AgentState, ctx: RuntimeContext) -> Command[str]:
    clarification = state.route_clarification or state.data_clarification
    if clarification is None and state.knowledge_clarification is not None:
        clarification = MetricClarification(
            kind=ClarificationKind(state.knowledge_clarification.kind.value),
            message=state.knowledge_clarification.message,
        )
    if state.failures:
        return Command(update={"status": "failed"}, goto=END)
    answer = Answer(
        markdown="pending",
        confidence=0,
        sql="",
        assumptions=[],
        trace_id=ctx.trace_id,
        format_preference=format_preference(state),
        evidence_refs=EvidenceRefs(),
        degraded_components=list(dict.fromkeys(state.degraded_components)),
        abstained=True,
        attempted_sources=[]
        if clarification
        else attempted_sources(state.route.route if state.route else None),
        unanswered=[clarification.message if clarification else MISSING_EVIDENCE],
    )
    answer = finalize_answer(answer, EvidenceBundle())
    return Command(
        update={
            "answer": answer,
            "clarification": clarification,
            "abstained": True,
            "status": "abstained",
        },
        goto=END,
    )


async def _single_answer(state: AgentState, ctx: RuntimeContext, bundle: EvidenceBundle) -> Answer:
    claims: list[Claim] = []
    passages: list[KnowledgePassage] = []
    preference = format_preference(state)
    if bundle.data:
        claims = await data_claims(state.question, bundle, preference, ctx)
    elif bundle.knowledge:
        if ctx.knowledge_generation is None:
            raise ConflictError("missing knowledge generation service")
        generated = await ctx.knowledge_generation.generate(
            bundle.knowledge.knowledge,
            deadline=ctx.deadline,
            format_preference=preference,
            presentation_request=state.question,
        )
        if generated.abstention is None:
            validate_citations(
                KnowledgeDraft(passages=generated.passages), bundle.knowledge.knowledge
            )
            passages = list(generated.passages)
            score = bundle.knowledge.knowledge.top_rerank_score
            claims = [
                Claim(
                    text=passage.text,
                    kind=ClaimKind.INFERENCE
                    if CAUSAL_MARKERS.search(passage.text)
                    else ClaimKind.FACT_DOCUMENT,
                    chunk_ids=list(passage.chunk_ids),
                    confidence=score if score is not None else 0.5,
                )
                for passage in passages
            ]
    unanswered = [] if claims else [MISSING_EVIDENCE]
    if any(claim.kind is ClaimKind.INFERENCE for claim in claims):
        unanswered.append(INFERENCE_EVIDENCE)
    answer = Answer(
        markdown="pending",
        confidence=0,
        trace_id=ctx.trace_id,
        format_preference=preference,
        claims=claims,
        attempted_sources=attempted_sources(state.route.route if state.route else Route.DATA_ONLY),
        unanswered=unanswered,
        sql=bundle.data.data.sql if bundle.data else "",
        assumptions=synthesis_input("answer assumptions", bundle).assumptions,
        evidence_refs=bundle.refs,
        citations=citations_for(SynthesisOutput(claims=claims), bundle),
        knowledge_passages=passages,
        degraded_components=list(dict.fromkeys(state.degraded_components)),
        abstained=not claims,
    )
    return finalize_answer(answer, bundle)


async def format_answer(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Own the final answer for analysis, clarification and honest abstention."""
    ctx = runtime.context
    try:
        ctx.deadline.check("format_answer")
        refs = state.evidence_refs
        if refs is None or (refs.data_snapshot_id is None and refs.knowledge_snapshot_id is None):
            return _no_evidence(state, ctx)
        bundle = await ctx.evidence.read_bundle(ctx.identity)
        if bundle.refs != refs:
            raise ConflictError("committed evidence missing")
        if state.route is not None and state.route.route is Route.BOTH:
            if state.synthesis is None:
                raise ConflictError("BOTH answer requires validated synthesis")
            answer = synthesis_answer(
                state.synthesis,
                bundle,
                state.degraded_components,
                format_preference(state),
                trace_id=ctx.trace_id,
            )
        else:
            answer = await _single_answer(state, ctx, bundle)
        status = (
            "abstained"
            if answer.abstained
            else ("degraded" if answer.degraded_components else "succeeded")
        )
        return Command(
            update={
                "answer": answer,
                "status": status,
                "abstained": answer.abstained,
            },
            goto=END,
        )
    except InsightPilotError as exc:
        return failed("format_answer", state, exc)
