"""Shared committed-evidence fixtures and explicit model scripts for synthesis."""

from dataclasses import replace
from datetime import date
from uuid import UUID

from langgraph.runtime import Runtime

from app.agents.contracts import EvidenceBundle, Route, RouteDecision, SynthesisResult
from app.agents.knowledge.nodes.package_evidence import package_evidence
from app.agents.nodes.synthesis import synthesize
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.agents.summarize import package_result
from app.retrieval.evidence import render_documents
from app.schemas.knowledge import FrozenPeriod, FrozenRangeScope
from app.schemas.synthesis import CellReference, Claim, ClaimKind, SynthesisOutput
from tests.agents.knowledge_support import ranked
from tests.agents.support import context, result
from tests.fakes.chat_model import FakeChatModel


def synthesis_draft(*, data: bool = True, knowledge_id: UUID | None = None) -> SynthesisOutput:
    claims = []
    if data:
        claims.append(
            Claim(
                text="GMV 为 42。",
                kind=ClaimKind.FACT_DATA,
                confidence=0.95,
                data_refs=[CellReference(row=0, column=0, value=42)],
            )
        )
    if knowledge_id:
        claims.append(
            Claim(
                text="政策说明退款规则。",
                kind=ClaimKind.FACT_DOCUMENT,
                chunk_ids=[knowledge_id],
                confidence=0.95,
            )
        )
    return SynthesisOutput(claims=claims, unanswered=["需要支付方式与实际使用记录以验证归因。"])


async def synthesis_context(
    *, data: bool = True, knowledge: bool = True
) -> tuple[RuntimeContext, AgentState, EvidenceBundle]:
    ctx = context()
    bundle = await ctx.evidence.commit_bundle(
        ctx.identity,
        package_result(result(), []) if data else None,
        package_evidence(ranked(), ctx) if knowledge else None,
    )
    identifier = bundle.knowledge.knowledge.chunks[0].chunk_id if bundle.knowledge else None
    ctx = replace(ctx, llm=FakeChatModel([synthesis_draft(data=data, knowledge_id=identifier)]))
    state = AgentState(
        **ctx.identity.model_dump(),
        question="为什么退款率上升?",
        evidence_refs=bundle.refs,
        route=RouteDecision(
            route=Route.BOTH, confidence=1, data_intent="计算退款", knowledge_intent="退款政策"
        ),
    )
    return ctx, state, bundle


async def synthesized(ctx: RuntimeContext, state: AgentState) -> SynthesisResult:
    command = await synthesize(state, Runtime(context=ctx))
    assert command.goto == "format_answer"
    return command.update["synthesis"]


def knowledge_text(ctx: RuntimeContext, text: str) -> EvidenceBundle:
    """Replace an explicit fixture snapshot, preserving its ID and exact rendered block."""
    saved = ctx.evidence.knowledge
    chunk = saved.knowledge.chunks[0].model_copy(
        update={
            "original_text": text,
            "generation_text": text,
            "effective_from": date(2026, 8, 5),
            "effective_to": date(2026, 8, 21),
        }
    )
    block = render_documents((chunk,))
    evidence = saved.knowledge.model_copy(
        update={
            "chunks": (chunk,),
            "generation_block": block,
            "generation_tokens": ctx.schema_token_counter.count(block),
            "time_scope": FrozenRangeScope(
                periods=(FrozenPeriod(start=date(2026, 7, 1), end=date(2026, 9, 1)),)
            ),
        }
    )
    ctx.evidence.knowledge = saved.model_copy(update={"knowledge": evidence})
    return EvidenceBundle(data=ctx.evidence.snapshot, knowledge=ctx.evidence.knowledge)
