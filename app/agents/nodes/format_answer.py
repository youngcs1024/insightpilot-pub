"""Render only committed evidence; own every final analytical answer."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.contracts import Answer, AnswerDraft, EvidenceBundle, EvidenceRefs, Route
from app.agents.data.caveats import append_caveats
from app.agents.nodes.common import failed
from app.agents.prompts import FORMAT_ANSWER, SYSTEM
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import ConflictError, InsightPilotError
from app.core.llm_config import ModelRole
from app.schemas.knowledge import KnowledgeGeneration
from app.schemas.memory import FormatPreferenceContent
from app.schemas.metric_resolution import ClarificationKind, MetricClarification


def format_preference(state: AgentState) -> FormatPreferenceContent | None:
    """Use only finalized presentation context, identically for every route."""
    if state.context is None or state.context.format_preference is None:
        return None
    return state.context.format_preference.model_copy(deep=True)


def _no_evidence(state: AgentState) -> Command[str]:
    clarification = state.data_clarification
    if clarification is None and state.knowledge_clarification is not None:
        clarification = MetricClarification(
            kind=ClarificationKind(state.knowledge_clarification.kind.value),
            message=state.knowledge_clarification.message,
        )
    if state.failures:
        return Command(update={"status": "failed"}, goto=END)
    if clarification is not None:
        return Command(update={"clarification": clarification,
                               "abstained": True, "status": "abstained"}, goto=END)
    answer = Answer(
        markdown="未找到足够的支持证据。请补充相关政策、时间范围或更具体的问题。",
        confidence=0, sql="", assumptions=[], evidence_refs=EvidenceRefs(), abstained=True,
    )
    return Command(update={"answer": answer,
                           "abstained": True, "status": "abstained"}, goto=END)


async def _data_draft(
    state: AgentState, ctx: RuntimeContext, bundle: EvidenceBundle
) -> AnswerDraft | None:
    if bundle.data is None:
        return None
    data = bundle.data.data
    draft = AnswerDraft(markdown="查询成功, 但未返回任何行 (no rows returned)。", confidence=1)
    if data.row_count:
        preference = format_preference(state)
        # BOTH gets a scoped data task, never an invitation to infer a policy cause.
        question = (state.route.data_intent if state.route else
                    state.rewritten.standalone if state.rewritten else state.question)
        draft = await ctx.llm.generate_structured(
            ModelRole.SYNTHESIS,
            [SystemMessage(content=SYSTEM + FORMAT_ANSWER), HumanMessage(content=json.dumps({
                "question": question,
                "generation_block": data.generation_block,
                "assumptions": data.assumptions,
                "sql_scope": data.sql,
                "sanity_flags": [flag.value for flag in data.sanity_flags],
                "format_preference": preference.model_dump() if preference else None,
            }, ensure_ascii=False))],
            AnswerDraft, deadline=ctx.deadline,
        )
    return draft.model_copy(update={"markdown": append_caveats(draft.markdown, data.sanity_flags)})


async def _knowledge_draft(state: AgentState, ctx: RuntimeContext, bundle: EvidenceBundle) -> KnowledgeGeneration | None:
    if bundle.knowledge is None:
        return None
    if ctx.knowledge_generation is None:
        raise ConflictError("missing knowledge generation service")
    return await ctx.knowledge_generation.generate(
        bundle.knowledge.knowledge, deadline=ctx.deadline,
        format_preference=format_preference(state), presentation_request=state.question
    )


def _assemble(
    state: AgentState, bundle: EvidenceBundle, data: AnswerDraft | None,
    knowledge: KnowledgeGeneration | None,
) -> Answer:
    both = state.route is not None and state.route.route is Route.BOTH
    valid_knowledge = knowledge is not None and knowledge.abstention is None
    missing = []
    if both and data is None:
        missing.append("data")
    if both and not valid_knowledge:
        missing.append("knowledge")
    degraded = list(dict.fromkeys([*state.degraded_components, *missing]))
    pieces = []
    confidence = []
    if data is not None:
        pieces.append(("### 数据结论\n\n" if both else "") + data.markdown)
        confidence.append(data.confidence)
    if valid_knowledge and knowledge is not None:
        text = "\n\n".join(
            passage.text + " " + " ".join(f"[{identifier}]" for identifier in passage.chunk_ids)
            for passage in knowledge.passages
        )
        pieces.append(("### 知识依据\n\n" if both else "") + text)
        score = bundle.knowledge.knowledge.top_rerank_score if bundle.knowledge else None
        confidence.append(score if score is not None else 0.5)
    abstained = not pieces
    if abstained:
        pieces.append("现有证据不足以形成可验证的回答。请补充相关政策或缩小问题范围。")
    if both:
        pieces.append("### 缺失信息\n\n" + (
            "；".join("业务数据未能核实" if name == "data" else "知识依据不足或不可用"
                     for name in missing) if missing else
            "以上分别呈现数据与文档事实，尚未建立因果关系；需要进一步的归因证据。"
        ))
    if degraded and not abstained:
        names = {"data": "业务数据", "knowledge": "知识依据", "rerank": "重排服务"}
        pieces.insert(0, "> 本次为部分回答，以下组件缺失或降级：" +
                      "、".join(names.get(name, name) for name in degraded) + "。")
        confidence.append(0.5)
    assumptions = list(dict.fromkeys([
        *(bundle.data.data.assumptions if bundle.data else []),
        *(bundle.knowledge.knowledge.assumptions if bundle.knowledge else []),
    ]))
    return Answer(
        markdown="\n\n".join(pieces), confidence=min(confidence) if confidence else 0,
        sql=bundle.data.data.sql if bundle.data else "", assumptions=assumptions,
        evidence_refs=bundle.refs,
        citations=list(knowledge.citations) if valid_knowledge and knowledge else [],
        knowledge_passages=list(knowledge.passages) if valid_knowledge and knowledge else [],
        degraded_components=degraded, abstained=abstained,
    )


async def format_answer(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """All routes share one final owner; no draft IDs can replace committed IDs."""
    ctx = runtime.context
    try:
        ctx.deadline.check("format_answer")
        refs = state.evidence_refs
        if refs is None or (refs.data_snapshot_id is None and refs.knowledge_snapshot_id is None):
            return _no_evidence(state)
        bundle = await ctx.evidence.read_bundle(ctx.identity)
        if bundle.refs != refs:
            raise ConflictError("committed evidence missing")
        data = await _data_draft(state, ctx, bundle)
        knowledge = await _knowledge_draft(state, ctx, bundle)
        answer = _assemble(state, bundle, data, knowledge)
        status = "abstained" if answer.abstained else (
            "degraded" if answer.degraded_components else "succeeded"
        )
        return Command(update={"answer": answer, "status": status,
                               "abstained": answer.abstained}, goto=END)
    except InsightPilotError as exc:
        return failed("format_answer", state, exc)
