"""Generate prose from a committed snapshot, with program-owned evidence fields."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.contracts import Answer, AnswerDraft
from app.agents.data.caveats import append_caveats
from app.agents.nodes.common import failed
from app.agents.prompts import FORMAT_ANSWER, SYSTEM
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import ConflictError, InsightPilotError
from app.core.llm_config import ModelRole


async def format_answer(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """No LLM-selected SQL or reference IDs may replace committed evidence."""
    ctx = runtime.context
    try:
        ctx.deadline.check("format_answer")
        if (
            state.evidence_refs is None
            or state.evidence_refs.data_snapshot_id is None
            or state.prepared is None
        ):
            raise ConflictError("missing committed evidence references")
        snapshot = await ctx.evidence.find(ctx.identity, state.evidence_refs.data_snapshot_id)
        if snapshot is None:
            raise ConflictError("committed evidence missing")
        data = snapshot.data
        # This block was budgeted (including JSON escaping) before commit. Audit
        # rows/statistics can be more complete; never render those into the prompt.
        draft = AnswerDraft(markdown="查询成功, 但未返回任何行 (no rows returned)。", confidence=1)
        if data.row_count:
            draft = await ctx.llm.generate_structured(
                ModelRole.SYNTHESIS,
                [
                    SystemMessage(content=SYSTEM + FORMAT_ANSWER),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "question": (
                                    state.rewritten.standalone
                                    if state.rewritten
                                    else state.prepared.question
                                ),
                                "generation_block": data.generation_block,
                                "assumptions": data.assumptions,
                                "sql_scope": data.sql,
                                "sanity_flags": [flag.value for flag in data.sanity_flags],
                            },
                            ensure_ascii=False,
                        )
                    ),
                ],
                AnswerDraft,
                deadline=ctx.deadline,
            )
        answer = Answer(
            markdown=append_caveats(draft.markdown, data.sanity_flags),
            confidence=draft.confidence,
            sql=data.sql,
            assumptions=data.assumptions,
            evidence_refs=state.evidence_refs,
        )
        return Command(update={"answer": answer, "status": "succeeded"}, goto=END)
    except InsightPilotError as exc:
        return failed("format_answer", state, exc)
