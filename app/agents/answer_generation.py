"""Single data-source statements use the same immutable reference guard as synthesis."""

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.contracts import DataAnswerDraft, EvidenceBundle
from app.agents.prompts import FORMAT_ANSWER, SYNTHESIS_REPAIR, SYSTEM
from app.agents.runtime import RuntimeContext
from app.agents.synthesis_generation import synthesis_input
from app.agents.synthesis_validation import validate_output
from app.core.errors import FabricatedCitation, SynthesisValidationError
from app.core.llm_config import ModelRole
from app.schemas.memory import FormatPreferenceContent
from app.schemas.synthesis import Claim, ClaimKind, RowCountReference, SynthesisOutput

logger = structlog.get_logger(__name__)


async def data_claims(
    question: str,
    bundle: EvidenceBundle,
    preference: FormatPreferenceContent | None,
    ctx: RuntimeContext,
) -> list[Claim]:
    """One content repair, no additional numerical evidence or operational retry layer."""
    if bundle.data is None:
        return []
    data = bundle.data.data
    if data.row_count == 0:
        return [Claim(
            text="查询成功, 但未返回任何行 (no rows returned)。",
            kind=ClaimKind.FACT_DATA,
            confidence=1,
            data_refs=[RowCountReference(value=0)],
        )]
    source = synthesis_input(question, bundle)
    for attempt in (1, 2):
        ctx.deadline.check("format_answer")
        draft = await ctx.llm.generate_structured(
            ModelRole.SYNTHESIS,
            [
                SystemMessage(content=SYSTEM + FORMAT_ANSWER + (
                    "\n" + SYNTHESIS_REPAIR if attempt > 1 else ""
                )),
                HumanMessage(content=json.dumps({
                    "question": question,
                    "generation_block": data.generation_block,
                    "assumptions": data.assumptions,
                    "sql_scope": data.sql,
                    "sanity_flags": [flag.value for flag in data.sanity_flags],
                    "format_preference": preference.model_dump() if preference else None,
                }, ensure_ascii=False)),
            ],
            DataAnswerDraft,
            deadline=ctx.deadline,
        )
        try:
            checked = validate_output(SynthesisOutput(claims=draft.claims), source)
        except (FabricatedCitation, SynthesisValidationError) as exc:
            logger.exception("answer_reference_rejected", attempt=attempt, code=exc.code)
            continue
        if any(claim.data_refs and claim.kind is not ClaimKind.UNSUPPORTED
               for claim in checked.claims):
            return checked.claims
        return []
    return []
