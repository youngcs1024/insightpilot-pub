"""Best-effort incremental summaries with durable, monotonic coverage."""

import asyncio
from time import monotonic

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.contracts import TurnIdentity
from app.agents.prompts import SUMMARIZE
from app.agents.runtime import StructuredLlmPort
from app.core.budget import ContextBudget, ContextSlot, TokenCounter
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.llm_config import ModelRole
from app.core.observability import TraceMetadata, observe
from app.db.models.turn import TurnStatus
from app.schemas.summary import SummaryOutput, SummaryWork
from app.services.conversations import ConversationService
from app.services.llm.budget import prompt_budget

logger = structlog.get_logger(__name__)


async def update_summary(
    work: SummaryWork, llm: StructuredLlmPort, counter: TokenCounter, *, deadline: Deadline
) -> str:
    """Summarize only terminal nonfailed pairs, preserving explicit status and chronology."""
    if work.status in {TurnStatus.FAILED, TurnStatus.RUNNING}:
        return work.existing_summary
    result = await llm.generate_structured(
        ModelRole.SUMMARIZE,
        [SystemMessage(content=SUMMARIZE), HumanMessage(content=work.model_dump_json())],
        SummaryOutput,
        deadline=deadline,
        budget=prompt_budget(summary=work.existing_summary),
    )
    result = SummaryOutput.model_validate(result.model_dump())
    summary = result.summary.strip()
    if not summary:
        # Revalidate a whitespace-only response without inventing a summary.
        SummaryOutput.model_validate({"summary": summary})
    ContextBudget(counter).charge(ContextSlot.SUMMARY, summary)
    old_tokens = counter.count(work.existing_summary)
    new_tokens = counter.count(summary)
    if old_tokens and new_tokens > old_tokens * 1.5:
        logger.warning("summary_growth_suspicious", old_tokens=old_tokens, new_tokens=new_tokens)
    return summary


class SummaryService:
    """Load bounded work in short sessions; never hold a DB transaction during generation."""

    def __init__(
        self,
        conversations: ConversationService,
        settings: Settings,
        llm: StructuredLlmPort,
        counter: TokenCounter,
    ) -> None:
        self.conversations = conversations
        self.llm = llm
        self.counter = counter
        self.timeout_s = settings.http.request_timeout_s
        self.database_timeout_s = settings.database.command_timeout_s

    async def run(self, identity: TurnIdentity) -> None:
        """Failure retains committed progress and never changes the completed answer."""
        deadline = Deadline(monotonic() + self.timeout_s)
        try:
            with observe("conversation_summary", TraceMetadata(turn_id=str(identity.turn_id))):
                async with asyncio.timeout(deadline.remaining()):
                    await self._catch_up(identity, deadline)
        except Exception:
            logger.exception("conversation_summary_failed", exc_info=False)

    async def _catch_up(self, identity: TurnIdentity, deadline: Deadline) -> None:
        while True:
            deadline.check("summary_load")
            async with asyncio.timeout(deadline.budget(self.database_timeout_s)):
                work = await self.conversations.summary_work(identity)
            if work is None:
                return
            summary = await update_summary(work, self.llm, self.counter, deadline=deadline)
            deadline.check("summary_commit")
            async with asyncio.timeout(deadline.budget(self.database_timeout_s)):
                advanced = await self.conversations.advance_summary(identity, work, summary)
            logger.info(
                "conversation_summary_advanced" if advanced else "conversation_summary_stale",
                covered_seq=work.covered_seq,
            )
