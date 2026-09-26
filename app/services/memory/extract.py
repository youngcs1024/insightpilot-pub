"""Conservative extraction after finalization, with bounded transactional writes."""

import asyncio
from time import monotonic

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import metrics
from pydantic import ValidationError as PydanticValidationError

from app.agents.contracts import Answer, TurnIdentity
from app.agents.prompts import MEMORY_EXTRACT
from app.agents.runtime import StructuredLlmPort
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.errors import (
    ConflictError,
    LlmStructuredOutputError,
    MemoryExtractionError,
    NotFoundError,
)
from app.core.llm_config import ModelRole
from app.core.observability import TraceMetadata, observe
from app.db.models.turn import TurnRole, TurnStatus
from app.db.session import Database
from app.repositories.memory import MemoryRepository
from app.repositories.turns import TurnRepository
from app.schemas.memory_extraction import MemoryExtraction, MemoryExtractionInput
from app.schemas.memory_write import WriteOutcome
from app.services.memory.supersede import write

logger = structlog.get_logger(__name__)
MIN_CONFIDENCE = 0.7
_failures = metrics.get_meter(__name__).create_counter("memory_extraction_failures")


def eligible(inputs: MemoryExtractionInput) -> bool:
    """Successful user messages or inconsistent answers are not completed analysis turns."""
    return (
        inputs.role is TurnRole.ASSISTANT
        and inputs.status is TurnStatus.SUCCEEDED
        and not inputs.answer.abstained
        and not inputs.answer.degraded_components
    )


async def extract(
    inputs: MemoryExtractionInput, llm: StructuredLlmPort, *, deadline: Deadline
) -> MemoryExtraction:
    """Return validated candidates; persistence and exception isolation belong to the caller."""
    if not eligible(inputs):
        logger.info("memory_extraction_rejected", reason="turn_status", status=inputs.status.value)
        return MemoryExtraction()
    deadline.check("memory_extract")
    try:
        result = await llm.generate_structured(
            ModelRole.MEMORY_EXTRACT,
            [
                SystemMessage(content=MEMORY_EXTRACT),
                HumanMessage(content=inputs.model_dump_json()),
            ],
            MemoryExtraction,
            deadline=deadline,
        )
        # Revalidate even an adapter returning mutated or model_construct() output.
        result = MemoryExtraction.model_validate(result.model_dump(mode="json", warnings=False))
    except (PydanticValidationError, LlmStructuredOutputError):
        logger.exception("memory_extraction_rejected", reason="schema", exc_info=False)
        raise LlmStructuredOutputError() from None
    deadline.check("memory_extract_complete")
    accepted = MemoryExtraction()
    for candidate in result.candidates:
        if candidate.confidence < MIN_CONFIDENCE:
            logger.info("memory_candidate_rejected", reason="confidence")
        elif (
            not candidate.evidence_quote.strip()
            or candidate.evidence_quote not in inputs.user_message
        ):
            logger.warning("memory_candidate_rejected", reason="evidence_quote")
        else:
            accepted.candidates.append(candidate)
    logger.info("memory_extraction_completed", accepted=len(accepted.candidates))
    return accepted


class MemoryExtractionService:
    """Own fresh sessions and a new finite background budget, never a request session."""

    def __init__(self, database: Database, settings: Settings, llm: StructuredLlmPort) -> None:
        self.database = database
        self.llm = llm
        self.timeout_s = settings.http.request_timeout_s
        self.database_timeout_s = settings.database.command_timeout_s

    async def run(self, identity: TurnIdentity) -> None:
        """Let spawn observe/count a sanitized error without changing the completed turn."""
        deadline = Deadline(monotonic() + self.timeout_s)
        try:
            with observe("memory_extract", TraceMetadata(turn_id=str(identity.turn_id))):
                await self._process(identity, deadline)
        except Exception:
            _failures.add(1)
            logger.exception(
                "memory_extraction_failed", turn_id=str(identity.turn_id), exc_info=False
            )
            raise MemoryExtractionError() from None

    async def _process(self, identity: TurnIdentity, deadline: Deadline) -> None:
        async with asyncio.timeout(deadline.remaining()):
            inputs = await self._load(identity)
            if inputs is None:
                return
            result = await extract(inputs, self.llm, deadline=deadline)
            if result.candidates:
                await self._write(identity, result)

    async def _load(self, identity: TurnIdentity) -> MemoryExtractionInput | None:
        async with (
            asyncio.timeout(self.database_timeout_s),
            self.database.session() as session,
        ):
            repo = TurnRepository(session, identity.user_id)
            turn = await repo.get(identity.conversation_id, identity.turn_id)
            if turn is None:
                raise NotFoundError()
            if turn.role is not TurnRole.ASSISTANT or turn.status is not TurnStatus.SUCCEEDED:
                logger.info(
                    "memory_extraction_rejected", reason="turn_status", status=turn.status.value
                )
                return None
            if turn.answer is None or turn.reply_to_turn_id is None:
                raise ConflictError("Memory extraction requires completed turn provenance")
            source = await repo.get(identity.conversation_id, turn.reply_to_turn_id)
            if source is None or source.role is not TurnRole.USER:
                raise ConflictError("Memory extraction requires an owned user message")
            return MemoryExtractionInput(
                role=turn.role,
                status=turn.status,
                user_message=source.content,
                answer=Answer.model_validate(turn.answer),
            )

    async def _write(self, identity: TurnIdentity, result: MemoryExtraction) -> list[WriteOutcome]:
        outcomes: list[WriteOutcome] = []
        async with (
            self.database.session() as session,
            asyncio.timeout(self.database_timeout_s),
            session.begin(),
        ):
            repo = MemoryRepository(session, identity.user_id)
            await repo.lock_writes()
            for candidate in result.candidates:
                outcome = await write(candidate, repo, identity.turn_id)
                if outcome.superseded_id in {item.memory_id for item in outcomes}:
                    logger.warning(
                        "memory_same_turn_conflict",
                        turn_id=str(identity.turn_id),
                        old_memory_id=str(outcome.superseded_id),
                        new_memory_id=str(outcome.memory_id),
                    )
                outcomes.append(outcome)
        for outcome in outcomes:
            logger.info(
                "memory_write_completed",
                turn_id=str(identity.turn_id),
                status=outcome.status.value,
                memory_id=str(outcome.memory_id),
                superseded_id=str(outcome.superseded_id) if outcome.superseded_id else None,
            )
        return outcomes
