"""Detached extraction inputs and committed source pairs for focused memory tests."""

from app.agents.contracts import Answer, EvidenceRefs, TurnIdentity
from app.db.models.turn import TurnRole, TurnStatus
from app.db.session import Database
from app.repositories.turns import TurnRepository
from app.schemas.memory_extraction import MemoryCandidate, MemoryExtractionInput
from tests import factories

DURABLE = "以后退款率都按退款申请时间算"
TRANSIENT = "看一下华东8月的退款率"


def candidate(**updates: object) -> MemoryCandidate:
    """Make a valid lasting date convention with an exact quote."""
    return MemoryCandidate.model_validate(
        {
            "memory_type": "metric_override",
            "content": {"metric_key": "refund_rate", "patch": {"date_field": "r.requested_at"}},
            "summary": "退款率按申请时间算",
            "confidence": 0.9,
            "evidence_quote": DURABLE,
            **updates,
        }
    )


def extraction_input(**updates: object) -> MemoryExtractionInput:
    """Supply a valid completed assistant envelope without claiming evidence quality."""
    return MemoryExtractionInput.model_validate(
        {
            "role": TurnRole.ASSISTANT,
            "status": TurnStatus.SUCCEEDED,
            "user_message": DURABLE,
            "answer": Answer(
                markdown="已完成本次分析。",
                confidence=0.9,
                assumptions=[],
                sql="",
                evidence_refs=EvidenceRefs(),
                trace_id="memory-test",
            ),
            **updates,
        }
    )


async def source_pair(
    database: Database, inputs: MemoryExtractionInput, *, owner: TurnIdentity | None = None
) -> TurnIdentity:
    """Commit provenance before the background service opens independent sessions."""
    async with database.session() as session, session.begin():
        if owner is None:
            user = factories.user()
            session.add(user)
            await session.flush()
            user_id = user.id
        else:
            user_id = owner.user_id
        conversation = factories.conversation(user_id)
        session.add(conversation)
        await session.flush()
        turn = await TurnRepository(session, user_id).create_pair(
            conversation.id, inputs.user_message, "memory-test"
        )
        turn.status = inputs.status
        turn.answer = inputs.answer.model_dump(mode="json")
        turn.content = inputs.answer.markdown
        return TurnIdentity(user_id=user_id, conversation_id=conversation.id, turn_id=turn.id)
