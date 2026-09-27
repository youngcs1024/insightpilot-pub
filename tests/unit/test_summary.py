"""Incremental summary behavior and complete tool-aware history boundaries."""

from time import monotonic
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from structlog.testing import capture_logs

from app.agents.contracts import TurnIdentity
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.errors import ContextBudgetExceeded, LlmUnavailableError
from app.db.models.turn import TurnStatus
from app.schemas.summary import SummaryOutput, SummaryWork
from app.services.context_history import complete_tool_groups, history_text, trim_context_messages
from app.services.conversations import ConversationService
from app.services.schema_tokens import SchemaTokenCounter
from app.services.summary import SummaryService, update_summary
from tests.fakes.chat_model import FakeChatModel


def work(status: TurnStatus = TurnStatus.SUCCEEDED) -> SummaryWork:
    return SummaryWork(expected_seq=2, covered_seq=4, existing_summary="用户查看华东七月GMV。", latest_user="八月呢？", latest_answer="八月查询失败，只有七月数据。", status=status)


async def test_incremental_summary_extends_not_rewrites() -> None:
    llm = FakeChatModel([SummaryOutput(summary="用户查看华东七月GMV；八月请求尚未获得数据。")])
    result = await update_summary(work(TurnStatus.DEGRADED), llm, SchemaTokenCounter(), deadline=Deadline(monotonic()+10))
    assert "七月" in result and "八月" in result
    content = str(llm.calls[0].messages[1].content)
    assert work().existing_summary in content and "degraded" in content


@pytest.mark.parametrize("status", [TurnStatus.SUCCEEDED, TurnStatus.DEGRADED, TurnStatus.ABSTAINED])
async def test_normal_terminal_statuses_are_summary_data(status: TurnStatus) -> None:
    llm = FakeChatModel([SummaryOutput(summary="仍需澄清八月范围。")])
    await update_summary(work(status), llm, SchemaTokenCounter(), deadline=Deadline(monotonic()+10))
    assert status.value in str(llm.calls[0].messages[1].content)


@pytest.mark.parametrize("status", [TurnStatus.FAILED, TurnStatus.RUNNING])
async def test_failed_or_running_turn_does_not_generate(status: TurnStatus) -> None:
    llm = FakeChatModel([])
    assert await update_summary(work(status), llm, SchemaTokenCounter(), deadline=Deadline(monotonic()+10)) == work().existing_summary
    assert not llm.calls


async def test_summary_overflow_rejected() -> None:
    llm = FakeChatModel([SummaryOutput(summary=" word" * 600)])
    with pytest.raises(ContextBudgetExceeded):
        await update_summary(work(), llm, SchemaTokenCounter(), deadline=Deadline(monotonic()+10))


async def test_growth_is_logged_not_claimed_as_semantic_validation() -> None:
    llm = FakeChatModel([SummaryOutput(summary="较长但仍在预算内的摘要，保留七月话题和八月未回答的问题。" * 3)])
    with capture_logs() as logs:
        await update_summary(work(), llm, SchemaTokenCounter(), deadline=Deadline(monotonic()+10))
    assert any(event["event"] == "summary_growth_suspicious" for event in logs)


async def test_summary_failure_keeps_previous(settings: Settings) -> None:
    conversations = AsyncMock(spec=ConversationService)
    conversations.summary_work.return_value = work()
    llm = FakeChatModel([LlmUnavailableError()])
    summary = SummaryService(conversations, settings, llm, SchemaTokenCounter())
    await summary.run(TurnIdentity(user_id=uuid4(), conversation_id=uuid4(), turn_id=uuid4()))
    conversations.advance_summary.assert_not_awaited()


async def test_stale_summary_reloads_before_retry(settings: Settings) -> None:
    conversations = AsyncMock(spec=ConversationService)
    conversations.summary_work.side_effect = [work(), None]
    conversations.advance_summary.return_value = False
    summary = SummaryService(conversations, settings, FakeChatModel([SummaryOutput(summary="更新摘要。")]), SchemaTokenCounter())
    await summary.run(TurnIdentity(user_id=uuid4(), conversation_id=uuid4(), turn_id=uuid4()))
    assert conversations.summary_work.await_count == 2
    conversations.advance_summary.assert_awaited_once()


def messages() -> list[BaseMessage]:
    return [
        HumanMessage(content="old " * 300), AIMessage(content="old answer"),
        HumanMessage(content="calculate"),
        AIMessage(content="", tool_calls=[{"name":"foo","args":{},"id":"a"}, {"name":"foo","args":{},"id":"b"}]),
        ToolMessage(content="one", tool_call_id="a"), ToolMessage(content="two", tool_call_id="b"),
        AIMessage(content="done"), HumanMessage(content="next"),
    ]


@pytest.mark.parametrize("limit", [15, 40, 90, 200, 1500])
def test_trim_messages_preserves_tool_call_pairs(limit: int) -> None:
    counter = SchemaTokenCounter()
    trimmed = trim_context_messages(messages(), counter, max_tokens=limit)
    assert counter.count(history_text(trimmed)) <= limit
    calls = [call["id"] for m in trimmed if isinstance(m, AIMessage) for call in m.tool_calls]
    results = [m.tool_call_id for m in trimmed if isinstance(m, ToolMessage)]
    assert sorted(calls) == sorted(results)
    assert not trimmed or isinstance(trimmed[0], HumanMessage)


def test_incomplete_multi_tool_group_is_removed_whole() -> None:
    original = messages()
    incomplete = [m for m in original if not isinstance(m, ToolMessage) or m.tool_call_id != "b"]
    clean = complete_tool_groups(incomplete)
    assert not any(isinstance(m, ToolMessage) or isinstance(m, AIMessage) and m.tool_calls for m in clean)
    assert clean[-1].content == "next"
