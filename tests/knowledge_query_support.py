"""Knowledge-topic fixtures shared by agent, security and storage acceptance."""

from datetime import date
from uuid import uuid4

from app.schemas.knowledge_query import KnowledgeHistoryTurn, KnowledgeRewrite
from app.schemas.retrieval import PolicyPeriod, RangeTimeScope


def topic(year: int = 2026, month: int = 8) -> KnowledgeHistoryTurn:
    return KnowledgeHistoryTurn(turn_id=uuid4(), question=f"{year}年{month}月退货政策",
        answer_summary="退货条件和运费依照适用政策。",
        time_scope=RangeTimeScope(periods=[PolicyPeriod(start=date(year, month, 1), end=date(year, month + 1, 1), label=f"{year}年{month}月")]))


def rewrite(turn: KnowledgeHistoryTurn, text: str = "退货政策中的运费规则") -> KnowledgeRewrite:
    return KnowledgeRewrite(standalone=text, referenced_turn_ids=[turn.turn_id])
