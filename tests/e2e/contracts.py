"""Typed control contracts available only inside the isolated test deployment."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.core.observability import TraceMetadata


class Scenario(StrEnum):
    DATA = "data"
    KNOWLEDGE = "knowledge"
    BOTH = "both"
    CLARIFY = "clarify"
    FOLLOWUP = "followup"
    CHAOS = "chaos"
    RED_SQL = "red_sql"
    RED_CREDENTIAL = "red_credential"
    RED_DOCUMENT = "red_document"
    RED_FALSE_POLICY = "red_false_policy"
    RED_CITATION = "red_citation"
    RED_CAUSALITY = "red_causality"
    RED_CROSS_USER = "red_cross_user"
    RED_WIDEN = "red_widen"


class ScriptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: Scenario


class ScriptStatus(BaseModel):
    remaining: dict[str, int] = Field(default_factory=dict)
    calls: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    embed_query: int = 0
    embed_document: int = 0
    rerank: int = 0
    sql_waiting: bool = False


class SpanRecord(BaseModel):
    request_id: str
    name: str
    metadata: TraceMetadata


class Observations(BaseModel):
    spans: list[SpanRecord]


DATA_QUESTION = "2026 年 8 月的 GMV 是多少?"
KNOWLEDGE_QUESTION = "七天无理由退货有哪些例外?"
BOTH_QUESTION = "为什么2026年8月华东退款率比7月上升?"
CLARIFY_QUESTION = "帮我看看昨天那个"
FOLLOWUP_QUESTION = "那华南呢?"
