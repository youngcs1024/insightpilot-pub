"""Schema-keyed scripts, with strict request context and evidence-reference checks."""

import time
from collections import Counter
from uuid import UUID

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agents.contracts import (
    DataAnswerDraft,
    Route,
    RouteDecision,
    RouterInput,
    SqlGeneratorOutput,
)
from app.core.deadline import Deadline
from app.core.llm_config import ModelRole
from app.schemas.knowledge import KnowledgeDraft, KnowledgePassage
from app.schemas.memory_extraction import MemoryExtraction, MemoryExtractionInput
from app.schemas.metric_resolution import MetricIntent, RegionReference
from app.schemas.synthesis import (
    CellReference,
    Claim,
    ClaimKind,
    DataGenerationView,
    SynthesisOutput,
)
from app.services.llm.contracts import CompletionRequest
from tests.e2e.contracts import (
    BOTH_QUESTION,
    CLARIFY_QUESTION,
    DATA_QUESTION,
    FOLLOWUP_QUESTION,
    KNOWLEDGE_QUESTION,
    Scenario,
    ScriptStatus,
)
from tests.fakes.chat_model import FakeChatModel


class PromptMetadata(BaseModel):
    question: str = ""
    valid_chunk_ids: list[UUID] = Field(default_factory=list)
    generation_block: str = ""
    missing_components: list[str] = Field(default_factory=list)
    prior_queries_for_reference: list[str] = Field(default_factory=list)


def human_inputs(request: CompletionRequest) -> list[str]:
    return [message.content for message in request.messages if message.role == "user"]


def gmv_sql(*, south: bool) -> str:
    return (
        "SELECT SUM(o.gross_amount - COALESCE(o.discount_amount, 0)) AS gmv "
        "FROM biz.orders o JOIN biz.customers c USING (customer_id) "
        "WHERE o.paid_at >= TIMESTAMPTZ '2026-08-01T00:00:00+08:00' "
        "AND o.paid_at < TIMESTAMPTZ '2026-09-01T00:00:00+08:00' "
        "AND o.paid_at IS NOT NULL AND o.status <> 'cancelled' "
        "AND c.is_test_account = FALSE" + (" AND o.region_id IN (2)" if south else "")
    )


def refund_sql() -> str:
    return """WITH numerator AS (
SELECT date_trunc('month', r.requested_at AT TIME ZONE 'Asia/Shanghai') AS month,
COUNT(DISTINCT o.order_id)::numeric AS refunded
FROM biz.refunds r JOIN biz.orders o USING (order_id)
JOIN biz.customers c USING (customer_id)
WHERE r.requested_at >= TIMESTAMPTZ '2026-07-01T00:00:00+08:00'
AND r.requested_at < TIMESTAMPTZ '2026-09-01T00:00:00+08:00'
AND r.status <> 'rejected' AND o.paid_at IS NOT NULL AND o.status <> 'cancelled'
AND c.is_test_account = FALSE AND o.region_id IN (3) GROUP BY 1
), denominator AS (
SELECT date_trunc('month', o.paid_at AT TIME ZONE 'Asia/Shanghai') AS month,
COUNT(DISTINCT o.order_id) AS paid FROM biz.orders o
JOIN biz.customers c USING (customer_id)
WHERE o.paid_at >= TIMESTAMPTZ '2026-07-01T00:00:00+08:00'
AND o.paid_at < TIMESTAMPTZ '2026-09-01T00:00:00+08:00'
AND o.paid_at IS NOT NULL AND o.status <> 'cancelled'
AND c.is_test_account = FALSE AND o.region_id IN (3) GROUP BY 1
) SELECT d.month, COALESCE(n.refunded, 0) / NULLIF(d.paid, 0) AS refund_rate
FROM denominator d LEFT JOIN numerator n USING (month) ORDER BY d.month"""


def data_claim(view: DataGenerationView, *, refund: bool) -> Claim:
    row, column = (1, 1) if refund else (0, 0)
    value = view.sample_rows[row][column]
    return Claim(
        text=f"{'退款申请订单率' if refund else 'GMV'}为 {value}。",
        kind=ClaimKind.FACT_DATA,
        confidence=1,
        data_refs=[CellReference(row=row, column=column, value=value)],
    )


def memory_response(inputs: list[str]) -> MemoryExtraction:
    """Ordinary fixture questions are transient; reject malformed completion envelopes."""
    source = MemoryExtractionInput.model_validate_json(inputs[0])
    assert source.role == "assistant"
    assert source.status == "succeeded"
    assert not source.answer.abstained
    assert not source.answer.degraded_components
    return MemoryExtraction()


class Script:
    """No live fallback; a schema can be consumed only the declared number of times."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.status = ScriptStatus()
        self.counts: Counter[str] = Counter()
        self.expected = Counter({"RouteDecision": 1})
        if scenario in {Scenario.DATA, Scenario.FOLLOWUP, Scenario.BOTH, Scenario.KNOWLEDGE}:
            self.expected["MemoryExtraction"] = 1
        if scenario in {Scenario.DATA, Scenario.FOLLOWUP, Scenario.BOTH, Scenario.CHAOS}:
            self.expected.update({"MetricIntent": 1, "SqlGeneratorOutput": 1})
        if scenario in {Scenario.DATA, Scenario.FOLLOWUP}:
            self.expected.update({"DataAnswerDraft": 1})
        if scenario is Scenario.KNOWLEDGE:
            self.expected.update({"KnowledgeDraft": 1})
        if scenario in {Scenario.BOTH, Scenario.CHAOS}:
            self.expected.update({"SynthesisOutput": 1})

    def snapshot(self) -> ScriptStatus:
        result = self.status.model_copy(deep=True)
        result.remaining = {
            name: count - self.counts[name] for name, count in self.expected.items()
        }
        return result

    async def complete(self, request: CompletionRequest) -> str:
        assert request.response_format is not None, "Native structured output required"
        name = request.response_format["json_schema"]["schema"]["title"]
        assert isinstance(name, str), "Unexpected schema name"
        assert self.counts[name] < self.expected[name], "Unexpected schema"
        self.counts[name] += 1
        self.status.calls.append(name)
        inputs = human_inputs(request)
        response = (
            memory_response(inputs) if name == "MemoryExtraction" else self.response(name, inputs)
        )
        fake = FakeChatModel([response])
        validated = await fake.generate_structured(
            ModelRole.ROUTER
            if name == "RouteDecision"
            else ModelRole.MEMORY_EXTRACT
            if name == "MemoryExtraction"
            else ModelRole.SYNTHESIS,
            [HumanMessage(content=content) for content in human_inputs(request)],
            type(response),
            deadline=Deadline(time.monotonic() + 10),
        )
        assert len(fake.calls) == 1
        return validated.model_dump_json()

    def response(self, name: str, inputs: list[str]) -> BaseModel:
        both = self.scenario in {Scenario.BOTH, Scenario.CHAOS}
        south = self.scenario is Scenario.FOLLOWUP
        metadata = PromptMetadata.model_validate_json(inputs[0])
        if name == "RouteDecision":
            source = RouterInput.model_validate_json(inputs[0])
            expected_question = {
                Scenario.DATA: DATA_QUESTION,
                Scenario.FOLLOWUP: FOLLOWUP_QUESTION,
                Scenario.KNOWLEDGE: KNOWLEDGE_QUESTION,
                Scenario.BOTH: BOTH_QUESTION,
                Scenario.CHAOS: BOTH_QUESTION,
                Scenario.CLARIFY: CLARIFY_QUESTION,
            }[self.scenario]
            assert source.question == expected_question, "Wrong router question"
            if south:
                assert any(
                    message.content == DATA_QUESTION
                    for message in source.routing_context.recent_messages
                ), "Missing history"
            route = (
                Route.BOTH
                if both
                else Route.KNOWLEDGE_ONLY
                if self.scenario is Scenario.KNOWLEDGE
                else Route.CLARIFY
                if self.scenario is Scenario.CLARIFY
                else Route.DATA_ONLY
            )
            return RouteDecision(
                route=route,
                confidence=1,
                data_intent="计算2026年7月和8月华东退款申请订单率"
                if both
                else "计算2026年8月华南GMV"
                if south
                else "计算2026年8月GMV",
                knowledge_intent="核查2026年7月和8月退款政策" if both else KNOWLEDGE_QUESTION,
                clarification_question="请说明要查询的指标或政策。"
                if route is Route.CLARIFY
                else "",
            )
        if name == "MetricIntent":
            return MetricIntent(
                metric_keys=["refund_rate" if both else "gmv"],
                period_expression="2026-07-01到2026-08-31" if both else "2026年8月",
                grain="month" if both else "total",
                region_mentioned=both or south,
                region=RegionReference(names=["华东"] if both else ["华南"] if south else []),
            )
        if name == "SqlGeneratorOutput":
            if south:
                assert metadata.prior_queries_for_reference, "Missing prior SQL"
                assert "华南" in metadata.question, "Missing resolved follow-up"
            return SqlGeneratorOutput(
                thinking="Fixed E2E script",
                sql=refund_sql() if both else gmv_sql(south=south),
                tables_used=["biz.orders", "biz.customers", *(["biz.refunds"] if both else [])],
            )
        if name == "DataAnswerDraft":
            view = DataGenerationView.model_validate_json(metadata.generation_block)
            return DataAnswerDraft(claims=[data_claim(view, refund=False)])
        assert metadata.valid_chunk_ids, "Missing real retrieved chunk IDs"
        if name == "KnowledgeDraft":
            return KnowledgeDraft(
                passages=(
                    KnowledgePassage(
                        text="定制商品与已拆封的个人卫生用品不适用七天无理由退货。",
                        chunk_ids=tuple(metadata.valid_chunk_ids),
                    ),
                )
            )
        assert name == "SynthesisOutput", "Unscripted output"
        claims = []
        if self.scenario is not Scenario.CHAOS:
            view = DataGenerationView.model_validate_json(inputs[1])
            claims.append(data_claim(view, refund=True))
        else:
            assert metadata.missing_components == ["data"]
        claims.append(
            Claim(
                text="退货规则列明适用例外。",
                kind=ClaimKind.FACT_DOCUMENT,
                chunk_ids=metadata.valid_chunk_ids,
                confidence=1,
            )
        )
        claims.append(
            Claim(
                text="政策导致退款率上升。",
                kind=ClaimKind.FACT_DOCUMENT,
                chunk_ids=metadata.valid_chunk_ids,
                confidence=1,
            )
        )
        return SynthesisOutput(claims=claims, unanswered=["需要额外归因证据。"])
