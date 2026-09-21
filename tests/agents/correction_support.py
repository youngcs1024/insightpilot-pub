"""Bounded SQL correction scenarios shared by graph and checkpoint tests."""

from tests.answer_support import data_draft
from app.agents.runtime import RuntimeContext
from app.schemas.mcp import QueryResultPayload
from app.schemas.sql_correction import CorrectionDecision, SqlCorrectionOutput
from app.services.metric_binding import build_binding
from tests.agents.support import context, metric_intent, sql_candidate
from tests.metric_resolution_support import request, schema

MAX_EXECUTIONS = 3


def correction_context(results: list[QueryResultPayload | Exception]) -> RuntimeContext:
    canonical = build_binding(request(), schema()).binding.resolved_expression
    unqualified = canonical.replace("biz.", "")
    renamed = canonical.replace(" AS gmv", " AS corrected_gmv")
    assert renamed != canonical
    return context(
        responses=[
            metric_intent(),
            sql_candidate(unqualified),
            SqlCorrectionOutput(decision=CorrectionDecision.CORRECTED, sql=canonical),
            SqlCorrectionOutput(decision=CorrectionDecision.CORRECTED, sql=renamed),
            data_draft(markdown="42", confidence=1),
        ]
        if len(results) == MAX_EXECUTIONS
        else [
            metric_intent(),
            sql_candidate(unqualified),
            SqlCorrectionOutput(decision=CorrectionDecision.CORRECTED, sql=canonical),
            data_draft(markdown="42", confidence=1),
        ],
        mcp_results=results,
    )
