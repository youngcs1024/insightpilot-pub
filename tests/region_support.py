"""Typed business-region responses for deterministic MCP contract tests."""

from app.schemas.mcp import ColumnSpec, QueryResultPayload, SqlValue
from app.services.regions import REGION_QUERY

FOLLOWUP_QUESTION = "那华南呢？"  # noqa: RUF001 -- exact roadmap acceptance question.


def region_result(rows: list[list[SqlValue]] | None = None) -> QueryResultPayload:
    values = (
        rows
        if rows is not None
        else [
            [3, "华东一区", "East China", "华东"],
            [2, "华南", "South China", None],
        ]
    )
    return QueryResultPayload(
        executed_sql=REGION_QUERY,
        limit_applied=True,
        execution_ms=1,
        mcp_call_id="region-lookup",
        columns=[
            ColumnSpec(name="region_id", type="int4"),
            ColumnSpec(name="name", type="text"),
            ColumnSpec(name="name_en", type="text"),
            ColumnSpec(name="renamed_from", type="text"),
        ],
        rows=values,
        row_count=len(values),
        result_truncated=False,
    )
