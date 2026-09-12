"""Exact returned-row statistics and explicitly bounded generation evidence."""

# ruff: noqa: PLR2004 -- explicit row, column and numeric boundary examples.

import json
from decimal import Decimal

import pytest

from app.agents.budget import DATA_TOKENS, token_bound
from app.agents.contracts import DataEvidence
from app.agents.data.summarize import generation_block, render_block, summarize_result
from app.agents.summarize import COLUMN_CAP, SAMPLE_CAP, package_result
from app.core.errors import ContextBudgetExceeded, McpResultError, ValidationError
from app.schemas.mcp import RESULT_CEILING, ColumnSpec, QueryResultPayload, SqlValue
from tests.agents.support import result

NULL_ROWS = 2


def multicolumn_result(rows: list[list[SqlValue]], columns: list[ColumnSpec]) -> QueryResultPayload:
    """Validate the complete shape at once instead of using the single-column factory."""
    return QueryResultPayload(
        executed_sql="SELECT 42",
        limit_applied=False,
        execution_ms=1,
        mcp_call_id="multicolumn-summary-case",
        columns=columns,
        rows=rows,
        row_count=len(rows),
        result_truncated=False,
    )


def test_statistics_computed_over_full_result_not_sample() -> None:
    payload = result([[str(i) + ".01"] for i in range(100)])
    payload.columns = [ColumnSpec(name="amount", type="numeric")]
    evidence = package_result(payload, [])
    assert Decimal(evidence.result_summary.columns[0].total) == Decimal("4951.00")
    assert len(evidence.rows) == SAMPLE_CAP
    assert evidence.result_summary.sample_truncated
    assert not evidence.result_summary.result_truncated


def test_decimal_precision_preserved() -> None:
    value = "123456789012345678901234567890.123456789"
    payload = result([[value], [value]])
    payload.columns = [ColumnSpec(name="amount", type="numeric")]
    assert (
        package_result(payload, []).result_summary.columns[0].total
        == "246913578024691357802469135780.246913578"
    )


def test_budget_enforced_and_flags_preserved() -> None:
    payload = result([["x" * 1000] for _ in range(30)])
    payload.columns = [ColumnSpec(name="description", type="text")]
    payload.result_truncated = True
    evidence = package_result(payload, [])
    block = json.loads(evidence.generation_block)
    assert block["result_truncated"]
    assert block["sample_truncated"]
    assert len(block["sample_rows"]) < SAMPLE_CAP
    assert block["statistics_scope"] == "returned_rows"
    assert token_bound(evidence.generation_block) <= DATA_TOKENS


def test_null_column_reported() -> None:
    evidence = package_result(result([[None], [None]]), [])
    assert evidence.result_summary.columns[0].null_count == NULL_ROWS
    assert evidence.result_summary.columns[0].total is None
    assert evidence.sanity_flags[0].value == "all_null"


def test_required_statistics_over_budget_fail_closed() -> None:
    payload = result()
    payload.columns[0].name = "wide" * DATA_TOKENS
    with pytest.raises(ContextBudgetExceeded):
        package_result(payload, [])


def test_wide_result_column_cap() -> None:
    column_count = 200
    payload = result([])
    payload.columns = [ColumnSpec(name=f"c{i}", type="int4") for i in range(column_count)]
    evidence = package_result(payload, [])
    block = json.loads(evidence.generation_block)
    assert 0 < len(block["statistics"]) <= COLUMN_CAP
    assert block["columns_omitted"] == column_count - len(block["statistics"])
    assert len(evidence.result_summary.columns) == COLUMN_CAP
    assert len(evidence.columns) == column_count
    assert token_bound(evidence.generation_block) <= DATA_TOKENS


def test_exact_cap_not_mislabeled_truncated() -> None:
    payload = result([[1]] * 1000)
    evidence = package_result(payload, [])
    assert evidence.result_summary.columns[0].total == "1000"
    assert evidence.result_summary.sample_truncated
    assert not evidence.result_summary.result_truncated


def test_mixed_decimal_scales_preserved() -> None:
    payload = result(
        [
            ["10000000000000000000000000000000000000000"],
            ["0.0000000000000000000000000000000000000001"],
        ]
    )
    payload.columns = [ColumnSpec(name="amount", type="numeric")]
    assert (
        package_result(payload, []).result_summary.columns[0].total
        == "10000000000000000000000000000000000000000.0000000000000000000000000000000000000001"
    )


def data_slot_size(block: str) -> int:
    # Match the real formatter, including the outer key and string escaping.
    return len(json.dumps({"generation_block": block}, ensure_ascii=False).encode("utf-8"))


def test_large_result_truncated_to_sample() -> None:
    payload = result([[i] for i in range(RESULT_CEILING)])
    evidence = package_result(payload, [])
    block = json.loads(evidence.generation_block)
    assert evidence.rows == payload.rows[:SAMPLE_CAP]
    assert block["sample_rows"] == payload.rows[:SAMPLE_CAP]
    assert block["returned_row_count"] == RESULT_CEILING
    assert block["sample_row_count"] == SAMPLE_CAP
    assert evidence.result_summary.columns[0].total == "12497500"
    assert data_slot_size(evidence.generation_block) <= DATA_TOKENS


@pytest.mark.parametrize("value", ['"' * 600, "\\" * 600, "退款\n\t" * 200])
def test_budget_enforced(value: str) -> None:
    payload = result([[value] for _ in range(SAMPLE_CAP)])
    payload.columns = [ColumnSpec(name="description", type="text")]
    evidence = package_result(payload, [])
    block = json.loads(evidence.generation_block)
    assert data_slot_size(evidence.generation_block) <= DATA_TOKENS
    assert token_bound(evidence.generation_block) <= DATA_TOKENS
    assert block["sample_rows"] == []
    assert block["sample_truncated"]
    assert not evidence.result_summary.sample_truncated
    assert evidence.rows == payload.rows
    assert len(block["statistics"]) == 1


def test_partial_sample_keeps_whole_values_and_original_order() -> None:
    payload = result([["x" * 100 + str(i)] for i in range(SAMPLE_CAP)])
    payload.columns = [ColumnSpec(name="description", type="text")]
    evidence = package_result(payload, [])
    block = json.loads(evidence.generation_block)
    count = block["sample_row_count"]
    assert 0 < count < SAMPLE_CAP
    assert block["sample_rows"] == payload.rows[:count]
    assert block["sample_truncated"]
    assert not evidence.result_summary.sample_truncated
    assert data_slot_size(evidence.generation_block) <= DATA_TOKENS


def test_statistics_only_then_column_omission() -> None:
    column_count = 200
    payload = multicolumn_result(
        [[100] * column_count for _ in range(50)],
        [ColumnSpec(name=f"amount_{i}", type="numeric") for i in range(column_count)],
    )
    evidence = package_result(payload, [])
    block = json.loads(evidence.generation_block)
    assert block["sample_rows"] == []
    assert block["sample_truncated"]
    assert 0 < len(block["statistics"]) < COLUMN_CAP
    assert block["columns_omitted"] == column_count - len(block["statistics"])
    assert block["returned_column_count"] == column_count
    assert [stats[0] for stats in block["statistics"]] == [
        f"amount_{i}" for i in range(len(block["statistics"]))
    ]
    assert all(stats[-1] == "5000" for stats in block["statistics"])
    assert len(evidence.columns) == column_count
    assert len(evidence.result_summary.columns) == COLUMN_CAP
    assert evidence.rows == payload.rows[:SAMPLE_CAP]
    assert data_slot_size(evidence.generation_block) <= DATA_TOKENS


def test_model_sample_projects_the_retained_columns() -> None:
    column_count = 31
    payload = multicolumn_result(
        [list(range(column_count))],
        [ColumnSpec(name=f"c{i}", type="int4") for i in range(column_count)],
    )
    summary = summarize_result(payload)
    block = json.loads(render_block(summary, summary.sample_rows, column_count, statistics_count=3))
    assert block["sample_rows"] == [[0, 1, 2]]
    assert [stats[0] for stats in block["statistics"]] == ["c0", "c1", "c2"]
    assert block["columns_omitted"] == 28
    assert summary.sample_rows == payload.rows


@pytest.mark.parametrize(
    ("row_count", "capped"), [(0, False), (10, False), (10, True), (21, False), (21, True)]
)
def test_sample_cap_distinct_from_result_cap(row_count: int, capped: bool) -> None:
    payload = result([[1] for _ in range(row_count)])
    payload.result_truncated = capped
    evidence = package_result(payload, [])
    block = json.loads(evidence.generation_block)
    assert evidence.result_summary.sample_truncated is (row_count > SAMPLE_CAP)
    assert block["sample_truncated"] is (len(block["sample_rows"]) < row_count)
    assert block["result_truncated"] is capped
    assert evidence.result_summary.result_truncated is capped
    assert block["statistics_scope"] == "returned_rows"


def test_empty_result_is_not_all_null() -> None:
    evidence = package_result(result([]), [])
    block = json.loads(evidence.generation_block)
    assert block["all_null_columns"] == []
    assert not block["sample_truncated"]
    assert block["sample_row_count"] == 0
    assert block["statistics"][0][-1] is None


def test_all_null_columns_are_explicit_in_generation_block() -> None:
    evidence = package_result(result([[None], [None]]), [])
    block = json.loads(evidence.generation_block)
    assert block["all_null_columns"] == [0]
    assert block["statistics"][0][2:] == [2, 0, None, None, None]


def test_text_distinct_date_range_and_nulls_cover_all_rows() -> None:
    payload = multicolumn_result(
        [["east", "2026-08-01"]] * SAMPLE_CAP + [["south", "2026-03-01"], [None, None]],
        [ColumnSpec(name="region", type="text"), ColumnSpec(name="day", type="date")],
    )
    summary = summarize_result(payload)
    assert summary.columns[0].distinct_count == 2
    assert summary.columns[0].null_count == 1
    assert summary.columns[1].minimum == "2026-03-01"
    assert summary.columns[1].maximum == "2026-08-01"


@pytest.mark.parametrize("value", ["invalid-number", "NaN", "Infinity"])
def test_invalid_numeric_result_is_typed_failure(value: str) -> None:
    payload = result([[value]])
    payload.columns = [ColumnSpec(name="amount", type="numeric")]
    with pytest.raises(McpResultError):
        summarize_result(payload)


@pytest.mark.parametrize("max_rows", [0, 1, 20, 200])
def test_custom_audit_sample_cap(max_rows: int) -> None:
    payload = result([[1]] * 201)
    summary = summarize_result(payload, max_rows=max_rows)
    assert len(summary.sample_rows) == max_rows
    assert summary.sample_truncated
    assert summary.columns[0].total == "201"
    block = json.loads(generation_block(summary, len(payload.columns)))
    assert len(block["sample_rows"]) <= SAMPLE_CAP


@pytest.mark.parametrize("max_rows", [-1, 201, True, 1.5])
def test_invalid_sample_cap_is_typed_failure(max_rows: int) -> None:
    with pytest.raises(ValidationError):
        summarize_result(result(), max_rows=max_rows)


def test_audit_sample_detaches_from_source_rows() -> None:
    payload = result([[1]])
    summary = summarize_result(payload)
    payload.rows[0][0] = 2
    assert summary.sample_rows == [[1]]
    assert summary.columns[0].total == "1"


def test_historical_full_width_summary_remains_readable() -> None:
    evidence = package_result(result(), [])
    historical = evidence.model_dump(mode="json")
    historical["result_summary"]["columns"] *= 40
    historical["generation_block"] = "historical renderer output"
    restored = DataEvidence.model_validate(historical)
    assert len(restored.result_summary.columns) == 40
    assert restored.generation_block == "historical renderer output"


def test_overlong_trailing_column_is_omitted_whole() -> None:
    payload = multicolumn_result(
        [[42, 99]],
        [ColumnSpec(name="kept", type="int4"), ColumnSpec(name="wide" * DATA_TOKENS, type="int4")],
    )
    evidence = package_result(payload, [])
    block = json.loads(evidence.generation_block)
    assert len(block["statistics"]) == 1
    assert block["statistics"][0][0] == "kept"
    assert block["statistics"][0][-1] == "42"
    assert block["columns_omitted"] == 1
    assert block["sample_rows"] == []
    assert len(evidence.result_summary.columns) == 2
    assert evidence.rows == [[42, 99]]


def test_empty_result_without_columns() -> None:
    payload = result([])
    payload.columns = []
    block = json.loads(package_result(payload, []).generation_block)
    assert block["statistics"] == []
    assert block["returned_column_count"] == 0
    assert block["columns_omitted"] == 0
    assert not block["sample_truncated"]
