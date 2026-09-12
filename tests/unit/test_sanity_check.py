"""Conservative observations inspect returned rows and never request correction."""

import asyncio
from collections.abc import Iterator
from decimal import Decimal
from unittest.mock import Mock

import pytest
from langgraph.runtime import Runtime
from pydantic import ValidationError
from structlog.testing import capture_logs

from app.agents.contracts import DataEvidence
from app.agents.data import sanity
from app.agents.data.nodes.sanity_check import sanity_check
from app.agents.data.state import DataAgentState
from app.agents.summarize import package_result
from app.core.config_models import SanitySettings
from app.schemas.mcp import ColumnSpec, QueryResultPayload
from app.schemas.sanity import SanityCheckResult, SanityFlag
from app.services.graph import serializer
from tests.agents.support import context
from tests.factories import query_result
from tests.factories import sanity_payload as payload


def test_empty_result_flagged_not_failed() -> None:
    outcome = sanity.check_result(payload([]), SanitySettings())
    assert outcome.flags == [SanityFlag.EMPTY_RESULT]
    assert not outcome.check_failed


def test_null_scalar_flagged() -> None:
    outcome = sanity.check_result(payload([[None]]), SanitySettings())
    assert outcome.flags == [SanityFlag.ALL_NULL, SanityFlag.SINGLE_NULL_SCALAR]
    assert not outcome.check_failed


def test_all_null_means_any_whole_column_not_any_null_cell() -> None:
    result = payload(
        [[None, 1], [None, 2]],
        columns=[ColumnSpec(name="amount", type="numeric"), ColumnSpec(name="count", type="int8")],
    )
    assert sanity.check_result(result, SanitySettings()).flags == [SanityFlag.ALL_NULL]
    assert sanity.check_result(payload([[None], [1]]), SanitySettings()).flags == []


def test_truncation_flagged() -> None:
    assert sanity.check_result(payload([[1]], truncated=True), SanitySettings()).flags == [
        SanityFlag.TRUNCATED
    ]
    evidence = package_result(payload([[1]] * 21), [])
    assert evidence.result_summary.sample_truncated
    assert evidence.limit_applied
    assert not evidence.sanity_flags


def test_negative_money_flagged() -> None:
    settings = SanitySettings(money_columns=["amount"])
    assert sanity.check_result(payload([["-0.01"]]), settings).flags == [SanityFlag.NEGATIVE_MONEY]
    assert sanity.check_result(payload([[-1]]), SanitySettings()).flags == []


@pytest.mark.parametrize("value", ["1000000000000.0000000000000000001", "-1000000000001"])
def test_extreme_magnitude_flagged(value: str) -> None:
    assert sanity.check_result(payload([[value]]), SanitySettings()).flags == [
        SanityFlag.EXTREME_MAGNITUDE
    ]


@pytest.mark.parametrize("value", ["1000000000000", "-1000000000000", "0", "-0.0"])
def test_threshold_boundary_and_normal_zero_are_not_suspicious(value: str) -> None:
    assert sanity.check_result(payload([[value]]), SanitySettings()).flags == []


def test_nonzero_expectation_and_cardinality_are_explicit() -> None:
    settings = SanitySettings(nonzero_columns=["amount"], expected_max_rows=1)
    assert sanity.check_result(payload([[0]]), settings).flags == [SanityFlag.SUSPICIOUS_ZERO]
    assert sanity.check_result(payload([[1], [2]]), settings).flags == [
        SanityFlag.CARDINALITY_SPIKE
    ]
    assert sanity.check_result(payload([[1], [2]]), SanitySettings()).flags == []


@pytest.mark.parametrize("column_type", ["text", "bool", "date"])
def test_non_numeric_columns_are_not_interpreted(column_type: str) -> None:
    settings = SanitySettings(money_columns=["amount"], nonzero_columns=["amount"])
    assert (
        sanity.check_result(
            payload([["-99999999999999999"], ["0"]], column_type=column_type), settings
        ).flags
        == []
    )


def test_boolean_values_in_numeric_column_are_not_numbers() -> None:
    settings = SanitySettings(nonzero_columns=["amount"], extreme_magnitude=Decimal("0.1"))
    assert sanity.check_result(payload([[True], [False]]), settings).flags == []


def test_missing_case_mismatched_and_duplicate_aliases_skip_business_rules() -> None:
    settings = SanitySettings(money_columns=["Amount"], nonzero_columns=["missing"])
    assert sanity.check_result(payload([[-1], [0]]), settings).flags == []
    duplicate = payload([[-1, 0]], columns=[ColumnSpec(name="amount", type="numeric")] * 2)
    settings = SanitySettings(money_columns=["amount"], nonzero_columns=["amount"])
    assert sanity.check_result(duplicate, settings).flags == []


def test_check_covers_rows_beyond_sample_and_deduplicates_flags() -> None:
    result = payload([[1]] * 20 + [[-1], [-2]])
    settings = SanitySettings(money_columns=["amount"])
    evidence = package_result(result, [], sanity_settings=settings)
    assert all(row == [1] for row in evidence.rows)
    assert evidence.sanity_flags == [SanityFlag.NEGATIVE_MONEY]


@pytest.mark.parametrize("value", ["invalid-number", "NaN", "Infinity", "-Infinity"])
def test_check_failure_keeps_prior_flags_and_logs_safe_event(value: str) -> None:
    result = payload([[value]], truncated=True)
    before = result.model_dump()
    with capture_logs() as logs:
        outcome = sanity.check_result(result, SanitySettings())
    assert outcome.check_failed
    assert outcome.flags == [SanityFlag.TRUNCATED]
    assert result.model_dump() == before
    assert any(log["event"] == "sanity_check_failed" for log in logs)
    assert all("rows" not in log for log in logs)


def test_injected_check_failure_does_not_lose_good_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(result: QueryResultPayload, settings: SanitySettings) -> Iterator[SanityFlag]:
        yield SanityFlag.TRUNCATED
        raise RuntimeError("synthetic check failure")

    monkeypatch.setattr(sanity, "result_flags", broken)
    evidence = package_result(payload([[42]], truncated=True), ["original assumption"])
    assert evidence.rows == [[42]]
    assert evidence.assumptions == ["original assumption"]
    assert evidence.sanity_flags == [SanityFlag.TRUNCATED]


def test_cancellation_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sanity, "result_flags", Mock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        sanity.check_result(payload([[1]]), SanitySettings())


@pytest.mark.parametrize("flag", list(SanityFlag))
async def test_flags_never_trigger_retry(flag: SanityFlag, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sanity, "result_flags", lambda *args: iter([flag]))
    ctx = context()
    state = DataAgentState(question="GMV", generated_sql="SELECT 42", query_result=query_result())
    before = state.model_dump()
    command = await sanity_check(state, Runtime(context=ctx))
    assert set(command.update) == {"sanity_check_result"}
    assert command.update["sanity_check_result"].flags == [flag]
    assert not command.goto
    assert state.model_dump() == before
    assert not ctx.mcp.calls
    assert not ctx.llm.calls


async def test_missing_node_result_is_an_advisory_check_failure() -> None:
    command = await sanity_check(DataAgentState(question="GMV"), Runtime(context=context()))
    assert command.update["sanity_check_result"].check_failed
    assert not command.goto


async def test_node_consumes_runtime_settings_and_serializes_its_output() -> None:
    ctx = context()
    ctx.settings.data_agent.sanity = SanitySettings(money_columns=["amount"])
    state = DataAgentState(question="GMV", query_result=payload([[-1]]))
    command = await sanity_check(state, Runtime(context=ctx))
    restored = DataAgentState.model_validate_json(
        state.model_copy(update=command.update).model_dump_json()
    )
    assert restored.sanity_check_result.flags == [SanityFlag.NEGATIVE_MONEY]
    assert restored.query_result == state.query_result


def test_trace_failure_cannot_fail_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sanity,
        "update_current_observation",
        Mock(side_effect=RuntimeError("synthetic trace error")),
    )
    outcome = sanity.check_result(payload([]), SanitySettings())
    assert outcome.flags == [SanityFlag.EMPTY_RESULT]
    assert not outcome.check_failed


def test_historical_state_and_evidence_remain_readable() -> None:
    state = DataAgentState.model_validate({"schema_version": 1, "question": "historical"})
    assert state.query_result is None
    assert not state.sanity_check_result.flags
    evidence = package_result(payload([]), []).model_dump(mode="json")
    evidence["sanity_flags"] = ["empty_result", "all_null"]
    assert DataEvidence.model_validate(evidence).sanity_flags == [
        SanityFlag.EMPTY_RESULT,
        SanityFlag.ALL_NULL,
    ]


def test_new_sanity_contracts_roundtrip_through_checkpoint_allowlist() -> None:
    serde = serializer()
    outcome = SanityCheckResult(flags=list(SanityFlag), check_failed=True)
    assert serde.loads_typed(serde.dumps_typed(outcome)) == outcome


def test_historical_enum_checkpoint_path_remains_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    serde = serializer()
    with monkeypatch.context() as patch:
        patch.setattr(SanityFlag, "__module__", "app.agents.contracts")
        encoded = serde.dumps_typed([SanityFlag.EMPTY_RESULT, SanityFlag.ALL_NULL])
    assert serde.loads_typed(encoded) == [SanityFlag.EMPTY_RESULT, SanityFlag.ALL_NULL]


@pytest.mark.parametrize(
    "invalid",
    [
        {"extreme_magnitude": "0"},
        {"extreme_magnitude": "-1"},
        {"extreme_magnitude": "1e101"},
        {"extreme_magnitude": "NaN"},
        {"extreme_magnitude": "Infinity"},
        {"expected_max_rows": -1},
        {"expected_max_rows": 1_000_000_001},
        {"expected_max_rows": 1.5},
        {"money_columns": [""]},
        {"money_columns": ["x" * 129]},
        {"nonzero_columns": ["x"] * 201},
        {"unknown": True},
    ],
)
def test_configuration_bounds(invalid: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SanitySettings.model_validate(invalid)


def test_nested_environment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_DATA_AGENT__SANITY__EXTREME_MAGNITUDE", "12.5")
    monkeypatch.setenv("IP_DATA_AGENT__SANITY__MONEY_COLUMNS", '["amount"]')
    monkeypatch.setenv("IP_DATA_AGENT__SANITY__NONZERO_COLUMNS", '["count"]')
    monkeypatch.setenv("IP_DATA_AGENT__SANITY__EXPECTED_MAX_ROWS", "0")
    settings = context().settings.data_agent.sanity
    assert settings.extreme_magnitude == Decimal("12.5")
    assert settings.money_columns == ["amount"]
    assert settings.nonzero_columns == ["count"]
    assert settings.expected_max_rows == 0
