"""A saved patch is an optional candidate; explicit intent and policy remain authoritative."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from app.core.deadline import Deadline
from app.core.errors import DeadlineExceededError, McpPolicyRejected, UpstreamUnavailableError
from app.schemas.mcp import PolicyReason, ValidationStatus
from app.schemas.metric_resolution import MetricPatch, RegionScope
from app.services.metric_binding import build_binding
from app.services.metric_override import resolve_binding
from tests.fakes.mcp_client import FakeMcpClient
from tests.metric_resolution_support import override, request, schema


def rejection() -> McpPolicyRejected:
    return McpPolicyRejected(ValidationStatus.INVALID, [PolicyReason.INVALID_METRIC_BINDING])


async def test_policy_fallback_preserves_explicit_fields_and_original_deadline() -> None:
    value = request(patch=MetricPatch(add_filters=["o.gross_amount > 100"]))
    value.override = override(MetricPatch(date_field="o.created_at"))
    value.region_scope = RegionScope(region_ids=[2])
    mcp = FakeMcpClient([])
    mcp.enqueue_metric(rejection())
    deadline = Deadline(float("inf"))
    with capture_logs() as logs:
        result = await resolve_binding(value, schema(), mcp, deadline=deadline)
    assert len(mcp.metric_calls) == 2
    assert mcp.metric_calls[0].date_field == "o.created_at"
    assert mcp.metric_calls[1].date_field == "o.paid_at"
    assert result.binding.override_id is None
    assert "o.gross_amount > 100" in result.binding.filters_applied
    assert "o.region_id IN (2)" in result.binding.filters_applied
    assert any("未通过数据访问校验" in note for note in result.assumptions)
    assert not any("自定义定义" in note for note in result.assumptions)
    assert any(event["event"] == "metric_saved_patch_rejected" for event in logs)


async def test_second_refusal_propagates_without_further_fallback() -> None:
    value = request()
    value.override = override(MetricPatch(date_field="o.created_at"))
    mcp = FakeMcpClient([])
    mcp.enqueue_metric(rejection(), rejection())
    with pytest.raises(McpPolicyRejected):
        await resolve_binding(value, schema(), mcp, deadline=Deadline(float("inf")))
    assert len(mcp.metric_calls) == 2


@pytest.mark.parametrize("saved", [None, MetricPatch(date_field="o.created_at")])
async def test_no_fallback_when_saved_patch_did_not_survive_precedence(saved: MetricPatch | None) -> None:
    value = request(patch=MetricPatch(date_field="o.paid_at"))
    value.override = override(saved) if saved else None
    mcp = FakeMcpClient([])
    mcp.enqueue_metric(rejection())
    with pytest.raises(McpPolicyRejected):
        await resolve_binding(value, schema(), mcp, deadline=Deadline(float("inf")))
    assert len(mcp.metric_calls) == 1


@pytest.mark.parametrize("failure", [UpstreamUnavailableError(), DeadlineExceededError(), asyncio.CancelledError()])
async def test_operational_failure_never_discards_memory(failure: BaseException) -> None:
    value = request()
    value.override = override(MetricPatch(date_field="o.created_at"))
    mcp = AsyncMock(resolve_metric=AsyncMock(side_effect=failure))
    with pytest.raises(type(failure)):
        await resolve_binding(value, schema(), mcp, deadline=Deadline(float("inf")))
    assert mcp.resolve_metric.await_count == 1


async def test_fallback_does_not_renew_expired_deadline() -> None:
    value = request()
    value.override = override(MetricPatch(date_field="o.created_at"))
    mcp = AsyncMock(resolve_metric=AsyncMock(side_effect=rejection()))
    with pytest.raises(DeadlineExceededError):
        await resolve_binding(value, schema(), mcp, deadline=Deadline(0))
    assert mcp.resolve_metric.await_count == 1


@pytest.mark.parametrize("patch", [
    MetricPatch(date_field="o.invented"),
    MetricPatch(add_filters=["o.invented > 0"]),
    MetricPatch(add_filters=["o.order_id IN (SELECT order_id FROM biz.orders)"]),
])
def test_invalid_saved_patch_is_atomic_and_noted(patch: MetricPatch) -> None:
    value = request()
    value.override = override(patch)
    result = build_binding(value, schema())
    assert result.binding.override_id is None
    assert result.binding.date_field == "o.paid_at"
    assert any("无法应用" in note for note in result.assumptions)


def test_partial_explicit_override_names_only_surviving_saved_fields() -> None:
    value = request(patch=MetricPatch(date_field="o.paid_at"))
    value.override = override(MetricPatch(date_field="o.created_at", add_filters=["o.gross_amount > 100"]))
    result = build_binding(value, schema())
    note = next(note for note in result.assumptions if "自定义定义" in note)
    assert "GMV" in note
    assert "2026-07-14" in note
    assert "o.gross_amount > 100" in note
    assert "o.created_at" not in note
    assert result.binding.override_id == value.override.id
