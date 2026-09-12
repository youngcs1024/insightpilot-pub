"""Exact region aliases, ambiguity, corrupt upstream results and MCP failures."""

import pytest

from app.core.errors import McpResultError, McpUnavailableError
from app.schemas.mcp import SqlValue
from app.schemas.metric_resolution import RegionReference
from app.services.regions import REGION_QUERY, RegionService, resolve_rows
from tests.auth_support import deadline
from tests.fakes.mcp_client import FakeMcpClient
from tests.region_support import region_result


@pytest.mark.parametrize("name", ["华东", "华东一区", "east china"])
async def test_region_alias_uses_stable_id(name: str) -> None:
    mcp = FakeMcpClient([region_result()])
    result = await RegionService(mcp).resolve(RegionReference(names=[name]), deadline=deadline())
    assert result.region_ids == [3]
    assert len(mcp.calls) == 1
    assert mcp.calls[0].arguments.sql == REGION_QUERY


@pytest.mark.parametrize(
    "names", [["missing"], ["华南", "missing"], ["'; DROP TABLE biz.orders;--"]]
)
async def test_unknown_region_never_invents_or_executes_input(names: list[str]) -> None:
    mcp = FakeMcpClient([region_result()])
    assert (
        await RegionService(mcp).resolve(RegionReference(names=names), deadline=deadline()) is None
    )
    assert mcp.calls[0].arguments.sql == REGION_QUERY


def test_ambiguous_alias_requires_clarification() -> None:
    rows = [[3, "first", "First", "old"], [4, "second", "Second", "old"]]
    assert resolve_rows(RegionReference(names=["old"]), region_result(rows)) is None


@pytest.mark.parametrize("identifier", [None, True, "3", -1])
def test_malformed_region_id_fails_closed(identifier: SqlValue) -> None:
    with pytest.raises(McpResultError):
        resolve_rows(RegionReference(names=["x"]), region_result([[identifier, "x", "X", None]]))


def test_truncated_catalog_is_not_an_authoritative_lookup() -> None:
    with pytest.raises(McpResultError):
        resolve_rows(
            RegionReference(names=["华南"]),
            region_result().model_copy(update={"result_truncated": True}),
        )


async def test_explicit_all_regions_and_missing_reference_need_no_query() -> None:
    mcp = FakeMcpClient([])
    service = RegionService(mcp)
    assert (
        await service.resolve(RegionReference(all_regions=True), deadline=deadline())
    ).region_ids == []
    assert await service.resolve(RegionReference(), deadline=deadline()) is None
    assert (
        await service.resolve(
            RegionReference(names=["华南"], all_regions=True), deadline=deadline()
        )
        is None
    )
    assert not mcp.calls


async def test_lookup_failure_propagates_without_new_retry_loop() -> None:
    mcp = FakeMcpClient([McpUnavailableError()])
    with pytest.raises(McpUnavailableError):
        await RegionService(mcp).resolve(RegionReference(names=["华南"]), deadline=deadline())
    assert len(mcp.calls) == 1


@pytest.mark.parametrize(
    "rows",
    [
        [[3, "x", "X", None], [3, "y", "Y", None]],
        [[3, 42, "X", None]],
        [[3, "", "X", None]],
    ],
)
def test_corrupt_alias_catalog_fails_closed(rows: list[list[SqlValue]]) -> None:
    with pytest.raises(McpResultError):
        resolve_rows(RegionReference(names=["x"]), region_result(rows))


def test_unexpected_columns_fail_closed() -> None:
    payload = region_result()
    payload.columns[0].name = "unexpected"
    with pytest.raises(McpResultError):
        resolve_rows(RegionReference(names=["华东"]), payload)
