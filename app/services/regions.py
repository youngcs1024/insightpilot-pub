"""Resolve explicit names against the business catalog through read-only MCP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.errors import McpResultError
from app.schemas.mcp import QueryArguments
from app.schemas.metric_resolution import RegionScope

if TYPE_CHECKING:
    from app.agents.runtime import McpPort
    from app.core.deadline import Deadline
    from app.schemas.mcp import QueryResultPayload
    from app.schemas.metric_resolution import RegionReference

REGION_QUERY = "SELECT region_id, name, name_en, renamed_from FROM biz.regions ORDER BY region_id"
REGION_LIMIT = 100


def resolve_rows(reference: RegionReference, result: QueryResultPayload) -> RegionScope | None:
    """Require exact, unique matches and a complete, typed catalog response."""
    if result.result_truncated or [column.name for column in result.columns] != [
        "region_id",
        "name",
        "name_en",
        "renamed_from",
    ]:
        raise McpResultError()
    aliases: dict[str, set[int]] = {}
    identifiers: set[int] = set()
    for row in result.rows:
        identifier, *names = row
        if (
            not isinstance(identifier, int)
            or isinstance(identifier, bool)
            or identifier <= 0
            or identifier in identifiers
        ):
            raise McpResultError()
        identifiers.add(identifier)
        for name in names:
            if name is None:
                continue
            if not isinstance(name, str) or not name.strip():
                raise McpResultError()
            aliases.setdefault(name.strip().casefold(), set()).add(identifier)
    selected: set[int] = set()
    for name in reference.names:
        matches = aliases.get(name.strip().casefold(), set())
        if len(matches) != 1:
            return None
        selected.update(matches)
    return RegionScope(region_ids=sorted(selected))


class RegionService:
    """Use the existing MCP timeout/retry policy without another retry layer."""

    def __init__(self, mcp: McpPort) -> None:
        self.mcp = mcp

    async def resolve(
        self, reference: RegionReference, *, deadline: Deadline
    ) -> RegionScope | None:
        """Unknown, conflicting or missing references require clarification."""
        deadline.check("resolve_region")
        if reference.all_regions:
            return None if reference.names else RegionScope()
        if not reference.names:
            return None
        result = await self.mcp.call_tool(
            "execute_readonly_query",
            QueryArguments(sql=REGION_QUERY, max_rows=REGION_LIMIT),
            deadline=deadline,
        )
        deadline.check("resolve_region_complete")
        return resolve_rows(reference, result)
