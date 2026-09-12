"""CI gate: compare all metadata with live MCP structure, bypassing caches."""

import asyncio

from app.core.errors import InsightPilotError
from scripts.schema_catalog_runtime import catalog_service


async def check() -> int:
    """Print the typed report and fail on any drift or unavailable dependency."""
    async with catalog_service() as service:
        report = await service.validate_against_live_schema()
        print(report.model_dump_json(indent=2))
        return 0 if report.valid else 1


def main() -> int:
    """Use a synchronous process entrypoint around the async service."""
    try:
        return asyncio.run(check())
    except InsightPilotError as exc:
        print(exc.code + ": " + exc.user_message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
