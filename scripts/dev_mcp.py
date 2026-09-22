"""Inspect the schema boundary using only API-role MCP client configuration."""

import argparse
import asyncio
import sys
import time

from pydantic import ValidationError

from app.clients.mcp_client import McpClient
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.errors import InsightPilotError
from app.core.logging import setup_logging
from app.schemas.schema_tools import GetSchemaArgs, SchemaResponse


async def read_schema(args: GetSchemaArgs) -> SchemaResponse:
    """Return the actual tool response without locally reconstructing schema text."""
    settings = Settings.load()
    settings.observability.log_level = "WARNING"
    setup_logging(settings)
    client = McpClient(settings.mcp)
    try:
        return await client.get_schema(
            args, deadline=Deadline(time.monotonic() + settings.schema_catalog.operation_timeout_s)
        )
    finally:
        await client.aclose()


def main() -> int:
    """Expose a bounded diagnostic for partial rejections and safe metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=["get_schema"])
    parser.add_argument("--tables")
    parser.add_argument("--include-samples", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    try:
        result = asyncio.run(read_schema(GetSchemaArgs(
            tables=args.tables.split(",") if args.tables is not None else None,
            include_samples=args.include_samples,
            refresh=args.refresh,
        )))
        print(result.model_dump_json(indent=2))
        return 0
    except ValidationError:
        print("VALIDATION_ERROR: Invalid schema tool arguments.", file=sys.stderr)
        return 1
    except InsightPilotError as exc:
        print(exc.code + ": " + exc.user_message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
