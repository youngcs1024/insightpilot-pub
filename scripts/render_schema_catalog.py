"""Render complete catalog text with reproducible, explicitly named token metrics."""

import asyncio
import sys

from app.agents.budget import token_bound
from app.core.errors import InsightPilotError
from scripts.schema_catalog_runtime import catalog_service
from scripts.schema_tokens import encoding


def measure(text: str) -> tuple[int, int]:
    """cl100k_base is a comparison baseline, not every model's billed token count."""
    return len(encoding().encode(text, disallowed_special=())), token_bound(text)


async def render() -> str:
    """Return the full valid schema without truncating required semantics."""
    async with catalog_service() as service:
        return await service.render()


def main() -> int:
    """Keep metrics on stderr so stdout remains a usable complete prompt block."""
    try:
        block = asyncio.run(render())
        tokens, upper = measure(block)
        print(block, end="")
        print(
            f"cl100k_base_tokens={tokens} utf8_byte_upper_bound={upper} target=1500 (advisory)",
            file=sys.stderr,
        )
        return 0
    except InsightPilotError as exc:
        print(exc.code + ": " + exc.user_message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
