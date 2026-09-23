"""Ingest only the authored attack corpus into the isolated red-team collection."""

import asyncio
from pathlib import Path

from scripts.ingest import IngestionProcessSettings, run


def main() -> None:
    raise SystemExit(
        asyncio.run(
            run(IngestionProcessSettings.load(), Path("data/corpus/adversarial"))
        )
    )


if __name__ == "__main__":
    main()
