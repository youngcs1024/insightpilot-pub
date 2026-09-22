"""Test-image ingestion entrypoint; PostgreSQL and Milvus remain real stores."""

import asyncio
from pathlib import Path
from tempfile import mkdtemp

from scripts.ingest import IngestionProcessSettings, run
from tests.e2e.dataset import corpus


def main() -> None:
    directory = Path(mkdtemp(prefix="e2e-corpus-")) / "corpus"
    corpus(directory)
    raise SystemExit(asyncio.run(run(IngestionProcessSettings.load(), directory)))


if __name__ == "__main__":
    main()
