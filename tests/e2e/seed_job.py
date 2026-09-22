"""Test-image seed entrypoint using the unchanged production export/import path."""

import asyncio
from pathlib import Path
from tempfile import mkdtemp

from data.seed.files import export
from scripts.seed import import_seed
from scripts.seed_settings import SeedSettings
from tests.e2e.dataset import business


def main() -> None:
    directory = Path(mkdtemp(prefix="e2e-seed-"))
    export(business(), directory)
    print(asyncio.run(import_seed(directory, SeedSettings.load())).model_dump_json())


if __name__ == "__main__":
    main()
