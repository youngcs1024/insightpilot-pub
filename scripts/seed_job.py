"""Explicit one-shot container: generate in writable /tmp, then atomically import."""

import asyncio
from pathlib import Path
from tempfile import mkdtemp

from data.seed.contracts import Parameters
from data.seed.files import export
from data.seed.generation import generate
from scripts.seed import import_seed
from scripts.seed_settings import SeedSettings


def main() -> None:
    """Use bounded container storage; no bind mount or database reset is required."""
    settings = SeedSettings.load()
    directory = Path(mkdtemp(prefix="insightpilot-seed-"))
    export(generate(Parameters()), directory)
    print(asyncio.run(import_seed(directory, settings)).model_dump_json())


if __name__ == "__main__":
    main()
