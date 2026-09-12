"""Probe the complete authenticated local-container to remote-model route."""

import argparse
import asyncio
import time

from app.clients.model_runtime import ModelRuntimeClient
from app.core.deadline import Deadline
from scripts.model_diagnostics_settings import ModelDiagnosticsSettings


async def run() -> None:
    """Print model identity and readiness only, never credentials or input text."""
    client = ModelRuntimeClient(ModelDiagnosticsSettings.load().model_runtime)
    try:
        print((await client.ready(deadline=Deadline(time.monotonic() + 2))).model_dump_json())
    finally:
        await client.aclose()


def main() -> None:
    """The probe is always an authenticated readiness operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", action="store_true")
    parser.parse_args()
    asyncio.run(run())


if __name__ == "__main__":
    main()
