"""Command-line boundary for the live provider probe."""

import argparse
import asyncio
from pathlib import Path

import httpx
from pydantic import ValidationError
from pydantic_settings import SettingsError

from spikes.provider.models import MODEL, Settings
from spikes.provider.reporting import write_report
from spikes.provider.runner import run

ROOT = Path(__file__).resolve().parents[2]


async def execute(settings: Settings, output: Path) -> int:
    """Open a private HTTP client with explicit TLS, timeout and redirect policy."""
    async with httpx.AsyncClient(
        timeout=settings.provider.timeout_seconds, trust_env=False, follow_redirects=False
    ) as client:
        report = await run(settings.provider, client)
    write_report(report, output)
    print(f"Evidence written to {output}; execution_complete={report.execution_complete}")
    return 0 if report.execution_complete else 1


def main() -> int:
    """Validate inputs without exposing secret-bearing exception messages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=MODEL, choices=[MODEL])
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/provider_capabilities.json"))
    args = parser.parse_args()
    output = (ROOT / args.out).resolve()
    if not output.is_relative_to(ROOT / "docs") or output.suffix != ".json":
        print("Output must be a .json file inside this project's docs directory.")
        return 2
    env_file = (ROOT / args.env_file).resolve() if args.env_file is not None else None
    if env_file is not None and (not env_file.is_relative_to(ROOT) or not env_file.is_file()):
        print("Explicit env file must exist inside the InsightPilot project.")
        return 2
    try:
        settings = Settings(_env_file=env_file)
    except (ValidationError, SettingsError, OSError):
        print(
            "Invalid probe configuration. Fill workspace ID and API key using spikes/provider/.env.example."
        )
        return 2
    try:
        return asyncio.run(execute(settings, output))
    except OSError:
        print("Could not write the probe artifacts. Check the output directory permissions.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
