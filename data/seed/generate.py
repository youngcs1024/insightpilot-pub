"""Generate reproducible synthetic business data without database credentials."""

import argparse
from pathlib import Path

from pydantic import ValidationError

from app.core.errors import InsightPilotError
from app.core.settings_base import PROJECT_ROOT
from data.seed.contracts import Parameters
from data.seed.files import export
from data.seed.generation import generate


def main() -> None:
    """Export one deterministic dataset or fail without replacing existing files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orders", type=int, default=50000)
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/seed/out")
    args = parser.parse_args()
    try:
        manifest = export(
            generate(Parameters(seed=args.seed, orders=args.orders, months=args.months)), args.out
        )
    except (InsightPilotError, ValidationError, OSError):
        parser.exit(
            1, "Seed generation failed; inspect parameters and use a new output directory.\n"
        )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
