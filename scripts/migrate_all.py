"""Project-anchored migration commands; configuration never enters a shell argument."""

import argparse
import asyncio

from alembic.config import Config
from pydantic import ValidationError

from alembic import command
from app.core.settings_base import PROJECT_ROOT
from scripts.migration_environment import MigrationError
from scripts.migration_settings import MigrationSettings, MigrationTarget
from scripts.setup_checkpointer import setup_checkpointer


def migration_config(target: MigrationTarget) -> Config:
    """Resolve the named Alembic environment independently of the caller's cwd."""
    return Config(str(PROJECT_ROOT / "alembic.ini"), ini_section=target.value)


def main() -> None:
    """Upgrade both histories, or generate one explicitly selected revision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upgrade", "revision"), nargs="?", default="upgrade")
    parser.add_argument("--database", choices=list(MigrationTarget), default=MigrationTarget.APP)
    parser.add_argument("--message")
    args = parser.parse_args()
    if args.action == "revision" and not args.message:
        parser.error("revision requires a nonempty --message")
    try:
        if args.action == "upgrade":
            for target in MigrationTarget:
                command.upgrade(migration_config(target), "head")
            asyncio.run(setup_checkpointer(MigrationSettings.load()))
        else:
            command.revision(
                migration_config(MigrationTarget(args.database)),
                message=args.message,
                autogenerate=True,
            )
    except (MigrationError, ValidationError):
        # Avoid printing validation input or chained driver exceptions containing credentials.
        parser.exit(1, "Migration failed; check operator configuration and database state.\n")


if __name__ == "__main__":
    main()
