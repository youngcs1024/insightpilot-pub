"""Print the PostgreSQL-backed metric prompt block using API-role configuration."""

import asyncio
import sys

from app.core.config_models import Settings
from app.core.errors import InsightPilotError
from app.core.logging import setup_logging
from app.db.session import Database
from app.services.metrics import MetricService, render_catalog_block


async def render() -> str:
    """Read global application data without accessing business credentials."""
    settings = Settings.load()
    settings.observability.log_level = "WARNING"
    setup_logging(settings)
    database = Database(settings.database)
    database.start()
    try:
        service = MetricService(database, settings.database)
        await service.validate_startup()
        return render_catalog_block(await service.list_active())
    finally:
        await database.aclose()


def main() -> int:
    """Return nonzero with a safe diagnostic when the catalog is unavailable."""
    try:
        print(asyncio.run(render()), end="")
        return 0
    except InsightPilotError as exc:
        print(exc.code + ": " + exc.user_message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
