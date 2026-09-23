"""Native calendar tool backed by the application's Shanghai period policy."""

from datetime import datetime

from app.services.periods import Period
from app.services.periods import resolve_period as _resolve_period


def resolve_period(
    expression: str, *, now: datetime, reference_period: Period | None = None
) -> Period:
    """Resolve one period using the injected instant, never the host clock."""
    return _resolve_period(expression, now=now, reference_period=reference_period)
