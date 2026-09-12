"""Detached catalog and fixed-clock fixtures shared by Step 2.6 tests."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.agents.runtime import RuntimeContext
from app.core.deadline import Deadline
from app.core.errors import MetricNotFound
from app.schemas.metric_resolution import MetricIntent, MetricPatch, SelectedMetricOverride
from app.schemas.metrics import Grain, MetricDefinition
from app.schemas.schema_catalog import SchemaCatalog
from app.services.metric_binding import BindingRequest
from app.services.periods import BUSINESS_TZ, resolve_period
from data.seed.metrics_loader import load_catalog as load_metrics
from data.seed.schema_metadata_loader import load_catalog as load_schema
from tests.agents.support import context

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 8, tzinfo=BUSINESS_TZ)
USER_ID = UUID("00000000-0000-0000-0000-000000000001")
OVERRIDE_ID = UUID("00000000-0000-0000-0000-000000000002")


def definition(key: str = "gmv") -> MetricDefinition:
    return next(
        d for d in load_metrics(ROOT / "data/seed/metrics.yaml").definitions if d.key == key
    )


def schema() -> SchemaCatalog:
    return load_schema(ROOT / "data/seed/schema_metadata.yaml")


def request(key: str = "gmv", *, patch: MetricPatch | None = None) -> BindingRequest:
    return BindingRequest(
        definition=definition(key),
        period=resolve_period("2026年8月", now=NOW),
        grain=Grain.TOTAL,
        user_id=USER_ID,
        explicit_patch=patch or MetricPatch(),
    )


def override(patch: MetricPatch, *, key: str = "gmv") -> SelectedMetricOverride:
    return SelectedMetricOverride(
        metric_key=key,
        patch=patch,
        id=OVERRIDE_ID,
        user_id=USER_ID,
        created_at=datetime(2026, 7, 14, tzinfo=BUSINESS_TZ),
        confidence=0.95,
    )


class Catalog:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Deadline | None]] = []

    async def list_active(self, *, deadline: Deadline | None = None) -> list[MetricDefinition]:
        self.calls.append(("list", deadline))
        return load_metrics(ROOT / "data/seed/metrics.yaml").definitions

    async def get_active(self, key: str, *, deadline: Deadline | None = None) -> MetricDefinition:
        self.calls.append((key, deadline))
        for item in load_metrics(ROOT / "data/seed/metrics.yaml").definitions:
            if item.key == key:
                return item
        raise MetricNotFound()


class Schema:
    async def snapshot(self, *, deadline: Deadline | None = None) -> SchemaCatalog:
        return schema()

    async def render(
        self, tables: list[str] | None = None, *, deadline: Deadline | None = None
    ) -> str:
        return "unused"


def runtime(intent: MetricIntent) -> RuntimeContext:
    ctx = context(responses=[intent])
    return replace(
        ctx,
        metrics=Catalog(),
        schema_catalog=Schema(),
        now=NOW,
        identity=ctx.identity.model_copy(update={"user_id": USER_ID}),
    )
