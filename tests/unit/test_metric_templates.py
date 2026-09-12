"""Catalog authoring, controlled rendering and typed failure contracts without I/O."""

# ruff: noqa: PLR2004 -- fixed six-metric acceptance contract.

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config_models import Settings
from app.core.errors import (
    DeadlineExceededError,
    MetricCatalogError,
    MetricNotFound,
    UnsupportedGrain,
    UpstreamUnavailableError,
)
from app.db.session import Database
from app.schemas.metrics import (
    Grain,
    MetricCatalog,
    MetricDefinition,
    MetricRenderContext,
    example_period,
)
from app.services.metric_templates import (
    render_catalog_block,
    render_expression,
    validate_definitions,
    validate_template,
)
from app.services.metrics import MetricService
from data.seed.metrics_loader import load_catalog
from scripts import render_metric_catalog

ROOT = Path(__file__).resolve().parents[2]
AUTHORING = ROOT / "data/seed/metrics.yaml"
SNAPSHOT = ROOT / "alembic/app/data/0007_metric_catalog.json"


def definitions() -> list[MetricDefinition]:
    return load_catalog(AUTHORING).definitions


def test_yaml_snapshot_and_example_rendering_are_bound() -> None:
    catalog = load_catalog(AUTHORING)
    assert catalog == MetricCatalog.model_validate_json(SNAPSHOT.read_text())
    assert len(catalog.definitions) == 6
    start, end = example_period()
    for item in catalog.definitions:
        grains = [Grain.TOTAL, Grain.MONTH if item.key.startswith("refund_") else Grain.CATEGORY]
        for example, grain in zip(item.examples, grains, strict=True):
            assert example.sql == render_expression(
                item, MetricRenderContext(period_start=start, period_end=end, grain=grain)
            )
    block = render_catalog_block(catalog.definitions)
    assert block == render_catalog_block(list(reversed(catalog.definitions)))
    for item in catalog.definitions:
        assert item.description in block
        assert item.expression_template.strip() in block
        assert f"Metric: {item.key} v1" in block
        assert all(e.sql in block for e in item.examples)


@pytest.mark.parametrize(
    "mutation", ["extra", "duplicate", "recursive", "empty", "missing", "type", "table", "grains"]
)
def test_invalid_authoring(tmp_path: Path, mutation: str) -> None:
    data = json.loads(SNAPSHOT.read_text())
    if mutation == "extra":
        data["definitions"][0]["unknown"] = True
    elif mutation == "missing":
        data["definitions"].pop()
    elif mutation == "type":
        data["definitions"][0]["version"] = "1"
    elif mutation == "table":
        data["definitions"][0]["base_tables"] = ["pg_catalog.pg_user"]
    elif mutation == "grains":
        data["definitions"][0]["supported_grains"] = ["total", "total"]
    source = yaml.safe_dump(data)
    if mutation == "duplicate":
        source += "schema_version: 1\n"
    elif mutation == "recursive":
        source = "definitions: &a [*a]"
    elif mutation == "empty":
        source = ""
    path = tmp_path / "invalid.yaml"
    path.write_text(source)
    with pytest.raises(MetricCatalogError):
        load_catalog(path)


@pytest.mark.parametrize(
    "template",
    [
        "{{ absent }}",
        "{{ period_start.upper() }}",
        "{% for x in grain %}{{ x }}{% endfor %}",
        "{% invalid %}",
        "{{ grain.__class__ }}",
    ],
)
def test_unknown_slots_and_executable_jinja_are_rejected(template: str) -> None:
    item = definitions()[0].model_copy(update={"expression_template": template})
    with pytest.raises(MetricCatalogError):
        validate_template(item)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "inactive"])
def test_startup_rejects_invalid_active_catalog(mutation: str) -> None:
    items = definitions()
    if mutation == "missing":
        items.pop()
    elif mutation == "duplicate":
        items.append(items[0])
    else:
        items[0].is_active = False
    with pytest.raises(MetricCatalogError):
        validate_definitions(items)


def test_unsupported_grain_lists_supported_values() -> None:
    item = next(d for d in definitions() if d.key == "refund_rate")
    start, end = example_period()
    with pytest.raises(UnsupportedGrain) as caught:
        render_expression(
            item, MetricRenderContext(period_start=start, period_end=end, grain=Grain.CATEGORY)
        )
    assert caught.value.supported == ("total", "day", "week", "month", "region")


@pytest.mark.parametrize(
    ("start", "end", "grain"),
    [
        ("2026-08-01", "2026-09-01", "total"),
        ("2026-09-01T00:00:00+08:00", "2026-08-01T00:00:00+08:00", "total"),
        ("2026-08-01T00:00:00+08:00", "2026-09-01T00:00:00+08:00", "total'; SELECT 1"),
    ],
)
def test_untrusted_context_rejected(start: str, end: str, grain: str) -> None:
    with pytest.raises(ValidationError):
        MetricRenderContext.model_validate(
            {"period_start": start, "period_end": end, "grain": grain}
        )


def test_utc_instants_render_as_shanghai_and_all_grains_render() -> None:
    context = MetricRenderContext.model_validate(
        {"period_start": "2026-07-31T16:00:00Z", "period_end": "2026-08-31T16:00:00Z"}
    )
    for item in definitions():
        for grain in item.supported_grains:
            sql = render_expression(item, context.model_copy(update={"grain": grain}))
            assert "2026-08-01T00:00:00+08:00" in sql
            assert "2026-09-01T00:00:00+08:00" in sql
            assert "{{" not in sql


async def test_service_startup_and_public_read_methods(settings: Settings) -> None:
    item = definitions()[0]
    service = MetricService(Database(settings.database), settings.database)
    with patch.object(service, "_read", AsyncMock(return_value=[item])):
        assert await service.get_active(item.key) == item
        assert await service.get_version(item.key, 1) == item
        assert await service.list_active() == [item]
    with patch.object(service, "list_active", AsyncMock(return_value=definitions())):
        await service.validate_startup()
    with (
        patch.object(service, "list_active", AsyncMock(return_value=[])),
        pytest.raises(MetricCatalogError),
    ):
        await service.validate_startup()


def test_cli_safe_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        render_metric_catalog, "render", AsyncMock(side_effect=UpstreamUnavailableError("SECRET"))
    )
    assert render_metric_catalog.main() == 1
    assert "SECRET" not in capsys.readouterr().err


@pytest.mark.parametrize("retryable", [True, False])
async def test_only_typed_transient_reads_retry(settings: Settings, retryable: bool) -> None:

    attempts = 0
    session = AsyncMock(spec=AsyncSession)
    database = MagicMock(spec=Database)
    item = definitions()[0]

    @asynccontextmanager
    async def session_scope() -> AsyncIterator[AsyncSession]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if retryable:
                raise UpstreamUnavailableError()
            raise MetricNotFound()
        yield session

    database.session.side_effect = session_scope
    service = MetricService(database, settings.database)
    repository = AsyncMock()
    repository.get.return_value = item
    with patch("app.services.metrics.MetricRepository", return_value=repository):
        if retryable:
            assert await service.get_active("gmv") == item
            assert attempts == 2
        else:
            with pytest.raises(MetricNotFound):
                await service.get_active("gmv")
            assert attempts == 1


async def test_read_deadline_cancels_and_closes_session(settings: Settings) -> None:

    closed = False
    database = MagicMock(spec=Database)

    @asynccontextmanager
    async def session_scope() -> AsyncIterator[None]:
        nonlocal closed
        try:
            await asyncio.Event().wait()
            yield
        finally:
            closed = True

    database.session.side_effect = session_scope
    service = MetricService(
        database, settings.database.model_copy(update={"command_timeout_s": 0.01})
    )
    with pytest.raises(DeadlineExceededError):
        await service.list_active()
    assert closed
