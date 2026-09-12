"""SQL, provenance and assumptions must describe the same patched metric."""

# ruff: noqa: PLR2004 -- fixed precedence and confidence acceptance values.

from datetime import datetime
from uuid import uuid4

import pytest
import sqlglot
from pydantic import TypeAdapter, ValidationError
from sqlglot import exp

from app.agents.contracts import DataEvidence, MetricBinding, ResolvedMetricBinding
from app.core.errors import InvalidMetricPatch, InvalidMetricPatchError, MetricCatalogError
from app.schemas.metric_resolution import (
    BindingSource,
    MetricPatch,
    MetricPatchEntry,
    MetricPatches,
    RegionScope,
    SelectedOverrides,
)
from app.schemas.metrics import Grain
from app.services.metric_binding import build_binding, merge_explicit
from app.services.metric_patch_sql import canonical_filter, query_scopes
from tests.metric_resolution_support import OVERRIDE_ID, definition, override, request, schema


def test_binding_defaults_are_the_published_sql() -> None:
    for key in ("gmv", "aov", "order_count", "active_customer", "refund_rate", "refund_count"):
        result = build_binding(request(key), schema())
        assert sqlglot.parse_one(
            result.binding.resolved_expression, read="postgres"
        ) == sqlglot.parse_one(definition(key).examples[0].sql, read="postgres")
        assert result.binding.definition_version == definition(key).version
        assert all(
            source.source is BindingSource.COMPANY for source in result.binding.field_sources
        )


def test_date_patch_changes_only_metric_period_and_grain() -> None:
    value = request("refund_rate", patch=MetricPatch(date_field="o.paid_at"))
    value.grain = Grain.MONTH
    result = build_binding(value, schema()).binding
    query = sqlglot.parse_one(result.resolved_expression, read="postgres")
    numerator, denominator = query.ctes
    assert "r.requested_at" not in numerator.sql("postgres")
    assert "o.paid_at" in numerator.sql("postgres")
    assert "o.paid_at" in denominator.sql("postgres")
    assert "FULL OUTER JOIN" in result.resolved_expression
    assert "两侧独立聚合" in result.resolved_description


def test_common_filters_modify_both_refund_scopes() -> None:
    value = request(
        "refund_rate",
        patch=MetricPatch(
            remove_filters=["o.status != 'cancelled'"],
            add_filters=["o.gross_amount > 100"],
        ),
    )
    result = build_binding(value, schema()).binding
    query = sqlglot.parse_one(result.resolved_expression, read="postgres")
    for cte in query.ctes:
        assert "o.status <> 'cancelled'" not in cte.sql("postgres")
        assert "o.gross_amount > 100" in cte.sql("postgres")
    assert "r.status <> 'rejected'" in query.ctes[0].sql("postgres")
    assert "r.status" not in query.ctes[1].sql("postgres")


def test_filter_precedence_is_by_canonical_predicate() -> None:
    value = request(patch=MetricPatch(add_filters=["o.status != 'cancelled'"]))
    value.override = override(MetricPatch(remove_filters=["o.status <> 'cancelled'"]))
    result = build_binding(value, schema())
    assert canonical_filter("o.status <> 'cancelled'") in result.binding.filters_applied
    assert result.binding.override_id is None
    assert not any("自定义定义" in a for a in result.assumptions)


def test_conjunction_can_be_removed_and_explicit_remove_is_idempotent() -> None:
    value = request(patch=MetricPatch(remove_filters=["o.gross_amount > 100"]))
    value.override = override(MetricPatch(add_filters=["o.gross_amount > 100 AND o.promo_id = 17"]))
    result = build_binding(value, schema()).binding
    assert "o.gross_amount > 100" not in result.resolved_expression
    assert "o.promo_id = 17" in result.resolved_expression
    value = request(patch=MetricPatch(remove_filters=["o.status != 'cancelled'"]))
    value.override = override(MetricPatch(remove_filters=["o.status <> 'cancelled'"]))
    assert "o.status" not in build_binding(value, schema()).binding.resolved_expression


def test_expression_replaces_only_final_projection() -> None:
    result = build_binding(request(patch=MetricPatch(expression="SUM(o.gross_amount)")), schema())
    query = sqlglot.parse_one(result.binding.resolved_expression, read="postgres")
    assert query.expressions[1].alias == "gmv"
    assert query.expressions[1].this == sqlglot.parse_one("SUM(o.gross_amount)")
    assert "c.is_test_account = FALSE" in result.binding.resolved_expression
    assert "SUM(o.gross_amount)" in result.binding.resolved_description
    assert "不含运费" not in result.binding.resolved_description


def test_refund_expression_uses_existing_aggregate_outputs() -> None:
    patch = MetricPatch(
        expression="COALESCE(n.refunded_orders, 0) / NULLIF(d.paid_orders, 0) * 100"
    )
    query = sqlglot.parse_one(
        build_binding(request("refund_rate", patch=patch), schema()).binding.resolved_expression,
        read="postgres",
    )
    baseline = sqlglot.parse_one(definition("refund_rate").examples[0].sql, read="postgres")
    assert query.ctes == baseline.ctes
    assert query.expressions[1].alias == "refund_rate"


@pytest.mark.parametrize(
    "patch",
    [
        MetricPatch(date_field="r.requested_at"),
        MetricPatch(date_field="o.gross_amount"),
        MetricPatch(date_field="o.invented"),
        MetricPatch(add_filters=["n.paid_orders > 0"]),
        MetricPatch(add_filters=["o.unknown = 1"]),
        MetricPatch(add_filters=["paid_at IS NOT NULL"]),
        MetricPatch(add_filters=["o.order_id IN (SELECT order_id FROM biz.orders)"]),
        MetricPatch(add_filters=["pg_sleep(10) = 0"]),
        MetricPatch(add_filters=["o.order_id > 0; DELETE FROM biz.orders"]),
        MetricPatch(add_filters=["o.order_id > 0 -- comment"]),
        MetricPatch(add_filters=["1 + 2"]),
        MetricPatch(remove_filters=["o.order_id > 0"]),
        MetricPatch(add_filters=["o.promo_id = 17"], remove_filters=["o.promo_id = 17"]),
        MetricPatch(expression="SELECT 1"),
        MetricPatch(expression="pg_catalog.sum(o.gross_amount)"),
        MetricPatch(expression="SUM(o.gross_amount) OVER ()"),
        MetricPatch(expression="SUM(o.gross_amount) + o.order_id"),
        MetricPatch(expression="SUM(COUNT(o.order_id))"),
        MetricPatch(expression="SUM(i.quantity)"),
    ],
)
def test_invalid_patch_never_changes_definition(patch: MetricPatch) -> None:
    value = request(patch=patch)
    before = value.model_dump()
    with pytest.raises(InvalidMetricPatchError):
        build_binding(value, schema())
    assert value.model_dump() == before


def test_invalid_saved_patch_falls_back_atomically_and_is_explained() -> None:
    value = request()
    value.override = override(MetricPatch(date_field="o.created_at", expression="SUM(o.unknown)"))
    result = build_binding(value, schema())
    assert result.binding.date_field == "o.paid_at"
    assert result.binding.override_id is None
    assert any("保存的指标偏好" in assumption for assumption in result.assumptions)


@pytest.mark.parametrize("confidence", [0.89, 0.9])
def test_expression_patch_requires_high_confidence(confidence: float) -> None:
    value = request()
    value.override = override(MetricPatch(expression="SUM(o.gross_amount)"))
    value.override.confidence = confidence
    result = build_binding(value, schema())
    assert (result.binding.override_id == OVERRIDE_ID) == (confidence >= 0.9)


def test_foreign_user_override_never_applies() -> None:
    value = request()
    value.override = override(MetricPatch(date_field="o.created_at"))
    value.override.user_id = uuid4()
    assert build_binding(value, schema()).binding.override_id is None


def test_finalized_explicit_fields_beat_extraction() -> None:
    merged = merge_explicit(
        MetricPatch(date_field="o.created_at", remove_filters=["o.status <> 'cancelled'"]),
        MetricPatch(date_field="o.paid_at", add_filters=["o.status != 'cancelled'"]),
    )
    assert merged.date_field == "o.paid_at"
    assert merged.remove_filters == []
    assert merged.add_filters == [canonical_filter("o.status <> 'cancelled'")]


def test_region_ids_are_bound_as_literals_in_both_scopes() -> None:
    value = request("refund_rate")
    value.region_scope = RegionScope(region_ids=[3])
    result = build_binding(value, schema()).binding
    for cte in sqlglot.parse_one(result.resolved_expression).ctes:
        assert "o.region_id IN (3)" in cte.sql()
    assert result.region_scope == value.region_scope


def test_contracts_reject_duplicates_and_naive_preferences() -> None:
    saved = override(MetricPatch())
    with pytest.raises(ValidationError):
        SelectedOverrides(items=[saved, saved])
    entry = MetricPatchEntry(metric_key="gmv", patch=MetricPatch())
    with pytest.raises(ValidationError):
        MetricPatches(items=[entry, entry])
    with pytest.raises(ValidationError):
        type(saved).model_validate({**saved.model_dump(), "created_at": datetime(2026, 1, 1)})


def test_historical_binding_and_new_binding_are_readable() -> None:
    resolved = build_binding(request(), schema()).binding
    legacy = MetricBinding.model_validate(
        {
            key: value
            for key, value in resolved.model_dump().items()
            if key in MetricBinding.model_fields
        }
    )
    reader = TypeAdapter(DataEvidence.model_fields["metric_bindings"].annotation)
    old, new = reader.validate_json(
        "[" + legacy.model_dump_json() + "," + resolved.model_dump_json() + "]"
    )
    assert type(old) is MetricBinding
    assert type(new) is ResolvedMetricBinding
    assert new.resolved_expression == resolved.resolved_expression
    assert len(sqlglot.parse(new.resolved_expression)) == 1
    assert isinstance(sqlglot.parse_one(new.resolved_expression), exp.Select)


@pytest.mark.parametrize("ids", [[], [3]])
def test_finalized_region_beats_saved_filter(ids: list[int]) -> None:
    value = request()
    value.override = override(MetricPatch(add_filters=["o.region_id = 2"]))
    value.region_scope = RegionScope(region_ids=ids)
    result = build_binding(value, schema()).binding
    assert "o.region_id = 2" not in result.resolved_expression
    assert ("o.region_id IN (3)" in result.resolved_expression) == bool(ids)
    assert result.override_id is None


def test_derived_select_columns_are_available() -> None:
    query = sqlglot.parse_one("WITH n AS (SELECT 1 AS value) SELECT n.value FROM n")
    assert isinstance(query, exp.Select)
    assert query_scopes(query, schema())[-1].columns == {"n.value": None}


def test_unsupported_derived_union_fails_with_catalog_error() -> None:
    query = sqlglot.parse_one(
        "WITH n AS (SELECT 1 AS value UNION ALL SELECT 2 AS value) SELECT n.value FROM n"
    )
    assert isinstance(query, exp.Select)
    with pytest.raises(MetricCatalogError):
        query_scopes(query, schema())


def test_invalid_patch_legacy_alias_preserves_identity_and_code() -> None:
    assert InvalidMetricPatch is InvalidMetricPatchError
    assert InvalidMetricPatchError.code == "INVALID_METRIC_PATCH"
