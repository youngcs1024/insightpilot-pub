"""Deterministic export and corruption detection without a database."""

from datetime import timedelta
from pathlib import Path
from random import Random
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from data.seed.contracts import Parameters, SeedConflictError, SeedError
from data.seed.files import export, read
from data.seed.generation import generate, month_start, shopping_time
from data.seed.rows import RegionsRow


def test_seed_deterministic(tmp_path: Path) -> None:
    params = Parameters(orders=100, months=1)
    first = export(generate(params), tmp_path / "first")
    second = export(generate(params), tmp_path / "second")
    assert first == second
    for name in ("orders.jsonl", "order_items.jsonl", "manifest.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert export(generate(params), tmp_path / "first") == first
    different = export(generate(Parameters(seed=5, orders=100, months=1)), tmp_path / "different")
    assert first.files[4].sha256 != different.files[4].sha256
    with pytest.raises(SeedConflictError):
        export(generate(Parameters(seed=5, orders=100, months=1)), tmp_path / "first")


@pytest.mark.parametrize(("orders", "months"), [(1, 1), (10, 2), (100, 18), (100, 120)])
def test_custom_scale(orders: int, months: int, tmp_path: Path) -> None:
    dataset = generate(Parameters(orders=orders, months=months))
    export(dataset, tmp_path / "seed")
    manifest, loaded = read(tmp_path / "seed")
    assert len(loaded.tables[4]) == orders
    assert manifest.parameters.months == months
    assert month_start(0).isoformat() == "2026-08-31T16:00:00+00:00"


@pytest.mark.parametrize("values", [{"orders": 0}, {"months": 0}, {"months": 121}, {"seed": -1}])
def test_invalid_parameters(values: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        Parameters(**values)


def test_corrupt_export_rejected(tmp_path: Path) -> None:
    export(generate(Parameters(orders=10, months=1)), tmp_path / "seed")
    (tmp_path / "seed" / "orders.jsonl").write_text("{}\n")
    with pytest.raises(SeedError):
        read(tmp_path / "seed")


def test_naive_timestamp_rejected() -> None:

    with pytest.raises(ValidationError):
        RegionsRow(
            region_id=1,
            name="a",
            name_en="a",
            renamed_from=None,
            effective_from="2026-01-01T00:00:00",
        )


def test_export_sorts_primary_keys(tmp_path: Path) -> None:
    dataset = generate(Parameters(orders=10, months=1))
    original = export(dataset, tmp_path / "ordered")
    for rows in dataset.tables:
        rows.reverse()
    assert export(dataset, tmp_path / "reversed") == original


def test_money_inconsistency_fails_before_export(tmp_path: Path) -> None:
    dataset = generate(Parameters(orders=10, months=1))
    dataset.tables[4][0] = dataset.tables[4][0].model_copy(update={"gross_amount": 0})
    with pytest.raises(SeedError, match="totals differ"):
        export(dataset, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_weekend_weights_use_shanghai_calendar() -> None:
    rng = Random(42)  # noqa: S311 -- synthetic generation regression.
    start = month_start(-18)  # March 1, 2025 is Saturday in Shanghai, Friday in UTC.
    with patch.object(rng, "choices", wraps=rng.choices) as choices:
        shopping_time(rng, start, start + timedelta(days=7))
    assert choices.call_args_list[0].kwargs["weights"] == [115, 115, 100, 100, 100, 100, 100]
