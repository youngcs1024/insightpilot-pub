"""Real SQL gates for every declared slice, plus atomic ETL and reproducibility."""

# Numeric expectations are independent, frozen Step 0.7 acceptance quotas.
# ruff: noqa: PLR2004

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from data.seed.contracts import Parameters, SeedConflictError
from data.seed.files import export
from data.seed.generation import generate
from scripts.seed import import_seed
from scripts.seed_settings import SeedSettings
from tests.seed_cases import SeedTrap, load_cases
from tests.seed_support import seed_database_stack, seed_directory, seeded

pytestmark = pytest.mark.integration
__all__ = ["seed_directory", "seeded"]
seed_stack = seed_database_stack
CASES = load_cases().traps


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
async def test_declared_trap(seeded: SeedSettings, seed_directory: Path, case: SeedTrap) -> None:
    trap, naive_sql, correct_sql = case.id, case.naive_sql, case.correct_sql
    engine = create_async_engine(seeded.seed.url)
    try:
        async with engine.connect() as connection:
            naive = (await connection.execute(text(naive_sql))).all()
            correct = (await connection.execute(text(correct_sql))).all()
        n = Decimal(naive[0][-1]) if trap == "T7" else Decimal(naive[0][0])
        c = Decimal(correct[1][-1]) if trap == "T7" else Decimal(correct[0][0])
        assert c > 0
        delta = abs(n - c) / abs(c)
        low, high = case.lower_delta, case.upper_delta
        assert low <= delta <= high, (trap, naive, correct, delta)
        if trap == "T7":
            assert tuple(naive[0][:2]) == (1600, 230)
            assert tuple(correct[0][:3]) == (False, 600, 30)
            assert tuple(correct[1][:3]) == (True, 1000, 200)
        if trap == "T3":
            assert (n, c) == (92, 100)
        if trap == "T5":
            assert (n, c) == (1800, 1600)
        if trap == "T8":
            assert (n, c) == (440, 400)
        (seed_directory.parent / f"{trap}.json").write_text(
            json.dumps(
                {
                    "trap": trap,
                    "naive": [list(row) for row in naive],
                    "correct": [list(row) for row in correct],
                    "relative_delta": delta,
                },
                default=str,
                indent=2,
            )
            + "\n"
        )
    finally:
        await engine.dispose()


async def test_all_null_companion(seeded: SeedSettings) -> None:
    case = next(case for case in CASES if case.id == "T6")
    naive, correct = case.naive_sql, case.correct_sql
    engine = create_async_engine(seeded.seed.url)
    try:
        async with engine.connect() as connection:
            assert (
                await connection.scalar(
                    text(naive.rstrip().rstrip(";") + " AND i.item_discount IS NULL")
                )
                is None
            )
            assert (
                await connection.scalar(
                    text(correct.rstrip().rstrip(";") + " AND i.item_discount IS NULL")
                )
                > 0
            )
    finally:
        await engine.dispose()


async def test_seed_replay_and_conflict(
    seeded: SeedSettings, seed_directory: Path, tmp_path: Path
) -> None:
    results = await asyncio.gather(
        import_seed(seed_directory, seeded), import_seed(seed_directory, seeded)
    )
    assert [r.outcome for r in results] == ["unchanged", "unchanged"]
    other = tmp_path / "other"
    export(generate(Parameters(seed=7, orders=100, months=1)), other)
    with pytest.raises(SeedConflictError):
        await import_seed(other, seeded)


async def test_row_counts_and_refund_observation(
    seeded: SeedSettings, seed_directory: Path
) -> None:
    engine = create_async_engine(seeded.seed.url)
    try:
        async with engine.connect() as connection:
            status = (
                await connection.execute(
                    text("SELECT status,count(*) FROM biz.orders GROUP BY 1 ORDER BY 2 DESC")
                )
            ).all()
            assert dict(status) == {
                "delivered": 38000,
                "shipped": 4000,
                "cancelled": 3000,
                "closed": 2500,
                "paid": 2000,
                "created": 500,
            }
            cross = await connection.scalar(
                text("""SELECT avg((date_trunc('month',r.requested_at AT TIME ZONE 'Asia/Shanghai')<>
                date_trunc('month',o.paid_at AT TIME ZONE 'Asia/Shanghai'))::int)
                FROM biz.refunds r JOIN biz.orders o USING(order_id)""")
            )
            assert Decimal(".18") <= cross <= Decimal(".30")
            assert await connection.scalar(text("SELECT count(*) FROM biz.refunds")) == 3400
            assert await connection.scalar(text("SELECT count(*) FROM biz.order_items")) in range(
                135000, 145001
            )
            assert (
                await connection.scalar(
                    text(
                        "SELECT count(*) FROM biz.orders WHERE status='cancelled' AND paid_at >= TIMESTAMPTZ '2026-08-01 00:00+08'"
                    )
                )
                == 600
            )
            assert (
                await connection.scalar(
                    text("SELECT count(*) FROM biz.customers WHERE is_test_account")
                )
                == 200
            )
            null_share = await connection.scalar(
                text("""SELECT avg((i.item_discount IS NULL)::int)
                FROM biz.order_items i JOIN biz.orders o USING(order_id)
                WHERE o.created_at<TIMESTAMPTZ '2026-01-01 00:00+08'""")
            )
            assert Decimal(".28") <= null_share <= Decimal(".32")
            assert await connection.scalar(text("SELECT 1::numeric/NULLIF(0,0)")) is None
            assert await connection.scalar(
                text("SELECT count(*)>count(DISTINCT order_id) FROM biz.refunds")
            )
            assert not await connection.scalar(
                text("SELECT has_schema_privilege('mcp_ro','ops','USAGE')")
            )
            assert not await connection.scalar(
                text("SELECT has_schema_privilege('app_rw','ops','USAGE')")
            )
        (seed_directory.parent / "distribution.json").write_text(
            json.dumps(
                {
                    "statuses": dict(status),
                    "cross_month_share": str(cross),
                    "historical_null_share": str(null_share),
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        await engine.dispose()


def test_baseline_seed_deterministic(seed_directory: Path, tmp_path: Path) -> None:
    second = tmp_path / "second"
    export(generate(Parameters()), second)
    assert (second / "manifest.json").read_bytes() == (
        seed_directory / "manifest.json"
    ).read_bytes()


async def test_reference_money_time_and_identity_sql(seeded: SeedSettings) -> None:
    engine = create_async_engine(seeded.seed.url)
    try:
        async with engine.connect() as connection:
            assert (
                await connection.scalar(
                    text("""SELECT count(*) FROM biz.orders o JOIN (
                SELECT order_id,sum(quantity*unit_price) gross,
                sum(COALESCE(item_discount,0)) discount FROM biz.order_items GROUP BY order_id
                ) i USING(order_id) WHERE o.gross_amount<>i.gross OR o.discount_amount<>i.discount""")
                )
                == 0
            )
            assert (
                await connection.scalar(
                    text("""SELECT count(*) FROM (
                SELECT phone,count(*) n FROM biz.customers GROUP BY phone HAVING count(*)>1
                ) d WHERE n=2""")
                )
                == 40
            )
            assert (
                await connection.scalar(
                    text("""SELECT count(*) FROM biz.orders o JOIN biz.promotions p USING(promo_id)
                WHERE o.paid_at<p.starts_at OR o.paid_at>=p.ends_at""")
                )
                == 0
            )
            assert (
                await connection.scalar(
                    text("""SELECT count(*) FROM biz.refunds r JOIN biz.orders o USING(order_id)
                WHERE r.requested_at<o.paid_at OR r.completed_at<r.requested_at
                OR r.requested_at>=TIMESTAMPTZ '2026-12-15 00:00+08'""")
                )
                == 0
            )
            for period in ("2026-04-01", "2026-06-01"):
                aliases = (
                    await connection.execute(
                        text("""SELECT count(*) FILTER(WHERE r.name='华东一区'),
                    count(*) FILTER(WHERE r.renamed_from='华东') FROM biz.orders o JOIN biz.regions r USING(region_id)
                    WHERE o.paid_at>=CAST(:period AS timestamptz) AND o.paid_at<CAST(:period AS timestamptz)+interval '1 month'"""),
                        {"period": datetime.fromisoformat(period + "T00:00:00+08:00")},
                    )
                ).one()
                assert aliases[0] == aliases[1] > 0
            # The actual dataset is bounded, including the exact exclusive endpoint.
            assert (
                await connection.scalar(
                    text(
                        "SELECT count(*) FROM biz.orders WHERE paid_at>=TIMESTAMPTZ '2026-09-01 00:00+08'"
                    )
                )
                == 0
            )
    finally:
        await engine.dispose()


async def test_refund_and_boundary_semantics(seeded: SeedSettings) -> None:
    engine = create_async_engine(seeded.seed.url)
    try:
        async with engine.connect() as connection:
            refund_sql = next(case.correct_sql for case in CASES if case.id == "T2")
            assert (
                await connection.scalar(
                    text(
                        refund_sql.replace("2026-08-01", "2027-08-01").replace(
                            "2026-09-01", "2027-09-01"
                        )
                    )
                )
                is None
            )
            boundary = (
                await connection.execute(
                    text("""SELECT
                count(*) FILTER(WHERE o.created_at<TIMESTAMPTZ '2026-08-01 08:00+08'),
                count(*) FILTER(WHERE o.created_at>=TIMESTAMPTZ '2026-08-01 08:00+08'
                    AND o.created_at<TIMESTAMPTZ '2026-08-02 00:00+08'),
                count(*) FILTER(WHERE o.created_at>=TIMESTAMPTZ '2026-08-02 00:00+08'),
                count(DISTINCT o.region_id)
                FROM biz.orders o JOIN biz.customers c USING(customer_id)
                WHERE NOT c.is_test_account AND o.status<>'cancelled'
                AND o.created_at>=TIMESTAMPTZ '2026-08-01 00:00+08'
                AND o.created_at<TIMESTAMPTZ '2026-08-02 08:00+08'""")
                )
            ).one()
            assert tuple(boundary) == (12, 88, 4, 5)
            # Exact start is included and exact end is excluded by the same predicate used in trap SQL.
            assert (
                await connection.scalar(
                    text("""SELECT count(*) FROM (VALUES
                (TIMESTAMPTZ '2026-08-01 00:00+08'),(TIMESTAMPTZ '2026-09-01 00:00+08')) v(t)
                WHERE t>=TIMESTAMPTZ '2026-08-01 00:00+08' AND t<TIMESTAMPTZ '2026-09-01 00:00+08'""")
                )
                == 1
            )
            assert (
                await connection.scalar(
                    text("""SELECT count(*) FROM (
                SELECT r.order_id,sum(r.amount) amount FROM biz.refunds r GROUP BY r.order_id
                ) r JOIN biz.orders o USING(order_id) WHERE r.amount>o.gross_amount-o.discount_amount""")
                )
                == 0
            )
            assert await connection.scalar(
                text("""SELECT count(DISTINCT o.order_id)=
                (SELECT count(*) FROM biz.orders e WHERE EXISTS(SELECT 1 FROM biz.refunds r
                WHERE r.order_id=e.order_id AND r.status<>'rejected'))
                FROM biz.orders o JOIN biz.refunds r USING(order_id) WHERE r.status<>'rejected'""")
            )
    finally:
        await engine.dispose()
