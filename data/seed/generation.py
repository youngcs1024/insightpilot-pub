"""Constrained allocation followed by deterministic sampling, with exact money."""

# Fixed numeric quotas and IDs below are the executable Step 0.7 design.
# ruff: noqa: PLR2004

from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from random import Random
from zoneinfo import ZoneInfo

from data.seed.contracts import END, Dataset, Parameters, SeedError
from data.seed.rows import (
    CustomersRow,
    InventoryRow,
    OrderItemsRow,
    OrdersRow,
    ProductsRow,
    PromotionsRow,
    RefundsRow,
    RegionsRow,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
MONTH_COUNTS = (
    2200,
    2250,
    2300,
    2350,
    2400,
    2450,
    2500,
    2550,
    2900,
    3300,
    2600,
    2550,
    2700,
    2750,
    2800,
    2900,
    3500,
    5000,
)
CENT = Decimal("0.01")


def money(cents: int) -> Decimal:
    """Encode integer cents without binary float arithmetic."""
    return (Decimal(cents) / 100).quantize(CENT)


def month_start(offset: int) -> datetime:
    """Offset from September 2026, always at local midnight."""
    year, month = divmod(2026 * 12 + 8 + offset, 12)
    return datetime(year, month + 1, 1, tzinfo=SHANGHAI).astimezone(UTC)


def allocate(total: int, weights: tuple[int, ...] | list[int]) -> list[int]:
    """Largest remainder allocation with stable tie-breaking."""
    denominator = sum(weights)
    values = [total * weight // denominator for weight in weights]
    # Descending fractional remainders, stable index ties.
    priority = sorted(range(len(weights)), key=lambda i: (-(total * weights[i] % denominator), i))
    for index in priority[: total - sum(values)]:
        values[index] += 1
    return values


def dimensions(
    rng: Random, start: datetime
) -> tuple[list[RegionsRow], list[PromotionsRow], list[ProductsRow], list[CustomersRow]]:
    """Generate fixed dimension populations before transactions."""
    regions = [
        RegionsRow(
            region_id=i,
            name=name,
            name_en=en,
            renamed_from="华东" if i == 3 else None,
            effective_from=month_start(-4) if i == 3 else month_start(-18),
        )
        for i, (name, en) in enumerate(
            zip(
                ("华北", "华南", "华东一区", "华中", "西部"),
                ("North China", "South China", "East China", "Central China", "West China"),
                strict=True,
            ),
            1,
        )
    ]
    promotions = []
    for i in range(1, 41):
        duration = rng.randint(7, min(30, (END - start).days))
        begins = start + timedelta(days=rng.randint(0, (END - start).days - duration))
        promotions.append(
            PromotionsRow(
                promo_id=i,
                name=f"活动-{i:02d}",
                kind=("discount", "seasonal", "shipping")[(i - 1) % 3],
                starts_at=begins,
                ends_at=begins + timedelta(days=duration),
                rule_doc_ref=None,
            )
        )
    promotions[16] = PromotionsRow(
        promo_id=17,
        name="2026夏季活动",
        kind="seasonal",
        starts_at=month_start(-1) + timedelta(days=4),
        ends_at=month_start(-1) + timedelta(days=20),
        rule_doc_ref="promo_2026_summer.md",
    )
    products = []
    for i in range(1, 1201):
        price = rng.randint(2000, 100000)
        cost = (price * rng.randint(40, 75) + 50) // 100
        products.append(
            ProductsRow(
                product_id=i,
                sku=f"ZH-{i:06d}",
                category=("apparel", "beauty", "home", "electronics", "food")[(i - 1) % 5],
                list_price=money(price),
                cost=money(cost),
                launched_at=start - timedelta(days=rng.randint(30, 365)),
            )
        )
    customers = []
    for i in range(1, 8001):
        region = rng.choices(range(1, 6), weights=(18, 20, 30, 17, 15))[0]
        channel = ("organic", "search", "social", "affiliate")[(i - 1) % 4]
        if i <= 440:
            region, channel = 3, "affiliate"
        if i > 7800:
            region = 2 if i <= 7900 else 3
        identity = i - 40 if 41 <= i <= 80 else i
        customers.append(
            CustomersRow(
                customer_id=i,
                region_id=region,
                channel=channel,
                registered_at=start,
                is_test_account=i > 7800,
                phone=f"IP{identity:010d}",
                email=f"customer-{i}@example.invalid",
            )
        )
    return regions, promotions, products, customers


def shopping_time(rng: Random, start: datetime, end: datetime) -> datetime:
    """Weekend-weighted calendar days and the declared Chinese hour distribution."""
    days = list(range((end - start).days))
    day = rng.choices(
        days,
        weights=[
            115 if (start + timedelta(days=d)).astimezone(SHANGHAI).weekday() >= 5 else 100
            for d in days
        ],
    )[0]
    hours = rng.choices(
        (tuple(range(19, 24)), (11, 12, 13), (8, 9, 10, 14, 15, 16, 17, 18), tuple(range(8))),
        weights=(45, 25, 25, 5),
    )[0]
    return start + timedelta(days=day, hours=rng.choice(hours), seconds=rng.randrange(3600))


def boundary_cell(value: datetime) -> bool:
    """Cells reserved by T3, including the next local day's early hours."""
    return month_start(-1) <= value < month_start(-1) + timedelta(days=1, hours=8)


def line_plan(
    rng: Random, order_id: int, products: list[ProductsRow], old: bool, item_offset: int
) -> list[OrderItemsRow]:
    """Reserve lines first; their totals determine the order."""
    count = rng.choices((1, 2, 3, 4, 5), weights=(15, 25, 30, 25, 5))[0]
    rows: list[OrderItemsRow] = []
    for product in sorted(rng.sample(products, count), key=lambda p: p.product_id):
        quantity = rng.choices((1, 2, 3), weights=(70, 25, 5))[0]
        gross = int(product.list_price * 100) * quantity
        discount = money((gross * rng.randint(0, 20) + 50) // 100)
        rows.append(
            OrderItemsRow(
                order_item_id=item_offset + len(rows) + 1,
                order_id=order_id,
                product_id=product.product_id,
                quantity=quantity,
                unit_price=product.list_price,
                item_discount=None if old and rng.randrange(10) < 3 else discount,
            )
        )
    return rows


def status_plan(rng: Random, counts: list[int], baseline: bool) -> list[list[str]]:
    """Allocate exact global statuses, including August's cancellation increase."""
    totals = allocate(sum(counts), [500, 2000, 4000, 38000, 3000, 2500])
    names = ("created", "paid", "shipped", "delivered", "cancelled", "closed")
    result: list[list[str]] = [[] for _ in counts]
    remaining = counts.copy()
    for name, total in zip(names, totals, strict=True):
        if name == "delivered":
            continue
        values = (
            [*allocate(2400, counts[:-1]), 600]
            if baseline and name == "cancelled"
            else allocate(total, counts if baseline else remaining)
        )
        for i, n in enumerate(values):
            result[i].extend([name] * n)
            remaining[i] -= n
    for i, n in enumerate(remaining):
        if n < 0:
            raise SeedError("Requested scale cannot allocate statuses.")
        result[i].extend(["delivered"] * n)
        rng.shuffle(result[i])
    if baseline:
        august = result[-1]
        for _ in range(1800):
            august.remove("delivered")
        result[-1] = ["delivered"] * 1800 + august
    return result


def august_customer(
    rng: Random, position: int, status: str, customers: list[CustomersRow], fallback: CustomersRow
) -> CustomersRow:
    """Reserve T5/T7/T8 accounts without adding background cohort members."""
    if position < 440:
        return customers[position]
    if position < 1600:
        return rng.choice(
            [
                c
                for c in customers
                if c.region_id == 3 and not c.is_test_account and c.channel != "affiliate"
            ]
        )
    if position < 1800:
        return rng.choice(customers[7900:8000])
    if status not in ("cancelled", "created"):
        return rng.choice([c for c in customers if c.region_id != 3])
    return fallback


def boundary_time(rng: Random, count: int, start: datetime, end: datetime) -> datetime:
    """Place 12/88/4 T3 orders, then exclude all three reserved cells."""
    if count < 12:
        return start + timedelta(seconds=rng.randrange(8 * 3600))
    if count < 100:
        return start + timedelta(hours=8, seconds=rng.randrange(16 * 3600))
    if count < 104:
        return start + timedelta(days=1, seconds=rng.randrange(8 * 3600))
    for _ in range(10000):
        candidate = shopping_time(rng, start, end)
        if not boundary_cell(candidate):
            return candidate
    raise SeedError("Boundary allocation exhausted bounded attempts.")


def choose_promo(
    rng: Random,
    promotions: list[PromotionsRow],
    paid: datetime | None,
    reserved: bool,
    exclude_summer: bool,
) -> int | None:
    """Choose only promotions active at payment, preserving the exact T7 cohort."""
    if reserved:
        return 17
    if paid is None or rng.randrange(5):
        return None
    active = [
        p
        for p in promotions
        if p.starts_at <= paid < p.ends_at and not (exclude_summer and p.promo_id == 17)
    ]
    return rng.choice(active).promo_id if active else None


def transaction_rows(
    rng: Random,
    parameters: Parameters,
    products: list[ProductsRow],
    customers: list[CustomersRow],
    promotions: list[PromotionsRow],
) -> tuple[list[OrdersRow], list[OrderItemsRow]]:
    """Allocate overlapping August cohorts before ordinary weighted traffic."""
    counts = (
        list(MONTH_COUNTS)
        if parameters.baseline
        else allocate(parameters.orders, [1] * parameters.months)
    )
    statuses = status_plan(rng, counts, parameters.baseline)
    pools = {i: [c for c in customers if c.region_id == i] for i in range(1, 6)}
    orders: list[OrdersRow] = []
    items: list[OrderItemsRow] = []
    boundary_slots = {
        region: deque(slot for slot in range(104) if slot % 5 + 1 == region)
        for region in range(1, 6)
    }
    for month, month_status in enumerate(statuses):
        start, end = (
            month_start(month - parameters.months),
            month_start(month - parameters.months + 1),
        )
        august = parameters.baseline and month == parameters.months - 1
        for position, status in enumerate(month_status):
            region = rng.choices(range(1, 6), weights=(18, 20, 30, 17, 15))[0]
            customer = rng.choice(pools[region])
            reserved_promo = august and position < 1000
            if august:
                customer = august_customer(rng, position, status, customers, customer)
            created = shopping_time(rng, start, end)
            eligible = not customer.is_test_account and status != "cancelled"
            if reserved_promo:
                created = shopping_time(rng, start + timedelta(days=4), start + timedelta(days=20))
            elif august and eligible:
                slots = boundary_slots[customer.region_id]
                slot = slots.popleft() if slots else 104
                created = boundary_time(rng, slot, start, end)
            paid = (
                None
                if status == "created"
                else min(
                    created + timedelta(minutes=rng.randint(1, 120)), end - timedelta(seconds=1)
                )
            )
            if reserved_promo and paid is not None:
                paid = min(paid, start + timedelta(days=20) - timedelta(seconds=1))
            # Leave at least one minute for payment even at a reserved interval end.
            if paid is not None and paid - created < timedelta(minutes=1):
                created = paid - timedelta(minutes=1)
            promo_id = choose_promo(
                rng, promotions, paid, reserved_promo, august and customer.region_id == 3
            )
            order_id = len(orders) + 1
            plan = line_plan(rng, order_id, products, created < month_start(-8), len(items))
            orders.append(
                OrdersRow(
                    order_id=order_id,
                    customer_id=customer.customer_id,
                    region_id=customer.region_id,
                    created_at=created,
                    paid_at=paid,
                    status=status,
                    gross_amount=sum((i.unit_price * i.quantity for i in plan), Decimal(0)),
                    discount_amount=sum((i.item_discount or Decimal(0) for i in plan), Decimal(0)),
                    shipping_fee=money(rng.choices((0, 600, 1000), weights=(70, 20, 10))[0]),
                    promo_id=promo_id,
                )
            )
            items.extend(plan)
    if parameters.baseline and any(boundary_slots.values()):
        raise SeedError("Unable to distribute T3 boundary quotas across five regions.")
    return orders, items


def refund_rows(
    rng: Random, parameters: Parameters, orders: list[OrdersRow], customers: list[CustomersRow]
) -> list[RefundsRow]:
    """Reserve T7 refund cohorts and condition bounded Gamma lags on month crossing."""
    eligible = [o for o in orders if o.paid_at and o.status != "cancelled"]
    count = min(len(eligible), max(1, parameters.orders * 3238 // 50000))
    selected: list[OrdersRow] = []
    if parameters.baseline:
        east = [
            o
            for o in eligible
            if o.region_id == 3
            and o.paid_at is not None
            and o.paid_at >= month_start(-1)
            and not customers[o.customer_id - 1].is_test_account
        ]
        selected.extend(rng.sample([o for o in east if o.promo_id == 17], 200))
        selected.extend(rng.sample([o for o in east if o.promo_id != 17], 30))
        selected.extend(
            rng.sample(
                [
                    o
                    for o in eligible
                    if o.region_id != 3
                    and o.paid_at is not None
                    and o.paid_at >= month_start(-1)
                    and not customers[o.customer_id - 1].is_test_account
                ],
                170,
            )
        )
        remaining = [o for o in eligible if o.paid_at is not None and o.paid_at < month_start(-1)]
    else:
        remaining = eligible
    selected.extend(rng.sample(remaining, count - len(selected)))
    duplicate = set(rng.sample([o.order_id for o in selected], round(count * 0.05)))
    # Last-day payments must cross: conditioning a Gamma on a few remaining
    # seconds is numerically impractical. Remaining slots keep the 22% quota.
    crossing = {
        o.order_id
        for o in selected
        if o.paid_at is not None
        and (o.paid_at + timedelta(days=1)).astimezone(SHANGHAI).month
        != o.paid_at.astimezone(SHANGHAI).month
    }
    candidates = [o.order_id for o in selected if o.order_id not in crossing]
    crossing.update(rng.sample(candidates, max(0, round(count * 0.22) - len(crossing))))
    rejected = {o.order_id for o in selected[-round(count * 0.10) :]} if count >= 10 else set()
    rows: list[RefundsRow] = []
    for order in selected:
        if order.paid_at is None:
            raise SeedError("Refund requires a paid order.")
        paid_month = order.paid_at.astimezone(SHANGHAI).month
        for _ in range(10000):
            lag = max(1, round(rng.gammavariate(2, 6) * 86400))
            requested = order.paid_at + timedelta(seconds=lag)
            if lag <= 90 * 86400 and (
                (requested.astimezone(SHANGHAI).month != paid_month) == (order.order_id in crossing)
            ):
                break
        else:
            raise SeedError("Refund lag allocation exhausted bounded attempts.")
        total = (
            int((order.gross_amount - order.discount_amount) * 100) * rng.randint(10, 100)
        ) // 100
        amounts = [total // 2, total - total // 2] if order.order_id in duplicate else [total]
        for amount in amounts:
            status = (
                "rejected"
                if order.order_id in rejected
                else rng.choices(("requested", "approved", "completed"), weights=(5, 10, 75))[0]
            )
            rows.append(
                RefundsRow(
                    refund_id=1,
                    order_id=order.order_id,
                    requested_at=requested,
                    completed_at=requested + timedelta(days=rng.randint(1, 14))
                    if status == "completed"
                    else None,
                    amount=money(amount),
                    reason_code=("quality", "size", "delayed", "changed_mind", "other")[
                        len(rows) % 5
                    ],
                    status=status,
                )
            )
    rows.sort(key=lambda r: (r.requested_at, r.order_id))
    for i, row in enumerate(rows, 1):
        row.refund_id = i
    return rows


def generate(parameters: Parameters) -> Dataset:
    """Generate the complete typed dataset without database or filesystem I/O."""
    rng = Random(parameters.seed)  # noqa: S311 -- deterministic synthetic data, no secrets.
    start = month_start(-parameters.months)
    regions, promotions, products, customers = dimensions(rng, start)
    orders, items = transaction_rows(rng, parameters, products, customers, promotions)
    first: dict[int, datetime] = {}
    for order in orders:
        first[order.customer_id] = min(
            first.get(order.customer_id, order.created_at), order.created_at
        )
    for customer in customers:
        customer.registered_at = (
            first[customer.customer_id] - timedelta(days=rng.randint(1, 365))
            if customer.customer_id in first
            else shopping_time(rng, start, END)
        )
    refunds = refund_rows(rng, parameters, orders, customers)
    inventory = [
        InventoryRow(
            product_id=p.product_id,
            warehouse=("WH-N", "WH-S", "WH-E")[(p.product_id - 1) % 3],
            quantity=rng.randint(0, 500),
            reorder_level=rng.randint(10, 50),
            updated_at=datetime(2026, 12, 14, 15, 59, 59, tzinfo=UTC),
        )
        for p in products
    ]
    return Dataset(
        parameters=parameters,
        tables=[
            list(regions),
            list(promotions),
            list(products),
            list(customers),
            list(orders),
            list(items),
            list(refunds),
            list(inventory),
        ],
    )
