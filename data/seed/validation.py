"""Cross-table seed invariants checked before export and before any import writes."""

# Numeric bounds are the declared Step 0.7 baseline acceptance contract.
# ruff: noqa: PLR2004

from collections import Counter, defaultdict
from datetime import timedelta
from decimal import Decimal
from typing import cast

from data.seed.contracts import CUTOFF, END, Dataset, SeedError
from data.seed.generation import MONTH_COUNTS, SHANGHAI, month_start
from data.seed.rows import (
    CustomersRow,
    OrderItemsRow,
    OrdersRow,
    ProductsRow,
    PromotionsRow,
    RefundsRow,
)


def require(condition: bool, message: str) -> None:
    """Fail with a typed, nonretryable validation error."""
    if not condition:
        raise SeedError(message)


def validate(dataset: Dataset) -> None:
    """Reconcile references, exact totals and times without SQL or generated expectations."""
    tables = dataset.tables
    require(len(tables) == 8, "Eight tables required.")
    products = {p.product_id: p for p in cast("list[ProductsRow]", tables[2])}
    customers = {c.customer_id: c for c in cast("list[CustomersRow]", tables[3])}
    orders = {o.order_id: o for o in cast("list[OrdersRow]", tables[4])}
    promos = {p.promo_id: p for p in cast("list[PromotionsRow]", tables[1])}
    require(len(orders) == dataset.parameters.orders, "Order count differs from parameters.")
    gross: dict[int, Decimal] = defaultdict(Decimal)
    discount: dict[int, Decimal] = defaultdict(Decimal)
    item_keys: set[tuple[int, int]] = set()
    for item in cast("list[OrderItemsRow]", tables[5]):
        require(item.order_id in orders and item.product_id in products, "Invalid item reference.")
        require(
            (item.order_id, item.product_id) not in item_keys, "Duplicate product within an order."
        )
        item_keys.add((item.order_id, item.product_id))
        require(
            products[item.product_id].launched_at <= orders[item.order_id].created_at,
            "Product not launched.",
        )
        require(item.quantity > 0 and item.unit_price > 0, "Invalid item amount.")
        value = item.unit_price * item.quantity
        require(Decimal(0) <= (item.item_discount or Decimal(0)) <= value, "Invalid line discount.")
        gross[item.order_id] += value
        discount[item.order_id] += item.item_discount or Decimal(0)
    for order in orders.values():
        require(order.customer_id in customers, "Invalid customer reference.")
        require(order.region_id == customers[order.customer_id].region_id, "Sale region mismatch.")
        require(
            customers[order.customer_id].registered_at < order.created_at,
            "Customer registered too late.",
        )
        require(
            month_start(-dataset.parameters.months) <= order.created_at < END,
            "Order outside window.",
        )
        require(
            order.gross_amount == gross[order.order_id]
            and order.discount_amount == discount[order.order_id],
            "Order/item totals differ.",
        )
        validate_payment(order, promos)
    requested: dict[int, Decimal] = defaultdict(Decimal)
    for refund in cast("list[RefundsRow]", tables[6]):
        require(refund.order_id in orders, "Invalid refund reference.")
        order = orders[refund.order_id]
        require(
            order.paid_at is not None and order.status != "cancelled",
            "Refund requires eligible payment.",
        )
        require(
            order.paid_at is not None and order.paid_at <= refund.requested_at < CUTOFF,
            "Refund time invalid.",
        )
        require(
            order.paid_at is not None and refund.requested_at - order.paid_at <= timedelta(days=90),
            "Refund lag exceeds 90 days.",
        )
        require(
            (refund.status == "completed") == (refund.completed_at is not None),
            "Refund completion mismatch.",
        )
        require(
            refund.completed_at is None or refund.requested_at <= refund.completed_at < CUTOFF,
            "Completion time invalid.",
        )
        require(refund.amount > 0, "Refund amount must be positive.")
        requested[order.order_id] += refund.amount
    for order_id, amount in requested.items():
        order = orders[order_id]
        require(
            amount <= order.gross_amount - order.discount_amount, "Refund exceeds net merchandise."
        )
    if dataset.parameters.baseline:
        validate_baseline(dataset)


def validate_payment(order: OrdersRow, promos: dict[int, PromotionsRow]) -> None:
    """Payment and promotion applicability are independent of generation order."""
    require((order.status == "created") == (order.paid_at is None), "Payment status mismatch.")
    require(
        order.paid_at is None or order.created_at <= order.paid_at < END, "Payment outside window."
    )
    require(
        order.paid_at is None
        or timedelta(minutes=1) <= order.paid_at - order.created_at <= timedelta(minutes=120),
        "Payment delay must be 1 to 120 minutes.",
    )
    if order.promo_id is not None:
        require(order.promo_id in promos, "Missing promotion.")
        promo = promos[order.promo_id]
        require(
            order.paid_at is not None and promo.starts_at <= order.paid_at < promo.ends_at,
            "Promotion not active.",
        )


def validate_baseline(dataset: Dataset) -> None:
    """Check frozen global allocations; SQL acceptance separately measures trap deltas."""
    tables = dataset.tables
    require(
        [len(tables[i]) for i in (0, 1, 2, 3, 4, 7)] == [5, 40, 1200, 8000, 50000, 1200],
        "Baseline dimensions differ.",
    )
    require(
        135000 <= len(tables[5]) <= 145000 and 3200 <= len(tables[6]) <= 3600,
        "Baseline fact counts differ.",
    )
    orders = cast("list[OrdersRow]", tables[4])
    counts = Counter(o.created_at.astimezone(SHANGHAI).strftime("%Y-%m") for o in orders)
    require([n for _, n in sorted(counts.items())] == list(MONTH_COUNTS), "Monthly quotas differ.")
    require(
        Counter(o.status for o in orders)
        == {
            "created": 500,
            "paid": 2000,
            "shipped": 4000,
            "delivered": 38000,
            "cancelled": 3000,
            "closed": 2500,
        },
        "Status quotas differ.",
    )
