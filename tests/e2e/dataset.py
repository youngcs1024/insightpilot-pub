"""Small explicit business facts; the expected totals never execute generated SQL."""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from data.seed.contracts import Dataset, Parameters
from data.seed.rows import (
    CustomersRow,
    InventoryRow,
    OrderItemsRow,
    OrdersRow,
    ProductsRow,
    RefundsRow,
    RegionsRow,
)

POLICY_TEXT = "定制商品与已拆封的个人卫生用品不适用七天无理由退货。退款申请需审核，政策本身不能证明退款率上升的原因。"


def instant(value: str) -> datetime:
    return datetime.fromisoformat(value + "+08:00")


def business() -> Dataset:
    """July east 1/2, August east 2/2; August GMV 600, south GMV 300."""
    origin = instant("2026-01-01T00:00:00")
    regions = [
        RegionsRow(
            region_id=2,
            name="华南",
            name_en="South China",
            renamed_from=None,
            effective_from=origin,
        ),
        RegionsRow(
            region_id=3,
            name="华东一区",
            name_en="East China",
            renamed_from="华东",
            effective_from=origin,
        ),
    ]
    customers = [
        CustomersRow(
            customer_id=i,
            region_id=region,
            registered_at=origin,
            channel="organic",
            is_test_account=test,
            phone=f"IP{i:010d}",
            email=f"e2e-{i}@example.invalid",
        )
        for i, region, test in ((1, 3, False), (2, 2, False), (3, 3, True))
    ]
    # Date, customer, net amount, status. Cancelled and test-account rows must not count.
    facts = [
        ("2026-07-10T10:00:00", 1, "100", "delivered"),
        ("2026-07-20T10:00:00", 1, "200", "delivered"),
        ("2026-08-10T10:00:00", 1, "100", "delivered"),
        ("2026-08-20T10:00:00", 1, "200", "delivered"),
        ("2026-08-15T10:00:00", 2, "300", "delivered"),
        ("2026-08-15T11:00:00", 1, "900", "cancelled"),
        ("2026-08-15T12:00:00", 3, "800", "delivered"),
    ]
    orders, items = [], []
    for index, (date, customer, amount, status) in enumerate(facts, 1):
        paid = instant(date)
        gross = Decimal(amount) + Decimal("10")
        orders.append(
            OrdersRow(
                order_id=index,
                customer_id=customer,
                region_id=customers[customer - 1].region_id,
                created_at=paid - timedelta(minutes=5),
                paid_at=paid,
                status=status,
                gross_amount=gross,
                discount_amount=Decimal("10"),
                shipping_fee=Decimal("5"),
                promo_id=None,
            )
        )
        items.append(
            OrderItemsRow(
                order_item_id=index,
                order_id=index,
                product_id=1,
                quantity=1,
                unit_price=gross,
                item_discount=Decimal("10"),
            )
        )
    refunds = [
        RefundsRow(
            refund_id=index,
            order_id=order,
            requested_at=orders[order - 1].paid_at + timedelta(days=1),
            completed_at=None,
            amount=Decimal("10"),
            reason_code="quality",
            status="requested",
        )
        for index, order in enumerate((1, 3, 4), 1)
    ]
    return Dataset(
        parameters=Parameters(orders=len(orders), months=2),
        tables=[
            regions,
            [],
            [
                ProductsRow(
                    product_id=1,
                    sku="E2E-001",
                    category="home",
                    list_price=Decimal("1000"),
                    cost=Decimal("1"),
                    launched_at=origin,
                )
            ],
            customers,
            orders,
            items,
            refunds,
            [
                InventoryRow(
                    product_id=1, warehouse="WH-E", quantity=100, reorder_level=1, updated_at=origin
                )
            ],
        ],
    )


def corpus(directory: Path) -> None:
    """Write new fixture-owned sources; no production corpus or files are replaced."""
    directory.mkdir(parents=True)
    names = ("returns", "refunds")
    texts = (
        POLICY_TEXT,
        "退款申请订单率按申请时间统计去重订单，以同期合格支付订单为分母。需排除取消订单和测试账号，相关性不能证明因果关系。",
    )
    for name, text in zip(names, texts, strict=True):
        (directory / f"{name}.md").write_text(
            f"---\ntitle: {name}\ndoc_type: policy\neffective_from: 2026-01-01\n"
            "effective_to: null\nsupersedes: null\n---\n\n# 退款规则\n\n" + text + "\n"
        )
    (directory / "MANIFEST.yaml").write_text(
        "schema_version: 1\ndocuments:\n"
        + "".join(f"- path: {name}.md\n  format: md\n  metadata_path: null\n" for name in names)
    )
