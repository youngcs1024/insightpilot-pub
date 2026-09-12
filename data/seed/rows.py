"""Typed business rows; storage encoding preserves exact money and UTC instants."""

from datetime import UTC, datetime
from decimal import Decimal

from pydantic import AwareDatetime, BaseModel, ConfigDict, field_serializer


class SeedRow(BaseModel):
    """Reject undeclared columns at the file boundary."""

    model_config = ConfigDict(extra="forbid")

    @field_serializer("*", check_fields=False, when_used="json")
    def canonical(self, value: object) -> object:
        """Stable UTC instants and two-place decimal money for export only."""
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
        if isinstance(value, Decimal):
            return format(value, ".2f")
        return value


class RegionsRow(SeedRow):
    """One regions record."""

    region_id: int
    name: str
    name_en: str
    renamed_from: str | None
    effective_from: AwareDatetime


class PromotionsRow(SeedRow):
    """One promotions record."""

    promo_id: int
    name: str
    kind: str
    starts_at: AwareDatetime
    ends_at: AwareDatetime
    rule_doc_ref: str | None


class ProductsRow(SeedRow):
    """One products record."""

    product_id: int
    sku: str
    category: str
    list_price: Decimal
    cost: Decimal
    launched_at: AwareDatetime


class CustomersRow(SeedRow):
    """One customers record."""

    customer_id: int
    region_id: int
    registered_at: AwareDatetime
    channel: str
    is_test_account: bool
    phone: str
    email: str


class OrdersRow(SeedRow):
    """One orders record."""

    order_id: int
    customer_id: int
    region_id: int
    created_at: AwareDatetime
    paid_at: AwareDatetime | None
    status: str
    gross_amount: Decimal
    discount_amount: Decimal
    shipping_fee: Decimal
    promo_id: int | None


class OrderItemsRow(SeedRow):
    """One order_items record."""

    order_item_id: int
    order_id: int
    product_id: int
    quantity: int
    unit_price: Decimal
    item_discount: Decimal | None


class RefundsRow(SeedRow):
    """One refunds record."""

    refund_id: int
    order_id: int
    requested_at: AwareDatetime
    completed_at: AwareDatetime | None
    amount: Decimal
    reason_code: str
    status: str


class InventoryRow(SeedRow):
    """One inventory record."""

    product_id: int
    warehouse: str
    quantity: int
    reorder_level: int
    updated_at: AwareDatetime


ROW_MODELS: tuple[type[SeedRow], ...] = (
    RegionsRow,
    PromotionsRow,
    ProductsRow,
    CustomersRow,
    OrdersRow,
    OrderItemsRow,
    RefundsRow,
    InventoryRow,
)
