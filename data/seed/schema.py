"""Business and seed-operation metadata; never loaded by the API."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import NAMING_CONVENTION

BUSINESS_METADATA = sa.MetaData(naming_convention=NAMING_CONVENTION)

REGIONS = sa.Table(
    "regions",
    BUSINESS_METADATA,
    sa.Column("region_id", sa.Integer, nullable=False, primary_key=True),
    sa.Column("name", sa.String(40), nullable=False, unique=True),
    sa.Column("name_en", sa.String(80), nullable=False, unique=True),
    sa.Column("renamed_from", sa.String(40), nullable=True),
    sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint(
        "renamed_from IS NULL OR renamed_from <> name", name=sa.schema.conv("ck_regions_rename")
    ),
    schema="biz",
)

PROMOTIONS = sa.Table(
    "promotions",
    BUSINESS_METADATA,
    sa.Column("promo_id", sa.Integer, nullable=False, primary_key=True),
    sa.Column("name", sa.String(100), nullable=False),
    sa.Column("kind", sa.String(20), nullable=False),
    sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("ends_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("rule_doc_ref", sa.String(200), nullable=True),
    sa.CheckConstraint(
        "kind IN ('discount','seasonal','shipping')", name=sa.schema.conv("ck_promotions_kind")
    ),
    sa.CheckConstraint("starts_at < ends_at", name=sa.schema.conv("ck_promotions_interval")),
    schema="biz",
)

PRODUCTS = sa.Table(
    "products",
    BUSINESS_METADATA,
    sa.Column("product_id", sa.BigInteger, nullable=False, primary_key=True),
    sa.Column("sku", sa.String(32), nullable=False, unique=True),
    sa.Column("category", sa.String(32), nullable=False),
    sa.Column("list_price", sa.Numeric(12, 2), nullable=False),
    sa.Column("cost", sa.Numeric(12, 2), nullable=False),
    sa.Column("launched_at", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint(
        "list_price > 0 AND cost >= 0 AND cost <= list_price",
        name=sa.schema.conv("ck_products_price"),
    ),
    sa.CheckConstraint(
        "category IN ('apparel','beauty','home','electronics','food')",
        name=sa.schema.conv("ck_products_category"),
    ),
    schema="biz",
)

CUSTOMERS = sa.Table(
    "customers",
    BUSINESS_METADATA,
    sa.Column("customer_id", sa.BigInteger, nullable=False, primary_key=True),
    sa.Column("region_id", sa.Integer, nullable=False),
    sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("channel", sa.String(20), nullable=False),
    sa.Column("is_test_account", sa.Boolean, nullable=False, server_default=sa.text("false")),
    sa.Column("phone", sa.String(24), nullable=False),
    sa.Column("email", sa.String(254), nullable=False, unique=True),
    sa.ForeignKeyConstraint(
        ["region_id"],
        ["biz.regions.region_id"],
        name="fk_customers_region_id_regions",
        ondelete="RESTRICT",
    ),
    sa.CheckConstraint(
        "channel IN ('organic','search','social','affiliate')",
        name=sa.schema.conv("ck_customers_channel"),
    ),
    schema="biz",
)

ORDERS = sa.Table(
    "orders",
    BUSINESS_METADATA,
    sa.Column("order_id", sa.BigInteger, nullable=False, primary_key=True),
    sa.Column("customer_id", sa.BigInteger, nullable=False),
    sa.Column("region_id", sa.Integer, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("gross_amount", sa.Numeric(12, 2), nullable=False),
    sa.Column("discount_amount", sa.Numeric(12, 2), nullable=False, server_default=sa.text("0")),
    sa.Column("shipping_fee", sa.Numeric(12, 2), nullable=False, server_default=sa.text("0")),
    sa.Column("promo_id", sa.Integer, nullable=True),
    sa.ForeignKeyConstraint(
        ["customer_id"],
        ["biz.customers.customer_id"],
        name="fk_orders_customer_id_customers",
        ondelete="RESTRICT",
    ),
    sa.ForeignKeyConstraint(
        ["region_id"],
        ["biz.regions.region_id"],
        name="fk_orders_region_id_regions",
        ondelete="RESTRICT",
    ),
    sa.ForeignKeyConstraint(
        ["promo_id"],
        ["biz.promotions.promo_id"],
        name="fk_orders_promo_id_promotions",
        ondelete="RESTRICT",
    ),
    sa.CheckConstraint(
        "status IN ('created','paid','shipped','delivered','cancelled','closed')",
        name=sa.schema.conv("ck_orders_status"),
    ),
    sa.CheckConstraint(
        "gross_amount >= 0 AND discount_amount >= 0 AND discount_amount <= gross_amount AND shipping_fee >= 0",
        name=sa.schema.conv("ck_orders_amount"),
    ),
    sa.CheckConstraint(
        "paid_at IS NULL OR paid_at >= created_at", name=sa.schema.conv("ck_orders_paid_time")
    ),
    sa.CheckConstraint(
        "(status = 'created' AND paid_at IS NULL) OR (status <> 'created' AND paid_at IS NOT NULL)",
        name=sa.schema.conv("ck_orders_payment"),
    ),
    schema="biz",
)

ORDER_ITEMS = sa.Table(
    "order_items",
    BUSINESS_METADATA,
    sa.Column("order_item_id", sa.BigInteger, nullable=False, primary_key=True),
    sa.Column("order_id", sa.BigInteger, nullable=False),
    sa.Column("product_id", sa.BigInteger, nullable=False),
    sa.Column("quantity", sa.Integer, nullable=False),
    sa.Column("unit_price", sa.Numeric(12, 2), nullable=False),
    sa.Column("item_discount", sa.Numeric(12, 2), nullable=True),
    sa.ForeignKeyConstraint(
        ["order_id"],
        ["biz.orders.order_id"],
        name="fk_order_items_order_id_orders",
        ondelete="RESTRICT",
    ),
    sa.ForeignKeyConstraint(
        ["product_id"],
        ["biz.products.product_id"],
        name="fk_order_items_product_id_products",
        ondelete="RESTRICT",
    ),
    sa.UniqueConstraint("order_id", "product_id", name="uq_order_items_order_id"),
    sa.CheckConstraint(
        "quantity > 0 AND unit_price > 0 AND (item_discount IS NULL OR (item_discount >= 0 AND item_discount <= quantity * unit_price))",
        name=sa.schema.conv("ck_order_items_amount"),
    ),
    schema="biz",
)

REFUNDS = sa.Table(
    "refunds",
    BUSINESS_METADATA,
    sa.Column("refund_id", sa.BigInteger, nullable=False, primary_key=True),
    sa.Column("order_id", sa.BigInteger, nullable=False),
    sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("amount", sa.Numeric(12, 2), nullable=False),
    sa.Column("reason_code", sa.String(24), nullable=False),
    sa.Column("status", sa.String(16), nullable=False),
    sa.ForeignKeyConstraint(
        ["order_id"],
        ["biz.orders.order_id"],
        name="fk_refunds_order_id_orders",
        ondelete="RESTRICT",
    ),
    sa.CheckConstraint("amount > 0", name=sa.schema.conv("ck_refunds_amount")),
    sa.CheckConstraint(
        "reason_code IN ('quality','size','delayed','changed_mind','other')",
        name=sa.schema.conv("ck_refunds_reason"),
    ),
    sa.CheckConstraint(
        "status IN ('requested','approved','rejected','completed')",
        name=sa.schema.conv("ck_refunds_status"),
    ),
    sa.CheckConstraint(
        "(status = 'completed' AND completed_at IS NOT NULL AND completed_at >= requested_at) OR (status <> 'completed' AND completed_at IS NULL)",
        name=sa.schema.conv("ck_refunds_completion"),
    ),
    schema="biz",
)

INVENTORY = sa.Table(
    "inventory",
    BUSINESS_METADATA,
    sa.Column("product_id", sa.BigInteger, nullable=False, primary_key=True),
    sa.Column("warehouse", sa.String(16), nullable=False),
    sa.Column("quantity", sa.Integer, nullable=False),
    sa.Column("reorder_level", sa.Integer, nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(
        ["product_id"],
        ["biz.products.product_id"],
        name="fk_inventory_product_id_products",
        ondelete="RESTRICT",
    ),
    sa.CheckConstraint(
        "quantity >= 0 AND reorder_level >= 0", name=sa.schema.conv("ck_inventory_quantity")
    ),
    sa.CheckConstraint(
        "warehouse IN ('WH-N','WH-S','WH-E')", name=sa.schema.conv("ck_inventory_warehouse")
    ),
    schema="biz",
)

sa.Index("ix_regions_renamed_from", REGIONS.c.renamed_from)
sa.Index("ix_promotions_starts_at", PROMOTIONS.c.starts_at, PROMOTIONS.c.ends_at)
sa.Index("ix_products_category", PRODUCTS.c.category)
sa.Index("ix_customers_region_id", CUSTOMERS.c.region_id)
sa.Index("ix_customers_phone", CUSTOMERS.c.phone)
sa.Index("ix_customers_registered_at", CUSTOMERS.c.registered_at)
sa.Index("ix_orders_customer_id", ORDERS.c.customer_id)
sa.Index("ix_orders_region_paid", ORDERS.c.region_id, ORDERS.c.paid_at)
sa.Index("ix_orders_promo_id", ORDERS.c.promo_id)
sa.Index("ix_orders_paid_at", ORDERS.c.paid_at)
sa.Index("ix_orders_created_at", ORDERS.c.created_at)
sa.Index("ix_orders_status", ORDERS.c.status)
sa.Index("ix_order_items_product_id", ORDER_ITEMS.c.product_id)
sa.Index("ix_refunds_order_id", REFUNDS.c.order_id)
sa.Index("ix_refunds_requested_at", REFUNDS.c.requested_at)
sa.Index("ix_refunds_status", REFUNDS.c.status)
sa.Index("ix_inventory_warehouse", INVENTORY.c.warehouse)

SEED_MANIFEST = sa.Table(
    "seed_manifest",
    BUSINESS_METADATA,
    sa.Column("dataset_id", sa.String(80), primary_key=True),
    sa.Column("manifest", JSONB, nullable=False),
    schema="ops",
)

TABLES = (
    REGIONS,
    PROMOTIONS,
    PRODUCTS,
    CUSTOMERS,
    ORDERS,
    ORDER_ITEMS,
    REFUNDS,
    INVENTORY,
)
