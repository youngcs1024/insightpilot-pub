"""Eight business tables and atomic seed import provenance."""

from alembic import op

revision = "0001_business_schema"
down_revision = "0000_business_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply frozen Step 0.7 DDL under biz_owner."""
    op.execute(
        """
        CREATE TABLE biz.regions (
            region_id INTEGER CONSTRAINT pk_regions PRIMARY KEY,
            name VARCHAR(40) NOT NULL CONSTRAINT uq_regions_name UNIQUE,
            name_en VARCHAR(80) NOT NULL CONSTRAINT uq_regions_name_en UNIQUE,
            renamed_from VARCHAR(40),
            effective_from TIMESTAMPTZ NOT NULL,
            CONSTRAINT ck_regions_rename CHECK (renamed_from IS NULL OR renamed_from <> name)
        )
        """
    )
    op.execute("CREATE INDEX ix_regions_renamed_from ON biz.regions (renamed_from)")
    op.execute(
        """
        CREATE TABLE biz.promotions (
            promo_id INTEGER CONSTRAINT pk_promotions PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            kind VARCHAR(20) NOT NULL,
            starts_at TIMESTAMPTZ NOT NULL,
            ends_at TIMESTAMPTZ NOT NULL,
            rule_doc_ref VARCHAR(200),
            CONSTRAINT ck_promotions_kind CHECK (kind IN ('discount','seasonal','shipping')),
            CONSTRAINT ck_promotions_interval CHECK (starts_at < ends_at)
        )
        """
    )
    op.execute("CREATE INDEX ix_promotions_starts_at ON biz.promotions (starts_at, ends_at)")
    op.execute(
        """
        CREATE TABLE biz.products (
            product_id BIGINT CONSTRAINT pk_products PRIMARY KEY,
            sku VARCHAR(32) NOT NULL CONSTRAINT uq_products_sku UNIQUE,
            category VARCHAR(32) NOT NULL,
            list_price NUMERIC(12,2) NOT NULL,
            cost NUMERIC(12,2) NOT NULL,
            launched_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT ck_products_price CHECK (list_price > 0 AND cost >= 0 AND cost <= list_price),
            CONSTRAINT ck_products_category CHECK
              (category IN ('apparel','beauty','home','electronics','food'))
        )
        """
    )
    op.execute("CREATE INDEX ix_products_category ON biz.products (category)")
    op.execute(
        """
        CREATE TABLE biz.customers (
            customer_id BIGINT CONSTRAINT pk_customers PRIMARY KEY,
            region_id INTEGER NOT NULL,
            registered_at TIMESTAMPTZ NOT NULL,
            channel VARCHAR(20) NOT NULL,
            is_test_account BOOLEAN NOT NULL DEFAULT FALSE,
            phone VARCHAR(24) NOT NULL,
            email VARCHAR(254) NOT NULL CONSTRAINT uq_customers_email UNIQUE,
            CONSTRAINT fk_customers_region_id_regions FOREIGN KEY (region_id)
              REFERENCES biz.regions(region_id) ON DELETE RESTRICT,
            CONSTRAINT ck_customers_channel CHECK (channel IN ('organic','search','social','affiliate'))
        )
        """
    )
    op.execute("CREATE INDEX ix_customers_region_id ON biz.customers (region_id)")
    op.execute("CREATE INDEX ix_customers_phone ON biz.customers (phone)")
    op.execute("CREATE INDEX ix_customers_registered_at ON biz.customers (registered_at)")
    op.execute(
        """
        CREATE TABLE biz.orders (
            order_id BIGINT CONSTRAINT pk_orders PRIMARY KEY,
            customer_id BIGINT NOT NULL,
            region_id INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            paid_at TIMESTAMPTZ,
            status VARCHAR(16) NOT NULL,
            gross_amount NUMERIC(12,2) NOT NULL,
            discount_amount NUMERIC(12,2) NOT NULL DEFAULT 0,
            shipping_fee NUMERIC(12,2) NOT NULL DEFAULT 0,
            promo_id INTEGER,
            CONSTRAINT fk_orders_customer_id_customers FOREIGN KEY (customer_id)
              REFERENCES biz.customers(customer_id) ON DELETE RESTRICT,
            CONSTRAINT fk_orders_region_id_regions FOREIGN KEY (region_id)
              REFERENCES biz.regions(region_id) ON DELETE RESTRICT,
            CONSTRAINT fk_orders_promo_id_promotions FOREIGN KEY (promo_id)
              REFERENCES biz.promotions(promo_id) ON DELETE RESTRICT,
            CONSTRAINT ck_orders_status CHECK
              (status IN ('created','paid','shipped','delivered','cancelled','closed')),
            CONSTRAINT ck_orders_amount CHECK (gross_amount >= 0 AND discount_amount >= 0
              AND discount_amount <= gross_amount AND shipping_fee >= 0),
            CONSTRAINT ck_orders_paid_time CHECK (paid_at IS NULL OR paid_at >= created_at),
            CONSTRAINT ck_orders_payment CHECK
              ((status = 'created' AND paid_at IS NULL) OR (status <> 'created' AND paid_at IS NOT NULL))
        )
        """
    )
    op.execute("CREATE INDEX ix_orders_customer_id ON biz.orders (customer_id)")
    op.execute("CREATE INDEX ix_orders_region_paid ON biz.orders (region_id, paid_at)")
    op.execute("CREATE INDEX ix_orders_promo_id ON biz.orders (promo_id)")
    op.execute("CREATE INDEX ix_orders_paid_at ON biz.orders (paid_at)")
    op.execute("CREATE INDEX ix_orders_created_at ON biz.orders (created_at)")
    op.execute("CREATE INDEX ix_orders_status ON biz.orders (status)")
    op.execute(
        """
        CREATE TABLE biz.order_items (
            order_item_id BIGINT CONSTRAINT pk_order_items PRIMARY KEY,
            order_id BIGINT NOT NULL,
            product_id BIGINT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price NUMERIC(12,2) NOT NULL,
            item_discount NUMERIC(12,2),
            CONSTRAINT uq_order_items_order_id UNIQUE (order_id, product_id),
            CONSTRAINT fk_order_items_order_id_orders FOREIGN KEY (order_id)
              REFERENCES biz.orders(order_id) ON DELETE RESTRICT,
            CONSTRAINT fk_order_items_product_id_products FOREIGN KEY (product_id)
              REFERENCES biz.products(product_id) ON DELETE RESTRICT,
            CONSTRAINT ck_order_items_amount CHECK (quantity > 0 AND unit_price > 0
              AND (item_discount IS NULL OR
                (item_discount >= 0 AND item_discount <= quantity * unit_price)))
        )
        """
    )
    op.execute("CREATE INDEX ix_order_items_product_id ON biz.order_items (product_id)")
    op.execute(
        """
        CREATE TABLE biz.refunds (
            refund_id BIGINT CONSTRAINT pk_refunds PRIMARY KEY,
            order_id BIGINT NOT NULL,
            requested_at TIMESTAMPTZ NOT NULL,
            completed_at TIMESTAMPTZ,
            amount NUMERIC(12,2) NOT NULL,
            reason_code VARCHAR(24) NOT NULL,
            status VARCHAR(16) NOT NULL,
            CONSTRAINT fk_refunds_order_id_orders FOREIGN KEY (order_id)
              REFERENCES biz.orders(order_id) ON DELETE RESTRICT,
            CONSTRAINT ck_refunds_amount CHECK (amount > 0),
            CONSTRAINT ck_refunds_reason CHECK
              (reason_code IN ('quality','size','delayed','changed_mind','other')),
            CONSTRAINT ck_refunds_status CHECK (status IN ('requested','approved','rejected','completed')),
            CONSTRAINT ck_refunds_completion CHECK
              ((status = 'completed' AND completed_at IS NOT NULL AND completed_at >= requested_at)
               OR (status <> 'completed' AND completed_at IS NULL))
        )
        """
    )
    op.execute("CREATE INDEX ix_refunds_order_id ON biz.refunds (order_id)")
    op.execute("CREATE INDEX ix_refunds_requested_at ON biz.refunds (requested_at)")
    op.execute("CREATE INDEX ix_refunds_status ON biz.refunds (status)")
    op.execute(
        """
        CREATE TABLE biz.inventory (
            product_id BIGINT CONSTRAINT pk_inventory PRIMARY KEY,
            warehouse VARCHAR(16) NOT NULL,
            quantity INTEGER NOT NULL,
            reorder_level INTEGER NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT fk_inventory_product_id_products FOREIGN KEY (product_id)
              REFERENCES biz.products(product_id) ON DELETE RESTRICT,
            CONSTRAINT ck_inventory_quantity CHECK (quantity >= 0 AND reorder_level >= 0),
            CONSTRAINT ck_inventory_warehouse CHECK (warehouse IN ('WH-N','WH-S','WH-E'))
        )
        """
    )
    op.execute("CREATE INDEX ix_inventory_warehouse ON biz.inventory (warehouse)")
    op.execute("CREATE SCHEMA ops AUTHORIZATION biz_owner")
    op.execute("REVOKE ALL ON SCHEMA ops FROM PUBLIC, app_rw, mcp_ro")
    op.execute("GRANT USAGE ON SCHEMA ops TO etl_rw")
    op.execute(
        "CREATE TABLE ops.seed_manifest (dataset_id VARCHAR(80) CONSTRAINT pk_seed_manifest PRIMARY KEY, manifest JSONB NOT NULL)"
    )
    op.execute("REVOKE ALL ON ops.seed_manifest FROM PUBLIC, app_rw, mcp_ro")
    op.execute("GRANT SELECT, INSERT ON ops.seed_manifest TO etl_rw")


def downgrade() -> None:
    """Remove only this revision's objects in dependency order."""
    op.drop_table("seed_manifest", schema="ops")
    op.execute("DROP SCHEMA ops")
    op.drop_table("inventory", schema="biz")
    op.drop_table("refunds", schema="biz")
    op.drop_table("order_items", schema="biz")
    op.drop_table("orders", schema="biz")
    op.drop_table("customers", schema="biz")
    op.drop_table("products", schema="biz")
    op.drop_table("promotions", schema="biz")
    op.drop_table("regions", schema="biz")
