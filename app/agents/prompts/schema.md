# Static business schema v1

Source: data/seed/DESIGN.md (Step 0.7); Phase 2 schema selection replaces this content.

CREATE TABLE biz.regions (
    region_id INTEGER CONSTRAINT pk_regions PRIMARY KEY,
    name VARCHAR(40) NOT NULL CONSTRAINT uq_regions_name UNIQUE,
    name_en VARCHAR(80) NOT NULL CONSTRAINT uq_regions_name_en UNIQUE,
    renamed_from VARCHAR(40),
    effective_from TIMESTAMPTZ NOT NULL,
    CONSTRAINT ck_regions_rename CHECK (renamed_from IS NULL OR renamed_from <> name)
);

CREATE TABLE biz.promotions (
    promo_id INTEGER CONSTRAINT pk_promotions PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    kind VARCHAR(20) NOT NULL,
    starts_at TIMESTAMPTZ NOT NULL,
    ends_at TIMESTAMPTZ NOT NULL,
    rule_doc_ref VARCHAR(200),
    CONSTRAINT ck_promotions_kind CHECK (kind IN ('discount','seasonal','shipping')),
    CONSTRAINT ck_promotions_interval CHECK (starts_at < ends_at)
);

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
);

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
);

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
);

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
);

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
);

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
);
