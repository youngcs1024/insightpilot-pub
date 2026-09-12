"""Independent metric SQL oracle; deliberately does not reuse production templates."""

# ruff: noqa: S608 -- independently authored test SQL with enum grains and fixture timestamps.

from app.schemas.metrics import Grain


def expected_sql(key: str, grain: Grain, start: str, end: str) -> str:
    """Express reference results using order/item CTEs and EXISTS for refunds."""
    paid = "paid_at >= TIMESTAMPTZ '" + start + "' AND paid_at < TIMESTAMPTZ '" + end + "'"
    requested = (
        "r.requested_at >= TIMESTAMPTZ '"
        + start
        + "' AND r.requested_at < TIMESTAMPTZ '"
        + end
        + "'"
    )

    def bucket(date: str) -> str:
        if grain == Grain.TOTAL:
            return "'total'"
        if grain == Grain.REGION:
            return "region_id"
        if grain == Grain.CATEGORY:
            return "category"
        return "date_trunc('" + grain.value + "', " + date + " AT TIME ZONE 'Asia/Shanghai')"

    grouping = " GROUP BY 1" if grain != Grain.TOTAL else ""
    qualified = """WITH eligible AS (
      SELECT o.* FROM biz.orders o
      WHERE o.paid_at IS NOT NULL AND o.status <> 'cancelled'
        AND EXISTS (SELECT 1 FROM biz.customers c WHERE c.customer_id=o.customer_id AND NOT c.is_test_account)
    )"""
    if key == "refund_rate":
        # Independently deduplicate (grain, order) before counting, instead of COUNT DISTINCT on a fanout join.
        numerator = (
            "SELECT DISTINCT e.order_id, "
            + bucket("r.requested_at")
            + " AS bucket FROM eligible e JOIN biz.refunds r ON r.order_id=e.order_id WHERE r.status <> 'rejected' AND "
            + requested
        )
        n = (
            "SELECT "
            + ("'total'" if grain == Grain.TOTAL else "bucket")
            + " AS grain, count(*)::numeric AS n FROM requested"
            + grouping
        )
        d = (
            "SELECT "
            + bucket("paid_at")
            + " AS grain, count(*) AS d FROM eligible WHERE "
            + paid
            + grouping
        )
        return (
            qualified
            + ", requested AS ("
            + numerator
            + "), n AS ("
            + n
            + "), d AS ("
            + d
            + ") SELECT COALESCE(n.grain,d.grain), COALESCE(n.n,0)/NULLIF(d.d,0) FROM n FULL JOIN d USING(grain) ORDER BY 1"
        )
    if key == "refund_count":
        return (
            qualified
            + " SELECT "
            + bucket("r.requested_at")
            + ", COUNT(*) FROM eligible e JOIN biz.refunds r ON e.order_id=r.order_id WHERE r.status='completed' AND "
            + requested
            + grouping
            + " ORDER BY 1"
        )
    amount = "gross_amount-discount_amount"
    source = "eligible e JOIN biz.customers c USING(customer_id)"
    if grain == Grain.CATEGORY:
        source += " JOIN biz.order_items i USING(order_id) JOIN biz.products p USING(product_id)"
        amount = "i.quantity*i.unit_price-CASE WHEN i.item_discount IS NULL THEN 0 ELSE i.item_discount END"
    expressions = {
        "gmv": "SUM(" + amount + ")",
        "order_count": "COUNT(DISTINCT e.order_id)",
        "aov": "SUM(" + amount + ") / NULLIF(COUNT(DISTINCT e.order_id),0)",
        "active_customer": "COUNT(DISTINCT c.phone)",
    }
    # Explicit sale-region alias avoids the customer's current-region field.
    group = "e.region_id" if grain == Grain.REGION else bucket("paid_at")
    return (
        qualified
        + " SELECT "
        + group
        + ", "
        + expressions[key]
        + " FROM "
        + source
        + " WHERE "
        + paid
        + grouping
        + " ORDER BY 1"
    )
