"""Fixed synthetic business texts; these are capacity inputs, not retrieval-quality labels."""

import hashlib

from spikes.capacity.contracts import Pair

TOPICS = (
    "签收后七天内未使用的商品可以申请无理由退货,生鲜及定制商品除外。",
    "退款申请时间与订单支付时间分别归属各自的业务月份,取消订单不计入销售额。",
    "华东地区八月促销期间物流延迟,签收后退货流程和运费规则保持不变。",
    "SKU-A1023 商品需保留原始包装,质量问题经核验后退货运费由商家承担。",
    "会员积分与优惠券不可兑换现金,退款金额以实际支付金额为准。",
)


def text_at(index: int) -> str:
    """Long enough to exercise actual 512-token truncation instead of short smoke input."""
    return f"业务记录 {index:05d}。" + TOPICS[index % len(TOPICS)] * 40


def fixed_pairs() -> list[Pair]:
    """Fifty varied deterministic pairs; reranking tokenizes to a maximum of 320."""
    queries = (
        "七天退货有哪些条件?",
        "退款率按哪个时间计算?",
        "华东八月物流怎么样?",
        "质量问题谁付运费?",
        "优惠券能换现金吗?",
    )
    return [Pair(query=queries[i % 5], passage=text_at(i)) for i in range(50)]


def corpus_hash(size: int) -> str:
    """Hash the exact ordered strings without storing sensitive user data."""
    return hashlib.sha256("\n".join(text_at(i) for i in range(size)).encode()).hexdigest()
