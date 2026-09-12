"""Fixed synthetic model workload; not a retrieval-quality evaluation dataset."""

import hashlib
import json

from pydantic import BaseModel

TOPICS = (
    "签收后七天内未使用的商品可以申请无理由退货,生鲜及定制商品除外。",
    "退款申请时间与订单支付时间分别归属各自的业务月份,取消订单不计入销售额。",
    "华东地区八月促销期间物流延迟,签收后退货流程和运费规则保持不变。",
    "SKU-A1023 商品需保留原始包装,质量问题经核验后退货运费由商家承担。",
    "会员积分与优惠券不可兑换现金,退款金额以实际支付金额为准。",
)
QUERIES = (
    "七天退货有哪些条件?",
    "退款率按哪个时间计算?",
    "华东八月物流怎么样?",
    "质量问题谁付运费?",
    "优惠券能换现金吗?",
)


class Pair(BaseModel):
    """Stable input order and text identity for sequential precision comparisons."""

    query: str
    passage: str


def pairs() -> list[Pair]:
    """Fifty varied pairs with passages long enough to exercise truncation."""
    return [
        Pair(query=QUERIES[i % 5], passage=f"业务记录 {i:05d}。" + TOPICS[i % 5] * 40)
        for i in range(50)
    ]


def input_hash() -> str:
    """Hash exact ordered UTF-8 input instead of trusting a precision label."""
    body = json.dumps([pair.model_dump() for pair in pairs()], ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()
