"""Conservative lexical shortcuts; uncertain questions always reach the classifier."""

import re

from app.agents.contracts import Route, RouteDecision, RoutingContext
from app.schemas.clarification import ClarificationCategory, ClarificationIntent, MissingDimension

CLARIFICATION_QUESTION = "请说明要查询的业务指标或政策，以及相关时间和地区。"
_WHY = re.compile(r"为什么|为何|原因|\bwhy\b", re.IGNORECASE)
_POLICY = re.compile(
    r"政策|规则|规定|流程|制度|\b(?:policy|policies|rules?|procedures?)\b", re.IGNORECASE
)
_DEFINITION = re.compile(
    r"定义|口径|怎么算|怎么计算|如何计算|计算方法|是什么|含义|意思|解释|\b(?:definition|define|meaning|explain|calculated)\b",
    re.IGNORECASE,
)
_REFERENCE = re.compile(
    r"那个|这个|上次|刚才|之前|上述|那.{0,8}呢|\b(?:that|it|previous|above|those)\b", re.IGNORECASE
)
_BARE_REFERENCE = re.compile(
    r"(?:请|帮我|再|看看|看一下|处理一下|分析一下|昨天|上次|刚才|之前|的|问题|那个|这个|吧|呢|\s|[\uff1f?\uff01!。.]|that|it|one|please|again)+",
    re.IGNORECASE,
)
_PERIOD = re.compile(
    r"(?<!\d)(?:20\d{2}年)?(?:1[0-2]|[1-9])月|20\d{2}-(?:0[1-9]|1[0-2])(?:\b|-)"
    r"|(?:上|本|这|下)(?:个)?(?:月|季度|周)|(?:今|去|明)年|今天|昨天|前天"
    r"|(?:最近|过去)\d+(?:天|周|个月)|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"|\b(?:last|this|next)\s+(?:month|quarter|week|year)\b|\b(?:today|yesterday|20\d{2})\b",
    re.IGNORECASE,
)
_METRICS = (
    ("gmv", re.compile(r"(?<![A-Za-z0-9_])GMV(?![A-Za-z0-9_])|商品交易总额", re.IGNORECASE)),
    ("order_count", re.compile(r"订单数|订单量|\border\s+count\b", re.IGNORECASE)),
    ("aov", re.compile(r"客单价|\baov\b|\baverage\s+order\s+value\b", re.IGNORECASE)),
    ("active_customer", re.compile(r"活跃客户数|\bactive\s+customers?\b", re.IGNORECASE)),
    ("refund_rate", re.compile(r"退款率|退款申请订单率|\brefund\s+rate\b", re.IGNORECASE)),
    ("refund_count", re.compile(r"退款笔数|\brefund\s+count\b", re.IGNORECASE)),
)


def prefilter(question: str, context: RoutingContext | None = None) -> RouteDecision | None:
    """Only classify explicit standalone patterns; history is never guessed."""
    question = question.strip()
    if _WHY.search(question):
        return None
    has_history = context is not None and (
        bool(context.summary.strip())
        or any(item.content.strip() for item in context.recent_messages)
    )
    if not question or (_REFERENCE.search(question) and _BARE_REFERENCE.fullmatch(question)):
        if question and has_history:
            return None
        return RouteDecision(
            route=Route.CLARIFY,
            confidence=0.9,
            decided_by="prefilter",
            clarification_question=CLARIFICATION_QUESTION,
            clarification_intent=ClarificationIntent(
                category=ClarificationCategory.AMBIGUOUS_REFERENCE,
                missing_dimensions=[MissingDimension.REFERENCE],
            ),
        )
    if _REFERENCE.search(question):
        return None
    return _standalone(question)


def _standalone(question: str) -> RouteDecision | None:
    """Recognize only explicit standalone metric or policy requests."""
    metrics = [key for key, pattern in _METRICS if pattern.search(question)]
    if metrics and _DEFINITION.search(question):
        return None
    policy = _POLICY.search(question)
    period = _PERIOD.search(question)
    if metrics and period and not policy:
        return RouteDecision(
            route=Route.DATA_ONLY,
            confidence=0.95,
            decided_by="prefilter",
            data_intent=question,
            metric_hints=metrics,
        )
    if policy and not metrics and not period:
        return RouteDecision(
            route=Route.KNOWLEDGE_ONLY,
            confidence=0.95,
            decided_by="prefilter",
            knowledge_intent=question,
        )
    return None
