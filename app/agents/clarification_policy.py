"""Deterministic clarification questions and transparent, unexecuted alternatives."""

from datetime import datetime

from app.agents.budget import bounded_text
from app.agents.contracts import Route
from app.agents.nodes.prefilter import prefilter
from app.agents.state import AgentState
from app.schemas.clarification import (
    ClarificationCapabilities,
    ClarificationCategory,
    ClarificationHistory,
    ClarificationIntent,
    MissingDimension,
)
from app.schemas.corpus import DocumentType
from app.schemas.metric_resolution import ClarificationKind, MetricClarification
from app.services.knowledge_time import parse_time
from app.services.periods import Period, resolve_period

_LOOP_THRESHOLD = 2

_KIND_DIMENSIONS = {
    ClarificationKind.REFERENCE_UNRESOLVED: MissingDimension.REFERENCE,
    ClarificationKind.METRIC_NOT_IDENTIFIED: MissingDimension.METRIC,
    ClarificationKind.METRIC_NOT_FOUND: MissingDimension.METRIC,
    ClarificationKind.PERIOD_UNRESOLVED: MissingDimension.PERIOD,
    ClarificationKind.UNSUPPORTED_GRAIN: MissingDimension.GRAIN,
    ClarificationKind.INVALID_EXPLICIT_PATCH: MissingDimension.DEFINITION,
    ClarificationKind.REGION_UNRESOLVED: MissingDimension.REGION,
}
_DOCUMENT_NAMES = {
    DocumentType.POLICY: "退货退款等业务政策",
    DocumentType.PROMO_RULE: "促销活动规则",
    DocumentType.METRIC_MEMO: "业务指标口径说明",
    DocumentType.SOP: "操作流程",
    DocumentType.REGION_RULE: "区域经营规则",
    DocumentType.ANALYSIS_NOTE: "经营分析记录",
}
_SCOPE_QUESTIONS = {
    MissingDimension.METRIC: "需要统计哪个业务指标",
    MissingDimension.PERIOD: "需要统计哪个完整时间段",
    MissingDimension.REGION: "需要查询哪个区域（也可明确选择全部区域）",
    MissingDimension.GRAIN: "需要按什么粒度统计（也可选择汇总）",
    MissingDimension.DEFINITION: "需要采用什么指标口径（也可选择公司口径）",
    MissingDimension.REFERENCE: "所指的是哪个具体问题",
}


def _category(kind: ClarificationKind) -> ClarificationCategory:
    if kind is ClarificationKind.REFERENCE_UNRESOLVED:
        return ClarificationCategory.AMBIGUOUS_REFERENCE
    if kind in {ClarificationKind.METRIC_NOT_FOUND, ClarificationKind.OUT_OF_SCOPE}:
        return ClarificationCategory.OUT_OF_SCOPE
    return ClarificationCategory.AMBIGUOUS_SCOPE


def _specialist_intent(state: AgentState, source: MetricClarification) -> ClarificationIntent:
    dimensions = [_KIND_DIMENSIONS[source.kind]] if source.kind in _KIND_DIMENSIONS else []
    if state.knowledge_clarification is not None:
        kind = ClarificationKind(state.knowledge_clarification.kind.value)
        dimensions = list(dict.fromkeys([*dimensions, _KIND_DIMENSIONS[kind]]))
    return ClarificationIntent(
        category=_category(source.kind), missing_dimensions=dimensions,
        metric_keys=[source.metric_key] if source.metric_key else [],
        subject=bounded_text(state.question, 1000),
    )


def source_clarification(state: AgentState) -> tuple[MetricClarification, ClarificationIntent]:
    """Map granular specialist reasons without inspecting their message text."""
    source = state.data_clarification
    if source is None and state.knowledge_clarification is not None:
        source = MetricClarification(
            kind=ClarificationKind(state.knowledge_clarification.kind.value),
            message=state.knowledge_clarification.message,
        )
    if source is not None:
        return source, _specialist_intent(state, source)
    if state.route is not None and state.route.clarification_intent is not None:
        intent = state.route.clarification_intent.model_copy(deep=True)
    else:
        # Historical/scripted v1 decisions had only a reference question.
        intent = ClarificationIntent(
            category=ClarificationCategory.AMBIGUOUS_REFERENCE,
            missing_dimensions=[MissingDimension.REFERENCE],
        )
    kind = {
        ClarificationCategory.AMBIGUOUS_REFERENCE: ClarificationKind.REFERENCE_UNRESOLVED,
        ClarificationCategory.AMBIGUOUS_SCOPE: ClarificationKind.METRIC_NOT_IDENTIFIED,
        ClarificationCategory.OUT_OF_SCOPE: ClarificationKind.OUT_OF_SCOPE,
    }[intent.category]
    if intent.missing_dimensions == [MissingDimension.PERIOD]:
        kind = ClarificationKind.PERIOD_UNRESOLVED
    return MetricClarification(kind=kind, message="pending"), intent


def _metric(intent: ClarificationIntent, capabilities: ClarificationCapabilities) -> str:
    names = {item.key: item.display_name for item in capabilities.metrics}
    selected = [names[key] for key in intent.metric_keys if key in names]
    return "、".join(selected) or names.get("gmv", next(iter(names.values()), ""))


def _example(
    intent: ClarificationIntent, capabilities: ClarificationCapabilities, period: Period
) -> str:
    metric = _metric(intent, capabilities)
    if metric:
        when = intent.period_expression or period.label
        return f"查询{when}的{metric}，使用公司口径。"
    if capabilities.document_categories:
        return f"查询{_DOCUMENT_NAMES[capabilities.document_categories[0]]}及其适用条件。"
    return "请先配置并发布业务指标或导入知识文档，然后重新提交问题。"


def _scope_suggestion(
    intent: ClarificationIntent, capabilities: ClarificationCapabilities, period: Period
) -> str:
    additions = {
        MissingDimension.PERIOD: f"统计期间改为{period.label}",
        MissingDimension.METRIC: f"指标选择{_metric(intent, capabilities)}",
        MissingDimension.REGION: "区域选择全部区域",
        MissingDimension.GRAIN: "统计粒度选择汇总",
        MissingDimension.DEFINITION: "指标口径选择公司口径",
    }
    dimensions = intent.missing_dimensions or [MissingDimension.METRIC, MissingDimension.PERIOD]
    changes = [additions[item] for item in dimensions if item in additions]
    if not intent.subject.strip() or not _metric(intent, capabilities):
        return _example(intent, capabilities, period)
    return intent.subject + "；其余明确条件保留，建议" + "，".join(changes) + "。"


def _supported_topic(topic: str, capabilities: ClarificationCapabilities, period: Period) -> bool:
    decision = prefilter(topic)
    if decision is None:
        return False
    if decision.route is Route.DATA_ONLY:
        return set(decision.metric_hints) <= {item.key for item in capabilities.metrics}
    if decision.route is Route.KNOWLEDGE_ONLY and capabilities.document_categories:
        return parse_time(topic, now=period.end).clarification is None
    return False


def _suggestion(
    intent: ClarificationIntent,
    capabilities: ClarificationCapabilities,
    history: ClarificationHistory,
    period: Period,
) -> str:
    if intent.category is ClarificationCategory.AMBIGUOUS_SCOPE:
        return _scope_suggestion(intent, capabilities, period)
    if intent.category is ClarificationCategory.AMBIGUOUS_REFERENCE:
        for topic in history.recent_topics:
            if _supported_topic(topic, capabilities, period):
                return topic
    return _example(intent, capabilities, period)


def _capabilities(capabilities: ClarificationCapabilities) -> str:
    metrics = "、".join(item.display_name for item in capabilities.metrics) or "暂无已发布指标"
    documents = "、".join(_DOCUMENT_NAMES[item] for item in capabilities.document_categories)
    return f"可选指标：{metrics}。文档类别：{documents or '暂无已发布知识文档'}。"


def _question(
    intent: ClarificationIntent, history: ClarificationHistory, period: Period
) -> str:
    if intent.category is ClarificationCategory.OUT_OF_SCOPE:
        return "这个请求超出业务数据与企业知识库的只读分析范围。"
    if intent.category is ClarificationCategory.AMBIGUOUS_REFERENCE:
        topics = "；".join(f"「{item}」" for item in history.recent_topics)
        context = f"本会话最近的问题有：{topics}。" if topics else "本会话没有可用的近期话题。"
        return f"你说的“那个”具体指哪个问题？{context}"
    dimensions = intent.missing_dimensions or [MissingDimension.METRIC, MissingDimension.PERIOD]
    question = "请明确：" + "；".join(_SCOPE_QUESTIONS[item] for item in dimensions) + "？"
    if MissingDimension.PERIOD in dimensions:
        question += "时间建议采用上一个完整自然月：" + period.as_assumption() + "。"
    return question


def render_clarification(
    state: AgentState, capabilities: ClarificationCapabilities, *, now: datetime
) -> MetricClarification:
    """Use program-owned defaults without querying data or changing explicit conditions."""
    source, intent = source_clarification(state)
    history = state.prepared.clarification_history if state.prepared else ClarificationHistory()
    period = resolve_period("上个月", now=now)
    if intent.category is ClarificationCategory.AMBIGUOUS_SCOPE:
        intent.subject = bounded_text(state.question, 1000)
    suggestion = _suggestion(intent, capabilities, history, period)
    loop = history.consecutive >= _LOOP_THRESHOLD
    opening = (
        "已连续澄清两轮，下面提供一个支持范围内的替代解释，供你确认。"
        if loop else _question(intent, history, period)
    )
    message = opening + " " + _capabilities(capabilities)
    message += " 下一步建议（尚未执行）：" + suggestion
    message += " 如接受，请在下一条消息确认；也可直接提交完整的新问题。"
    return MetricClarification(
        schema_version=2, kind=source.kind, message=message,
        category=intent.category, intent=intent, suggested_question=suggestion,
        recent_topics=list(history.recent_topics), loop_prevented=loop,
        metric_key=source.metric_key, available_metrics=[item.key for item in capabilities.metrics],
        supported_grains=list(source.supported_grains),
    )
