"""Pure final presentation: immutable evidence supplies every trusted appendix."""

from decimal import Decimal, InvalidOperation, localcontext

from app.agents.contracts import MAX_ANSWER_CHARS, Answer, EvidenceBundle
from app.agents.data.caveats import render_caveats
from app.agents.synthesis_generation import synthesis_input
from app.agents.synthesis_validation import claim_line, data_view
from app.core.errors import ContextBudgetExceeded
from app.schemas.mcp import SqlValue
from app.schemas.synthesis import Claim, ClaimKind

MAX_DISPLAY_EXPONENT = 1000

_SOURCE_NAMES = {"data": "业务数据", "knowledge": "企业知识库"}
_IMPACTS = {
    "data": "业务数据源不可用或缺少可用证据，未能核对实际订单数据。",
    "deadline": "分析已超时，仅呈现已完成的证据，尚未完成全部分析。",
    "knowledge": "知识依据不足或不可用，未能核实适用政策和业务规则。",
    "rerank": "重排服务不可用，文档依据使用未重排结果，相关性判断存在限制。",
    "memory": "偏好记忆不可用，本次未应用已保存的偏好。",
}


def confidence_for(answer: Answer, bundle: EvidenceBundle) -> float:
    """Compute source reliability; model self-ratings are not answer confidence."""
    if answer.abstained:
        return 0
    signals = []
    if bundle.data and any(claim.data_refs for claim in answer.claims):
        signals.append(max(0.0, 1 - 0.2 * len(bundle.data.data.sanity_flags)))
    if bundle.knowledge and any(claim.chunk_ids for claim in answer.claims):
        score = bundle.knowledge.knowledge.top_rerank_score
        signals.append(score if score is not None else 0.5)
    if answer.degraded_components:
        signals.append(0.5)
    if any(claim.kind is ClaimKind.INFERENCE for claim in answer.claims):
        signals.append(0.9)
    return min(signals) if signals else 0


def _display(value: SqlValue, decimals: int | None) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return str(value)
    if decimals is None:
        return str(value)
    try:
        number = Decimal(str(value))
        if not number.is_finite() or abs(number.adjusted()) > MAX_DISPLAY_EXPONENT:
            return str(value)
        with localcontext() as ctx:
            ctx.prec = max(28, len(number.as_tuple().digits) + abs(number.adjusted()) + 8)
            rounded = number.quantize(Decimal(1).scaleb(-decimals))
        rendered = format(rounded, f".{decimals}f")
        return f"≈ {rendered}（非精确零）" if number != 0 and rounded == 0 else rendered
    except InvalidOperation:
        return str(value)


def _values(claim: Claim, answer: Answer) -> str:
    places = answer.format_preference.decimals if answer.format_preference else None
    return "；".join(_display(ref.value, places) for ref in claim.data_refs)


def _cell(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )


def _claims(answer: Answer) -> list[str]:
    if answer.format_preference and answer.format_preference.prefer == "table":
        rows = [
            f"| {_cell(claim_line(claim))} | {_cell(_values(claim, answer)) or '—'} |"
            for claim in answer.claims
        ]
        return ["| 经核验的声明 | 数值展示 |\n| --- | --- |\n" + "\n".join(rows)]
    groups = (
        (ClaimKind.FACT_DATA, "数据结论"),
        (ClaimKind.FACT_DOCUMENT, "知识依据"),
        (ClaimKind.INFERENCE, "推断 (尚未建立因果关系)"),
        (ClaimKind.UNSUPPORTED, "无法证实"),
    )
    sections = []
    for kind, title in groups:
        lines = []
        for claim in answer.claims:
            if claim.kind is not kind:
                continue
            line = claim_line(claim)
            if claim.data_refs:
                line += "\n\n数值展示: " + _values(claim, answer)
            lines.append(line)
        if lines:
            sections.append(f"### {title}\n\n" + "\n\n".join(lines))
    return sections


def _fence(sql: str) -> str:
    fence = "```"
    while fence in sql:
        fence += "`"
    return f"{fence}sql\n{sql}\n{fence}"


def render_answer(answer: Answer, bundle: EvidenceBundle) -> str:
    """Keep all claims, citations and appendices or fail with a typed size error."""
    pieces = []
    if answer.degraded_components:
        sources = "、".join(
            name for present, name in (
                (bundle.data is not None, "业务数据"),
                (bundle.knowledge is not None, "企业知识库"),
            ) if present
        )
        pieces.append(
            "> ⚠️ 本次为部分回答。仅基于" + sources + "。"
            + " ".join(
                _IMPACTS.get(name, f"组件 {name} 降级，其提供的信息可能不完整。")
                for name in dict.fromkeys(answer.degraded_components)
            )
        )
    if answer.abstained:
        sources = "、".join(_SOURCE_NAMES[source] for source in answer.attempted_sources)
        pieces.append(
            f"已尝试核查{sources}，现有证据不足以形成可验证的回答。"
            if sources
            else "需要补充问题信息后才能继续分析。"
        )
    else:
        pieces.extend(_claims(answer))
    if answer.synthesis and answer.synthesis.conflicts:
        conflicts = [
            claim_line(answer.claims[item.left_claim])
            + "\n\n与\n\n"
            + claim_line(answer.claims[item.right_claim])
            for item in answer.synthesis.conflicts
        ]
        pieces.append(
            "### 证据冲突\n\n"
            + "\n\n---\n\n".join(conflicts)
            + "\n\n现有证据无法裁定，需进一步核实。"
        )
    if answer.synthesis and not answer.abstained:
        pieces.append("相关性不等于因果；以上事实尚未建立因果关系。")
    if answer.unanswered:
        if (
            answer.abstained
            and answer.format_preference
            and answer.format_preference.prefer == "table"
        ):
            pieces.append(
                "| 待补充信息 |\n| --- |\n"
                + "\n".join(f"| {_cell(text)} |" for text in answer.unanswered)
            )
        else:
            pieces.append(
                "### 缺失信息与待核实问题\n\n"
                + "\n".join(f"- {text}" for text in answer.unanswered)
            )
    if answer.citations:
        pieces.append(
            "### 引用来源\n\n"
            + "\n".join(
                f"- [{item.chunk_id}] {_cell(item.document_title)} · {_cell(item.heading_path)}"
                + (f" · 第 {item.page} 页" if item.page is not None else "")
                for item in answer.citations
            )
        )
    pieces.extend(_evidence_appendix(answer, bundle))
    markdown = "\n\n".join(pieces)
    if len(markdown) > MAX_ANSWER_CHARS:
        raise ContextBudgetExceeded()
    return markdown


def finalize_answer(answer: Answer, bundle: EvidenceBundle) -> Answer:
    """Return a fully rendered copy without mutating the source or structured claims."""
    return answer.model_copy(
        update={
            "markdown": render_answer(answer, bundle),
            "confidence": confidence_for(answer, bundle),
        }
    )


def _evidence_appendix(answer: Answer, bundle: EvidenceBundle) -> list[str]:
    pieces: list[str] = []
    if bundle.data:
        view = data_view(synthesis_input("render evidence scope", bundle))
        pieces.append(
            f"统计仅覆盖本次查询返回的 {bundle.data.data.row_count} 行，"
            "不能据此推断未返回的总体；如查询指定 top-N，则仅描述该范围。"
        )
        if view is not None and view.sample_truncated:
            pieces.append("用于本次回答的样本已截断；列统计基于全部返回行，不是从样本重新计算。")
        caveats = render_caveats(bundle.data.data.sanity_flags)
        if caveats:
            pieces.append(caveats.strip())
    if answer.assumptions or bundle.data:
        assumptions = answer.assumptions or ["证据快照未记录额外统计口径。"]
        pieces.append("---\n**统计口径**\n" + "\n".join(f"- {text}" for text in assumptions))
    if bundle.data:
        pieces.append("**执行的查询**\n" + _fence(bundle.data.data.sql))
    return pieces
