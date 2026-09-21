"""Safe deterministic degradation messages and post-deadline evidence presentation."""

from app.agents.contracts import EvidenceBundle, SynthesisResult
from app.agents.failures import FailureKind, NodeFailure
from app.agents.synthesis_generation import missing_sources, synthesis_input
from app.agents.synthesis_validation import data_view, validate_output
from app.schemas.synthesis import (
    CellReference,
    Claim,
    ClaimKind,
    DataReference,
    RowCountReference,
    SynthesisOutput,
)

MAX_CELL_CHARS = 500
MAX_EXCERPT_CHARS = 2000

FAILURE_MESSAGES = {
    FailureKind.MCP_UNAVAILABLE: "数据源当前不可用，未能核对实际订单数据。",
    FailureKind.MCP_POLICY_REJECTED: "业务数据查询被权限策略拒绝。",
    FailureKind.SQL_TIMEOUT: "业务数据查询超时。",
    FailureKind.SQL_GENERATION_FAILED: "业务数据查询生成失败。",
    FailureKind.SQL_VALIDATION_FAILED: "业务数据查询未通过校验。",
    FailureKind.SQL_EXECUTION_FAILED: "业务数据查询执行失败。",
    FailureKind.SQL_CORRECTION_EXHAUSTED: "业务数据查询修正后仍未成功。",
    FailureKind.RETRIEVAL_UNAVAILABLE: "企业知识库当前不可用，未能核实适用政策和业务规则。",
    FailureKind.RETRIEVAL_NO_EVIDENCE: "企业知识库未找到支持当前问题的证据。",
    FailureKind.MODEL_RUNTIME_UNAVAILABLE: "知识编码服务当前不可用，未能完成知识检索。",
    FailureKind.DEADLINE_EXCEEDED: "分析时间已用尽，未完成的分析无法继续。",
}


def public_failure(kind: FailureKind) -> str:
    """Only enum-selected fixed text is exposed; exception details remain internal."""
    return FAILURE_MESSAGES.get(kind, "分析服务未能完成本次操作。")


def missing_explanations(bundle: EvidenceBundle, failures: list[NodeFailure]) -> list[str]:
    """Describe absent sources and their known failures without guessing from prose."""
    missing = missing_sources(bundle)
    reasons = [public_failure(failure.kind) for failure in failures]
    if "data" in missing:
        reasons.append("业务数据未提供可核验的查询结果。")
    if "knowledge" in missing:
        reasons.append("企业知识库未提供可用的政策或业务规则证据。")
    return list(dict.fromkeys(reasons))[:8]


def _data_claims(bundle: EvidenceBundle) -> list[Claim]:
    view = data_view(synthesis_input("呈现已完成证据", bundle))
    if view is None:
        return []
    claims = [
        Claim(
            text="查询返回行数；统计范围仅限本次返回结果。",
            kind=ClaimKind.FACT_DATA,
            data_refs=[RowCountReference(value=view.returned_row_count)],
            confidence=1,
        )
    ]
    for row_index, row in enumerate(view.sample_rows[:3]):
        refs: list[DataReference] = [
            CellReference(row=row_index, column=column, value=value)
            for column, value in enumerate(row[:3])
            if len(str(value)) <= MAX_CELL_CHARS
        ]
        if refs:
            claims.append(
                Claim(
                    text="查询样本（按查询结果列顺序展示）。",
                    kind=ClaimKind.FACT_DATA,
                    data_refs=refs,
                    confidence=1,
                )
            )
    return claims


def _excerpt(text: str) -> str:
    # Cut at a complete textual boundary, never halfway through a number or date.
    if len(text) <= MAX_EXCERPT_CHARS:
        return text
    prefix = text[:MAX_EXCERPT_CHARS]
    boundary = max(prefix.rfind(mark) for mark in ("。", "；", "\n", " "))
    return prefix[: boundary + 1] if boundary > 0 else "已找到适用文档，内容见对应证据快照。"


def deadline_synthesis(bundle: EvidenceBundle, failures: list[NodeFailure]) -> SynthesisResult:
    """No model call, new query, inference or invented reference during finalization."""
    claims = _data_claims(bundle)
    if bundle.knowledge is not None:
        claims.extend(
            Claim(
                text=_excerpt(chunk.generation_text),
                kind=ClaimKind.FACT_DOCUMENT,
                chunk_ids=[chunk.chunk_id],
                confidence=0.5,
            )
            for chunk in bundle.knowledge.knowledge.chunks[:3]
        )
    checked = validate_output(
        SynthesisOutput(
            claims=claims,
            unanswered=[
                "分析已超时，本次仅展示已完成的证据，尚未完成综合分析或因果核实。",
                *missing_explanations(bundle, failures),
            ],
        ),
        synthesis_input("呈现已完成证据", bundle),
    )
    return SynthesisResult(
        **checked.model_dump(),
        evidence_refs=bundle.refs,
        missing_components=missing_sources(bundle),
        attempts=0,
    )
