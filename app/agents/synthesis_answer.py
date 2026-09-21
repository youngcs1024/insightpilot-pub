"""Deterministic presentation and commit validation of synthesis results."""

from app.agents.contracts import MAX_ANSWER_CHARS, Answer, EvidenceBundle, SynthesisResult
from app.agents.data.caveats import append_caveats
from app.agents.synthesis_generation import missing_sources, synthesis_input
from app.agents.synthesis_validation import citations_for, claim_line, validate_output
from app.core.errors import ConflictError, ContextBudgetExceeded
from app.schemas.knowledge import KnowledgePassage
from app.schemas.memory import FormatPreferenceContent
from app.schemas.synthesis import ClaimKind, SynthesisOutput


def validate_result(result: SynthesisResult, bundle: EvidenceBundle) -> None:
    """Recheck all durable content before committing or rendering an answer."""
    if result.evidence_refs != bundle.refs or result.missing_components != missing_sources(bundle):
        raise ConflictError("synthesis references differ from snapshots")
    if result.abstention is not None:
        if result.claims or result.conflicts or result.summary:
            raise ConflictError("abstention retains rejected synthesis")
        return
    draft = SynthesisOutput.model_validate(
        result.model_dump(include=set(SynthesisOutput.model_fields))
    )
    checked = validate_output(draft, synthesis_input("validate committed answer", bundle))
    if checked != draft or not any(
        claim.kind is not ClaimKind.UNSUPPORTED and (claim.data_refs or claim.chunk_ids)
        for claim in draft.claims
    ):
        raise ConflictError("synthesis was not validated")


def _claim_sections(
    result: SynthesisResult, preference: FormatPreferenceContent | None
) -> list[str]:
    if preference is not None and preference.prefer == "table":
        rows = [
            line.replace("|", "\\|").replace("\n", "<br>") for line in result.summary.split("\n\n")
        ]
        return ["| 经核验的声明 |\n| --- |\n" + "\n".join(f"| {line} |" for line in rows)]
    groups = (
        (ClaimKind.FACT_DATA, "数据结论"),
        (ClaimKind.FACT_DOCUMENT, "知识依据"),
        (ClaimKind.INFERENCE, "推断 (尚未建立因果关系)"),
        (ClaimKind.UNSUPPORTED, "无法证实"),
    )
    return [
        f"### {title}\n\n"
        + "\n\n".join(claim_line(claim) for claim in result.claims if claim.kind is kind)
        for kind, title in groups
        if any(claim.kind is kind for claim in result.claims)
    ]


def _markdown(
    result: SynthesisResult, preference: FormatPreferenceContent | None, degraded: list[str]
) -> str:
    if result.abstention is not None:
        pieces = ["现有证据不足以形成可验证的综合回答。"]
    else:
        pieces = _claim_sections(result, preference)
        if result.conflicts:
            pairs = [
                claim_line(result.claims[item.left_claim])
                + "\n\n与\n\n"
                + claim_line(result.claims[item.right_claim])
                for item in result.conflicts
            ]
            pieces.append(
                "### 证据冲突\n\n"
                + "\n\n---\n\n".join(pairs)
                + "\n\n现有证据无法裁定，需进一步核实。"
            )
        pieces.append("相关性不等于因果；以上事实尚未建立因果关系。")
    missing = [
        "业务数据未能核实" if name == "data" else "知识依据不足或不可用"
        for name in result.missing_components
    ]
    unanswered = [*missing, *result.unanswered]
    if unanswered:
        pieces.append(
            "### 缺失信息与待核实问题\n\n" + "\n".join(f"- {text}" for text in unanswered)
        )
    if degraded:
        names = {"data": "业务数据", "knowledge": "知识依据", "rerank": "重排服务"}
        pieces.insert(
            0,
            "> 本次为部分回答，以下组件缺失或降级: "
            + "、".join(names.get(name, name) for name in degraded)
            + "。",
        )
    return "\n\n".join(pieces)


def synthesis_answer(
    result: SynthesisResult,
    bundle: EvidenceBundle,
    degraded_components: list[str],
    preference: FormatPreferenceContent | None = None,
) -> Answer:
    """No model call can replace a checked claim with unchecked prose."""
    validate_result(result, bundle)
    degraded = list(dict.fromkeys([*degraded_components, *result.missing_components]))
    markdown = _markdown(result, preference, degraded)
    if bundle.data:
        markdown = append_caveats(markdown, bundle.data.data.sanity_flags)
    if len(markdown) > MAX_ANSWER_CHARS:
        raise ContextBudgetExceeded()
    signals = [claim.confidence for claim in result.claims]
    if bundle.knowledge and any(claim.chunk_ids for claim in result.claims):
        score = bundle.knowledge.knowledge.top_rerank_score
        signals.append(score if score is not None else 0.5)
    if any(claim.kind is ClaimKind.INFERENCE for claim in result.claims):
        signals.append(0.9)
    if degraded:
        signals.append(0.5)
    return Answer(
        markdown=markdown,
        confidence=min(signals) if signals and result.abstention is None else 0,
        sql=bundle.data.data.sql if bundle.data else "",
        assumptions=synthesis_input("answer assumptions", bundle).assumptions,
        evidence_refs=bundle.refs,
        citations=citations_for(result, bundle),
        knowledge_passages=[
            KnowledgePassage(text=claim_line(claim), chunk_ids=tuple(claim.chunk_ids))
            for claim in result.claims
            if claim.chunk_ids
        ],
        degraded_components=degraded,
        abstained=result.abstention is not None,
        synthesis=result,
    )


def validate_synthesis_answer(answer: Answer, bundle: EvidenceBundle) -> None:
    """The persisted prose and confidence must also be deterministic renderings."""
    if answer.synthesis is None:
        raise ConflictError("missing synthesis")
    for preference in (None, FormatPreferenceContent(prefer="table", decimals=2)):
        expected = synthesis_answer(
            answer.synthesis, bundle, answer.degraded_components, preference
        )
        if answer == expected:
            return
    raise ConflictError("answer differs from validated synthesis")
