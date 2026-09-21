"""Pure evidence validation shared by generation and the final commit boundary."""

import re
from decimal import Decimal

from pydantic import ValidationError as PydanticValidationError

from app.agents.contracts import MAX_ANSWER_CHARS, EvidenceBundle, SynthesisInput
from app.core.errors import (
    ContextBudgetExceeded,
    FabricatedCitation,
    SynthesisEvidenceError,
    SynthesisValidationError,
)
from app.retrieval.evidence import render_documents
from app.schemas.knowledge import Citation
from app.schemas.mcp import SqlValue
from app.schemas.synthesis import (
    CellReference,
    Claim,
    ClaimKind,
    DataGenerationView,
    DataReference,
    RowCountReference,
    SynthesisOutput,
)

CAUSAL_MARKERS = re.compile(
    r"导致|因为|造成|引起|源于|归因于|促使|caused?\s+by|causes?\b|because\b|"
    r"result(?:s|ed)?\s+in|(?:leads?|led)\s+to|due\s+to|attribut(?:ed|able)\s+to",
    re.IGNORECASE,
)
NUMBERS = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
LABELS = {
    ClaimKind.FACT_DATA: "数据事实",
    ClaimKind.FACT_DOCUMENT: "文档事实",
    ClaimKind.INFERENCE: "推断，尚未证实因果关系",
    ClaimKind.UNSUPPORTED: "无法证实",
}


def data_view(source: SynthesisInput) -> DataGenerationView | None:
    """Treat malformed persisted evidence as a storage failure, not a model repair."""
    if source.knowledge is not None and source.knowledge.generation_block != render_documents(
        source.knowledge.chunks
    ):
        raise SynthesisEvidenceError()
    if source.data is None:
        return None
    try:
        return DataGenerationView.model_validate_json(source.data.generation_block)
    except PydanticValidationError as exc:
        raise SynthesisEvidenceError() from exc


def reference_value(reference: DataReference, view: DataGenerationView | None) -> SqlValue:
    """Positions are bounded by the model view, never by the larger stored rows."""
    if view is None:
        raise SynthesisValidationError()
    try:
        if isinstance(reference, RowCountReference):
            value: SqlValue = view.returned_row_count
        elif isinstance(reference, CellReference):
            value = view.sample_rows[reference.row][reference.column]
        else:
            index = view.statistics_fields.index(reference.field)
            value = view.statistics[reference.column][index]
    except (IndexError, ValueError) as exc:
        raise SynthesisValidationError() from exc
    # In particular, True is not the number 1 and NULL is not zero.
    if type(value) is not type(reference.value) or value != reference.value:
        raise SynthesisValidationError()
    return value


def _numbers(text: str) -> set[Decimal]:
    values = set()
    for match in NUMBERS.findall(text):
        value = Decimal(match.rstrip("%").replace(",", ""))
        values.add(value / 100 if match.endswith("%") else value)
    return values


def _validate_claim(claim: Claim, source: SynthesisInput, view: DataGenerationView | None) -> None:
    allowed = (
        {chunk.chunk_id: chunk for chunk in source.knowledge.chunks} if source.knowledge else {}
    )
    if any(identifier not in allowed for identifier in claim.chunk_ids):
        raise FabricatedCitation()
    if not claim.text.strip():
        raise SynthesisValidationError()
    if claim.kind is ClaimKind.FACT_DATA and not claim.data_refs:
        raise SynthesisValidationError()
    if claim.kind is ClaimKind.FACT_DOCUMENT and not claim.chunk_ids:
        raise SynthesisValidationError()
    values = [reference_value(reference, view) for reference in claim.data_refs]
    if claim.kind is not ClaimKind.UNSUPPORTED:
        support = " ".join(
            [
                *(str(value) for value in values),
                *(
                    allowed[key].generation_text
                    for key in claim.chunk_ids
                    if claim.kind is not ClaimKind.FACT_DATA
                ),
            ]
        )
        if not _numbers(claim.text).issubset(_numbers(support)):
            raise SynthesisValidationError()


def claim_line(claim: Claim) -> str:
    """Labels remain visible even when text contains causal wording."""
    refs = " ".join(f"[{identifier}]" for identifier in dict.fromkeys(claim.chunk_ids))
    return f"【{LABELS[claim.kind]}】{claim.text}" + (f" {refs}" if refs else "")


def summary(claims: list[Claim]) -> str:
    """There is no separate unvalidated prose channel in the final summary."""
    return "\n\n".join(claim_line(claim) for claim in claims)


def validate_output(output: SynthesisOutput, source: SynthesisInput) -> SynthesisOutput:
    """Reject invalid references before downgrading facts and replacing the summary."""
    view = data_view(source)
    claims = []
    for claim in output.claims:
        _validate_claim(claim, source, view)
        kind = claim.kind
        if kind in {ClaimKind.FACT_DATA, ClaimKind.FACT_DOCUMENT} and CAUSAL_MARKERS.search(
            claim.text
        ):
            kind = ClaimKind.INFERENCE
        claims.append(claim.model_copy(deep=True, update={"kind": kind}))
    for conflict in output.conflicts:
        indices = (conflict.left_claim, conflict.right_claim)
        if indices[0] == indices[1] or any(index >= len(claims) for index in indices):
            raise SynthesisValidationError()
        if any(not (claims[index].data_refs or claims[index].chunk_ids) for index in indices):
            raise SynthesisValidationError()
    rendered = summary(claims)
    if len(rendered) > MAX_ANSWER_CHARS:
        raise ContextBudgetExceeded()
    unanswered = list(output.unanswered)
    if not unanswered and any(claim.kind is ClaimKind.INFERENCE for claim in claims):
        unanswered.append("需要能够核实上述推断的归因数据与适用政策依据。")
    return SynthesisOutput(
        claims=claims,
        conflicts=output.conflicts,
        unanswered=unanswered,
        summary=rendered,
    )


def citations_for(output: SynthesisOutput, bundle: EvidenceBundle) -> list[Citation]:
    """Only selected immutable chunks can supply display metadata."""
    allowed = (
        {chunk.chunk_id: chunk for chunk in bundle.knowledge.knowledge.chunks}
        if bundle.knowledge
        else {}
    )
    identifiers = dict.fromkeys(key for claim in output.claims for key in claim.chunk_ids)
    if any(identifier not in allowed for identifier in identifiers):
        raise FabricatedCitation()
    return [
        Citation.model_validate(allowed[key].model_dump(include=set(Citation.model_fields)))
        for key in identifiers
    ]
