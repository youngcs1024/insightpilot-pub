"""Explicit single-source claim scripts for the formatter generation boundary."""

from app.agents.contracts import DataAnswerDraft
from app.schemas.mcp import SqlValue
from app.schemas.synthesis import CellReference, Claim, ClaimKind, DataReference


def data_draft(
    markdown: str = "42 orders", *, confidence: float = 0.9,
    value: SqlValue = 42, reference: DataReference | None = None,
) -> DataAnswerDraft:
    """Keep source positions and values explicit in each deterministic model script."""
    return DataAnswerDraft(claims=[Claim(
        text=markdown, kind=ClaimKind.FACT_DATA, confidence=confidence,
        data_refs=[reference or CellReference(row=0, column=0, value=value)],
    )])
