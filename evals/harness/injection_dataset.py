"""Closed red-team case contract, independent of routing accuracy cases."""

from enum import StrEnum
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from app.core.errors import SchemaMetadataError
from app.schemas.mcp import Contract
from data.seed.schema_metadata_loader import check_keys
from evals.harness.contracts import EvaluationError

CASES = Path(__file__).resolve().parents[1] / "datasets/routing/injection.yaml"


class InjectionKind(StrEnum):
    """Exactly the eight Step 5.10 threat scenarios."""

    SQL_COMMAND = "sql_command"
    CREDENTIAL = "credential"
    RETRIEVED_INSTRUCTION = "retrieved_instruction"
    FALSE_POLICY = "false_policy"
    FABRICATED_CITATION = "fabricated_citation"
    UNSUPPORTED_CAUSALITY = "unsupported_causality"
    CROSS_USER = "cross_user"
    EXPLICIT_WIDEN = "explicit_widen"


class InjectionCase(Contract):
    """One authored attack and its required external-content precondition."""

    id: str = Field(pattern=r"^injection-[a-z-]+$")
    kind: InjectionKind
    question: str = Field(min_length=1, max_length=2000)
    followup: str | None = Field(default=None, min_length=1, max_length=2000)
    poison_source: str | None = Field(default=None, pattern=r"^[a-z_]+\.md$")

    @model_validator(mode="after")
    def expected_fields(self) -> "InjectionCase":
        """Require a real second turn and source for their respective attacks."""
        if (self.followup is not None) != (self.kind is InjectionKind.EXPLICIT_WIDEN):
            raise EvaluationError("Invalid injection follow-up")
        if (self.poison_source is not None) != (
            self.kind in {InjectionKind.RETRIEVED_INSTRUCTION, InjectionKind.FALSE_POLICY}
        ):
            raise EvaluationError("Invalid injection source")
        return self


def load_injections(path: Path = CASES) -> list[InjectionCase]:
    """A missing threat or duplicated input invalidates the resistance rate."""
    try:
        source = path.read_text(encoding="utf-8")
        node = yaml.compose(source)
        if node is not None:
            check_keys(node)
        cases = TypeAdapter(list[InjectionCase]).validate_python(yaml.safe_load(source))
    except (OSError, yaml.YAMLError, ValidationError, SchemaMetadataError) as exc:
        raise EvaluationError("Invalid injection dataset") from exc
    if (
        len(cases) != len(InjectionKind)
        or {case.kind for case in cases} != set(InjectionKind)
        or len({case.id for case in cases}) != len(cases)
        or len({case.question for case in cases}) != len(cases)
    ):
        raise EvaluationError("Incomplete or duplicate injection dataset")
    return cases
