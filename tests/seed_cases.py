"""Frozen SQL acceptance cases, independent of local development documents."""

from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError
from sqlglot import exp, parse
from sqlglot.errors import ParseError

from app.core.errors import ValidationError

CASE_FILE = Path(__file__).parent / "fixtures" / "seed_traps.json"
TRAP_IDS = frozenset(f"T{number}" for number in range(1, 9))
EvalCase = Annotated[str, Field(pattern=r"^nl2sql-[a-z]+-\d{3}$")]
Delta = Annotated[Decimal, Field(ge=Decimal("0.05"), allow_inf_nan=False)]


class SeedTrap(BaseModel):
    """One immutable SQL pair and its independently authored acceptance interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^T[1-8]$")
    naive_sql: str = Field(min_length=1)
    correct_sql: str = Field(min_length=1)
    lower_delta: Delta
    upper_delta: Delta
    eval_cases: tuple[EvalCase, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        """Reject malformed SQL, repeated cases and invalid numerical intervals."""
        if self.lower_delta > self.upper_delta:
            raise ValidationError("Seed acceptance bounds are reversed")
        if len(set(self.eval_cases)) != len(self.eval_cases):
            raise ValidationError("Seed evaluation identifiers must be unique")
        if self.naive_sql == self.correct_sql:
            raise ValidationError("Seed query pairs must differ")
        for query in (self.naive_sql, self.correct_sql):
            _validate_query(query)
        return self


def _validate_query(query: str) -> None:
    try:
        statements = parse(query, read="postgres")
    except ParseError as error:
        raise ValidationError("Seed query cannot be parsed") from error
    if (
        not query.rstrip().endswith(";")
        or len(statements) != 1
        or not isinstance(statements[0], exp.Query)
        or any(isinstance(node, (exp.DDL, exp.DML)) for node in statements[0].walk())
    ):
        raise ValidationError("Seed query must be one complete read-only query")


class SeedCases(BaseModel):
    """Complete frozen coverage of the eight business acceptance scenarios."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    traps: tuple[SeedTrap, ...]

    @model_validator(mode="after")
    def complete(self) -> Self:
        """Require each scenario once and disjoint evaluation identifiers."""
        if len(self.traps) != len(TRAP_IDS) or {trap.id for trap in self.traps} != TRAP_IDS:
            raise ValidationError("Seed cases must contain T1 through T8 exactly once")
        cases = [case for trap in self.traps for case in trap.eval_cases]
        if len(cases) != len(set(cases)):
            raise ValidationError("Seed evaluation identifiers cannot span multiple traps")
        return self


def load_cases(path: Path = CASE_FILE) -> SeedCases:
    """Read and validate a complete resource; missing evidence never becomes a skip."""
    try:
        return SeedCases.model_validate_json(path.read_bytes())
    except (OSError, PydanticValidationError) as error:
        raise ValidationError("Seed case resource is missing or invalid") from error
