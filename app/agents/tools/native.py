"""The two model-visible native capabilities and their validated dispatcher."""

from datetime import datetime
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.agents.tools import arithmetic, periods
from app.core.errors import NativeToolError
from app.services.llm.contracts import ToolDefinition
from app.services.periods import Period

type NativeKind = Literal["periods", "arithmetic"]


class ResolvePeriodArgs(BaseModel):
    """Only the expression is model-selected; time is request-owned."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    expression: str = Field(min_length=1, max_length=500)


class ArithmeticArgs(BaseModel):
    """One arithmetic capability with an explicit operation and operand order."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    operation: Literal["percentage_change", "growth_rate", "share_of_total"]
    first: str = Field(min_length=1, max_length=100)
    second: str = Field(min_length=1, max_length=100)


_DESCRIPTIONS = {
    "resolve_period": (
        "Resolve an expression to a Shanghai half-open interval. The current time and "
        "reference period are injected by the application."
    ),
    "calculate_percentage": (
        "Calculate a percentage. For percentage_change and growth_rate, first is the "
        "baseline and second is current. For share_of_total, first is the part and "
        "second is total. Results are percentage units, e.g. 12.5 means 12.5%."
    ),
}


def definitions(kinds: list[NativeKind]) -> list[ToolDefinition]:
    """Expose only requested native capabilities, in stable order and at most two."""
    requested = set(kinds)
    entries: list[tuple[str, type[BaseModel]]] = []
    if "periods" in requested:
        entries.append(("resolve_period", ResolvePeriodArgs))
    if "arithmetic" in requested:
        entries.append(("calculate_percentage", ArithmeticArgs))
    return [
        ToolDefinition(
            name=name,
            description=_DESCRIPTIONS[name],
            parameters=cast("dict[str, JsonValue]", model.model_json_schema()),
        )
        for name, model in entries
    ]


def invoke(
    name: str,
    arguments: str,
    *,
    kinds: list[NativeKind],
    now: datetime,
    reference_period: Period | None,
) -> str:
    """Validate a model call and return only a deterministic native result."""
    allowed = {item.name for item in definitions(kinds)}
    if name not in allowed:
        raise NativeToolError("Native capability is not available for this turn.")
    if name == "resolve_period":
        period_args = ResolvePeriodArgs.model_validate_json(arguments)
        return periods.resolve_period(
            period_args.expression, now=now, reference_period=reference_period
        ).model_dump_json()
    arithmetic_args = ArithmeticArgs.model_validate_json(arguments)
    operations = {
        "percentage_change": arithmetic.percentage_change,
        "growth_rate": arithmetic.growth_rate,
        "share_of_total": arithmetic.share_of_total,
    }
    return str(operations[arithmetic_args.operation](arithmetic_args.first, arithmetic_args.second))
