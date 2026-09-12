"""Deterministic stimuli, strict output checks and capability selection."""

import hashlib
import json
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from spikes.provider.models import (
    Attempt,
    Completion,
    Contract,
    Message,
    Observation,
    Outcome,
    ProbeKind,
    Report,
    Request,
    StructuredSample,
)

PROMPT_ROOT = Path(__file__).resolve().parents[2] / "app/agents/prompts"
PROMPTS = {
    name: (PROMPT_ROOT / f"provider_probe_{name}.md").read_text(encoding="utf-8")
    for name in ("structured", "parallel", "zero", "long", "latency")
}
SAMPLE_COUNTS = dict.fromkeys(ProbeKind, 1) | {
    ProbeKind.PROMPTED: 5,
    ProbeKind.ZERO: 3,
    ProbeKind.LATENCY: 5,
}
UNMEASURED = frozenset(
    {
        Outcome.AUTH,
        Outcome.RATE_LIMITED,
        Outcome.SERVER_ERROR,
        Outcome.CONNECTION,
        Outcome.TIMEOUT,
        Outcome.DEADLINE,
        Outcome.PROTOCOL,
        Outcome.UNKNOWN,
    }
)
MIN_CONTEXT_TOKENS = 7500
MAX_CONTEXT_TOKENS = 9000
MIN_LATENCY_TOKENS = 400
MAX_LATENCY_TOKENS = 650
TIERS: tuple[tuple[Literal[1, 2, 3], ProbeKind], ...] = (
    (1, ProbeKind.NATIVE),
    (2, ProbeKind.TOOL),
    (3, ProbeKind.PROMPTED),
)


class Markers(Contract):
    """Exact sentinel recall, without guessing whether upstream truncated input."""

    first: Literal["ip_begin_7491"]
    last: Literal["ip_end_2863"]


def make_request(probe: ProbeKind) -> Request:
    """Build a fixed, secret-free request for one experiment."""
    request = Request(messages=[Message(role="user", content=PROMPTS["structured"])])
    schema = StructuredSample.model_json_schema()
    if probe == ProbeKind.NATIVE:
        request.response_format = {
            "type": "json_schema",
            "json_schema": {"name": "retail_record", "strict": True, "schema": schema},
        }
    elif probe in {ProbeKind.TOOL, ProbeKind.PARALLEL}:
        names = (
            ["record_primary", "record_secondary"] if probe == ProbeKind.PARALLEL else ["record"]
        )
        request.tools = [
            {
                "type": "function",
                "function": {"name": name, "parameters": schema},
            }
            for name in names
        ]
        if probe == ProbeKind.PARALLEL:
            request.tool_choice = "auto"
            request.parallel_tool_calls = True
            request.messages[0].content = PROMPTS["parallel"]
        else:
            request.tool_choice = {"type": "function", "function": {"name": "record"}}
    else:
        set_text_stimulus(request, probe)
    return request


def set_text_stimulus(request: Request, probe: ProbeKind) -> None:
    """Select a text-only stimulus; schema prompting uses no API shaping option."""
    if probe in {ProbeKind.PROMPTED, ProbeKind.CHINESE}:
        request.messages[0].content += "\nJSON schema:\n" + json.dumps(
            StructuredSample.model_json_schema(), ensure_ascii=False
        )
    elif probe == ProbeKind.ZERO:
        request.messages[0].content = PROMPTS["zero"]
    elif probe == ProbeKind.LONG:
        # Approximately one token per common English word; usage is the actual measurement.
        request.messages[0].content = PROMPTS["long"].format(filler=" amber" * 8000)
    elif probe == ProbeKind.LATENCY:
        request.messages[0].content = PROMPTS["latency"]
        request.max_tokens = 600


def check_tools(completion: Completion, probe: ProbeKind) -> Outcome:
    """Check the requested names, multiplicity, and every tool argument schema."""
    calls = completion.choices[0].message.tool_calls
    if completion.choices[0].finish_reason not in {"stop", "tool_calls"}:
        return Outcome.UNKNOWN
    expected = ["record_primary", "record_secondary"] if probe == ProbeKind.PARALLEL else ["record"]
    for call in calls:
        StructuredSample.model_validate_json(call.function.arguments, strict=True)
    if sorted(call.function.name for call in calls) != sorted(expected):
        return Outcome.SCHEMA_INVALID
    return Outcome.SUPPORTED


def check_output(completion: Completion, probe: ProbeKind) -> Outcome:
    """Validate observable properties; an HTTP 200 alone proves nothing."""
    choice = completion.choices[0]
    content = choice.message.content or ""
    if probe in {ProbeKind.TOOL, ProbeKind.PARALLEL}:
        return check_tools(completion, probe)
    if choice.message.tool_calls or not content.strip():
        return Outcome.SCHEMA_INVALID
    if probe in {ProbeKind.NATIVE, ProbeKind.PROMPTED, ProbeKind.CHINESE}:
        StructuredSample.model_validate_json(content, strict=True)
        return Outcome.SUPPORTED if choice.finish_reason == "stop" else Outcome.UNKNOWN
    if probe in {ProbeKind.LONG, ProbeKind.LATENCY}:
        return check_token_measurement(completion, probe)
    return Outcome.SUPPORTED if choice.finish_reason == "stop" else Outcome.UNKNOWN


def check_token_measurement(completion: Completion, probe: ProbeKind) -> Outcome:
    """Require measured token usage for the context and latency experiments."""
    choice = completion.choices[0]
    usage = completion.usage
    if usage is None:
        return Outcome.UNKNOWN
    if probe == ProbeKind.LONG:
        Markers.model_validate_json(choice.message.content or "", strict=True)
        valid = (
            choice.finish_reason == "stop"
            and MIN_CONTEXT_TOKENS <= usage.prompt_tokens <= MAX_CONTEXT_TOKENS
        )
    else:
        valid = (
            choice.finish_reason in {"stop", "length"}
            and MIN_LATENCY_TOKENS <= usage.completion_tokens <= MAX_LATENCY_TOKENS
        )
    return Outcome.SUPPORTED if valid else Outcome.UNKNOWN


def examine(completion: Completion, observation: Observation, attempt: Attempt) -> None:
    """Retain hashes and typed metrics; never persist provider text or error bodies."""
    choice = completion.choices[0]
    attempt.usage = completion.usage
    attempt.tool_count = len(choice.message.tool_calls)
    attempt.output_sha256 = hashlib.sha256(
        choice.message.model_dump_json().encode("utf-8")
    ).hexdigest()
    reason = choice.finish_reason
    attempt.finish_reason = (
        cast("Literal['stop', 'length', 'tool_calls']", reason)
        if reason in ("stop", "length", "tool_calls")
        else "other"
    )
    try:
        attempt.outcome = check_output(completion, observation.probe)
    except ValidationError:
        # The validation exception includes provider text: deliberately do not log it.
        attempt.outcome = (
            Outcome.UNKNOWN if observation.probe == ProbeKind.LONG else Outcome.SCHEMA_INVALID
        )


def summarize(report: Report) -> None:
    """Choose a tier from measurements, keeping infrastructure failures inconclusive."""
    by_kind = {
        kind: [row for row in report.observations if row.probe == kind] for kind in ProbeKind
    }
    report.recommended_tier = None
    for tier, kind in TIERS:
        rows = by_kind[kind]
        if len(rows) == SAMPLE_COUNTS[kind] and all(
            row.outcome == Outcome.SUPPORTED for row in rows
        ):
            report.recommended_tier = tier
            break
    report.execution_complete = report.evidence_source == "live_api" and all(
        len(by_kind[kind]) == count
        and all(row.outcome not in UNMEASURED and row.attempts for row in by_kind[kind])
        for kind, count in SAMPLE_COUNTS.items()
    )
