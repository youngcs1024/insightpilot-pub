"""Fail-closed, allowlisted telemetry projections; never mutate business payloads."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping

from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch
from pydantic import BaseModel, SecretStr

from app.core.logging import _CREDENTIAL, _CREDENTIAL_ASSIGNMENT, _SECRET_KEY
from app.schemas.sanity import SanityFlag

MAX_DEPTH = 12
MAX_UNRESOLVED_REFERENCES = 20
_ROW_MARKER = re.compile(r"\[<redacted: [0-9]+ rows × [0-9]+ cols>\]")  # noqa: RUF001 -- contract notation.
REDACTED = "<redacted>"
_ROUTING = frozenset({"route", "original_route", "confidence", "decided_by", "prefilter_hit", "router_tokens"})
_SAFE = frozenset(
    {
        "referenced_prior_turn",
        "unresolved_reference_count",
        "user_id",
        "conversation_id",
        "turn_id",
        "request_id",
        "route",
        "original_route",
        "confidence",
        "decided_by",
        "prefilter_hit",
        "router_tokens",
        "status",
        "degraded_components",
        "role",
        "model",
        "attempt",
        "structured_tier",
        "fallback_index",
        "schema_strategy",
        "sanity_flags",
        "sanity_check_failed",
        "schema_tables",
        "schema_tokens",
        "schema_tokenizer",
        "schema_utf8_bytes",
        "prompt_tokens",
        "completion_tokens",
        "input",
        "output",
        "total",
        "cost_status",
        "duration_ms",
        "execution_ms",
        "mcp_call_id",
        "tool",
        "row_count",
        "returned_row_count",
        "columns",
        "name",
        "type",
        "rows",
        "schema_version",
        "result_truncated",
        "sample_truncated",
        "limit_applied",
        "snapshot_id",
        "code",
        "reason",
        "evidence",
        "result_summary",
    }
)
_SCALAR = _SAFE - {"input", "output", "total"}
_OTEL_SAFE = frozenset(
    {
        "langfuse.observation.type",
        "langfuse.observation.level",
        "langfuse.observation.model.name",
        "langfuse.environment",
        "langfuse.release",
        "langfuse.version",
        "langfuse.internal.as_root",
        "user.id",
        "session.id",
    }
)
_OTEL_JSON = frozenset(
    {
        "langfuse.observation.metadata",
        "langfuse.trace.metadata",
        "langfuse.observation.output",
        "langfuse.observation.usage_details",
    }
)


def mask(data: object) -> object:
    """Project result schemas and diagnostics, dropping all unrecognized content."""
    try:
        return _walk(data, 0)
    except Exception:
        return REDACTED


def _walk(data: object, depth: int) -> object:
    if depth > MAX_DEPTH or isinstance(data, SecretStr):
        return "***"
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="python")
    if isinstance(data, Mapping):
        return _mapping(data, depth)
    if isinstance(data, (tuple, list)):
        return [_walk(item, depth + 1) for item in data[:200]]
    if isinstance(data, str):
        return _CREDENTIAL_ASSIGNMENT.sub("***", _CREDENTIAL.sub("***", data))
    if data is None or isinstance(data, (bool, int, float)):
        return data
    return REDACTED


def _token_diagnostic(key: str, value: object) -> object:
    if key == "schema_tokenizer":
        return "cl100k_base" if value == "cl100k_base" else REDACTED
    return value if isinstance(value, int) else REDACTED


def _mapping(data: Mapping[object, object], depth: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        if key in _ROUTING:
            result[key] = _routing_diagnostic(key, value)
        elif key in {"referenced_prior_turn", "unresolved_reference_count"}:
            result[key] = _rewrite_diagnostic(key, value)
        elif key in {"sanity_flags", "sanity_check_failed"}:
            result[key] = _sanity_diagnostic(key, value)
        elif key in {"prompt_tokens", "completion_tokens", "schema_tokens", "schema_tokenizer"}:
            result[key] = _token_diagnostic(key, value)
        elif _SECRET_KEY.search(key):
            result[key] = "***"
        elif key == "rows":
            result[key] = _mask_rows(value, data.get("columns", []))
        elif key in _SCALAR:
            result[key] = _walk(value, depth + 1)
        else:
            result[key] = REDACTED
    return result


def _routing_diagnostic(key: str, value: object) -> object:
    if value is None:
        return None
    if key in {"route", "original_route"}:
        allowed = {"data_only", "knowledge_only", "both", "clarify"}
        if key == "route":
            allowed.add("DATA_ONLY")
        return value if isinstance(value, str) and value in allowed else REDACTED
    if key == "decided_by":
        return value if isinstance(value, str) and value in {"prefilter", "llm"} else REDACTED
    if key == "prefilter_hit":
        return value if isinstance(value, bool) else REDACTED
    if isinstance(value, bool):
        return REDACTED
    if key == "router_tokens":
        return value if isinstance(value, int) and value >= 0 else REDACTED
    return value if isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 else REDACTED


def _routing_attribute(key: str, value: object) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            pass
    safe = _routing_diagnostic(key, value)
    return safe if isinstance(safe, str) else json.dumps(safe)


def _rewrite_diagnostic(key: str, value: object) -> object:
    if key == "referenced_prior_turn":
        return value if isinstance(value, bool) else REDACTED
    return (
        value
        if isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_UNRESOLVED_REFERENCES
        else REDACTED
    )


def _rewrite_attribute(key: str, value: object) -> str:
    # Both the SDK hook and final exporter project these attributes.
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return json.dumps(REDACTED)
    return json.dumps(_rewrite_diagnostic(key, value))


def _mask_rows(value: object, columns: object) -> str:
    if isinstance(value, str) and _ROW_MARKER.fullmatch(value):
        return value
    nrows = len(value) if isinstance(value, (list, tuple)) else 0
    ncols = len(columns) if isinstance(columns, (list, tuple)) else 0
    return f"[<redacted: {nrows} rows × {ncols} cols>]"  # noqa: RUF001 -- contract notation.


def _sanity_diagnostic(key: str, value: object) -> object:
    if key == "sanity_check_failed":
        return value if isinstance(value, bool) else REDACTED
    if not isinstance(value, (list, tuple)):
        return REDACTED
    allowed = {flag.value for flag in SanityFlag}
    return [item for item in value[:8] if isinstance(item, str) and item in allowed]


def _sanity_attribute(key: str, value: object) -> str:
    # The SDK JSON-serializes metadata lists before the final OTLP projection.
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return REDACTED
    return json.dumps(_sanity_diagnostic(key, value))


def mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
    """Remove nonallowlisted attributes; hook failure makes the SDK drop the batch."""
    patches = {}
    for identifier, span in params.spans.items():
        patches[identifier] = OtelSpanPatch(
            delete_attributes=tuple(span.attributes),
            set_attributes=safe_attributes(span.attributes),
        )
    return MaskOtelSpansResult(span_patches=patches)


def _json_attribute(key: str, value: str) -> str:
    try:
        parsed = json.loads(value)
        if key == "langfuse.observation.usage_details":
            return json.dumps(
                {
                    k: v
                    for k, v in parsed.items()
                    if k in {"input", "output", "total"} and isinstance(v, int) and v >= 0
                }
            )
        return json.dumps(
            mask(parsed) if isinstance(parsed, dict) else REDACTED, ensure_ascii=False
        )
    except Exception:
        return json.dumps(REDACTED)


def safe_attributes(attributes: Mapping[str, object]) -> dict[str, str | bool]:
    """Project serialized SDK attributes through the same final allowlist."""
    replacements: dict[str, str | bool] = {}
    for key, value in attributes.items():
        if key in {"langfuse.internal.as_root", "langfuse.internal.is_app_root"} and isinstance(
            value, bool
        ):
            replacements[key] = value
            continue
        metadata_key = key.removeprefix("langfuse.observation.metadata.")
        if metadata_key != key and metadata_key in _ROUTING:
            replacements[key] = _routing_attribute(metadata_key, value)
            continue
        if metadata_key != key and metadata_key in {
            "referenced_prior_turn",
            "unresolved_reference_count",
        }:
            replacements[key] = _rewrite_attribute(metadata_key, value)
            continue
        if metadata_key != key and metadata_key in {"sanity_flags", "sanity_check_failed"}:
            replacements[key] = _sanity_attribute(metadata_key, value)
            continue
        if (
            metadata_key != key
            and metadata_key in _SCALAR
            and metadata_key not in {"rows", "columns", "result_summary", "evidence"}
        ) or (key in _OTEL_SAFE and isinstance(value, str)):
            replacements[key] = str(mask(value))
        elif key in _OTEL_JSON and isinstance(value, str):
            replacements[key] = _json_attribute(key, value)
    return replacements
