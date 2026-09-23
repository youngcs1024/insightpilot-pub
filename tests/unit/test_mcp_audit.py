"""Pure audit classification and request-header isolation contracts."""

import hashlib
import json

import httpx2
from mcp.types import CallToolResult

from app.clients.mcp_client import attach_correlation_header
from mcp_server.audit import AuditOutcome, classify_result, correlation_header, hash_arguments


def test_hash_uses_sorted_compact_json() -> None:
    arguments: dict[str, object] = {"sql": "SELECT '秘密'", "max_rows": 10}
    expected = hashlib.sha256(
        json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert hash_arguments(arguments) == expected
    assert hash_arguments({"max_rows": 10, "sql": "SELECT '秘密'"}) == expected


def test_policy_and_timeout_are_typed() -> None:
    rejected = CallToolResult(
        content=[],
        is_error=True,
        structured_content={
            "code": "MCP_POLICY_REJECTED",
            "message": "ignored",
            "reasons": ["table_not_allowed"],
        },
    )
    timed_out = CallToolResult(
        content=[], is_error=True, structured_content={"code": "SQL_TIMEOUT", "message": "ignored"}
    )
    assert classify_result("resolve_metric", rejected) == (
        AuditOutcome.POLICY_REJECTED,
        ["table_not_allowed"],
        None,
    )
    assert classify_result("execute_readonly_query", timed_out) == (AuditOutcome.TIMEOUT, None, None)


def test_unstructured_error_cannot_be_misclassified_as_success() -> None:
    result = CallToolResult(content=[], is_error=True)
    assert classify_result("get_schema", result) == (AuditOutcome.EXECUTION_ERROR, None, None)


def test_header_accepts_only_uuid() -> None:
    value = "A85DEFCB-46ED-4CA9-B30C-0E5CFE782EB7"
    assert correlation_header({"X-Request-ID": value}) == "a85defcb46ed4ca9b30c0e5cfe782eb7"
    assert correlation_header({"X-Request-ID": "user-supplied-prose"}) is None


async def test_http_header_is_derived_from_each_call_body() -> None:
    first = httpx2.Request(
        "POST",
        "http://mcp.invalid/mcp",
        json={
            "method": "tools/call",
            "params": {"_meta": {"insightpilot/request_id": "a" * 32}},
        },
    )
    second = httpx2.Request(
        "POST",
        "http://mcp.invalid/mcp",
        json={
            "method": "tools/call",
            "params": {"_meta": {"insightpilot/request_id": "b" * 32}},
        },
    )
    await attach_correlation_header(first)
    await attach_correlation_header(second)
    assert first.headers["X-Request-ID"] == "a" * 32
    assert second.headers["X-Request-ID"] == "b" * 32
