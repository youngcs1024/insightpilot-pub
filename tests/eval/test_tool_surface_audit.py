# ruff: noqa: PLR2004 -- exact measured tool counts are the Step 5.9 regression oracle.
"""Step 5.9 capability inventory and zero-savings decision contract."""

import json
import socket

import pytest
from psycopg_pool import AsyncConnectionPool

from scripts import tool_surface_audit


def test_offline_audit_never_connects_and_excludes_mcp_from_model_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_connection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("The offline audit attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", fail_connection)
    monkeypatch.setattr(AsyncConnectionPool, "open", fail_connection)

    assert tool_surface_audit.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mcp_tool_names"] == [
        "execute_readonly_query",
        "get_schema",
        "resolve_metric",
    ]
    assert report["model_visible_tool_names"] == ["resolve_period", "calculate_percentage"]
    assert not set(report["mcp_tool_names"]) & set(report["model_visible_tool_names"])
    assert report["model_visible_tool_count"] == 2
    assert report["model_visible_tool_block_tokens"] == 263
    assert report["tokenizer"] == "cl100k_base"
    assert report["retrieval_top_k"] == 3
    assert report["top_k_prunable_tool_count"] == 0
