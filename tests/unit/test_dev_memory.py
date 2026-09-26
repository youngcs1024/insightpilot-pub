"""The read-only explanation exposes exclusion reasons without hiding failures."""

import json
from unittest.mock import AsyncMock

import pytest

from app.agents.contracts import Route, RouteDecision
from app.schemas.memory_retrieval import MemoryDecision, MemoryReason, MemorySelection
from scripts import dev_memory
from tests.memory_retrieval_support import USER, stored


def test_explain_reports_excluded_metric(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    report = dev_memory.MemoryExplanation(
        route=RouteDecision(route=Route.KNOWLEDGE_ONLY, confidence=1, knowledge_intent="核查退货政策"),
        preparation=MemorySelection(),
        final=MemorySelection(decisions=[MemoryDecision(memory_id=stored().id, reason=MemoryReason.WRONG_ROUTE, score=0)]),
    )
    monkeypatch.setattr(dev_memory.MemoryProcessSettings, "load", lambda: None)
    monkeypatch.setattr(dev_memory, "run", AsyncMock(return_value=report))
    assert dev_memory.main(["retrieve", "--user", str(USER), "--question", "退货政策", "--explain"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["final"]["selected"] == []
    assert value["final"]["decisions"][0]["reason"] == "wrong_route"


def test_failed_read_has_nonzero_exit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    report = dev_memory.MemoryExplanation(
        route=RouteDecision(route=Route.KNOWLEDGE_ONLY, confidence=1, knowledge_intent="核查退货政策"),
        preparation=MemorySelection(), final=MemorySelection(failed=True, failure_code="database_error"),
    )
    monkeypatch.setattr(dev_memory.MemoryProcessSettings, "load", lambda: None)
    monkeypatch.setattr(dev_memory, "run", AsyncMock(return_value=report))
    assert dev_memory.main(["retrieve", "--user", str(USER), "--question", "政策", "--explain"]) == 1
    assert json.loads(capsys.readouterr().out)["final"]["failed"] is True
