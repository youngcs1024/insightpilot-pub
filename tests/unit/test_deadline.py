"""Deterministic monotonic budget arithmetic and typed expiration."""

import pytest

from app.core.deadline import Deadline
from app.core.errors import DeadlineExceededError


def test_budget_shrinks_to_remaining(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.deadline.time.monotonic", lambda: 95.0)
    deadline = Deadline(100)
    remaining, requested = 5, 2
    assert deadline.budget(30) == remaining
    assert deadline.budget(requested) == requested


def test_check_raises_when_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.deadline.time.monotonic", lambda: 101.0)
    deadline = Deadline(100)
    assert deadline.remaining() == 0
    with pytest.raises(DeadlineExceededError):
        deadline.check("query")
