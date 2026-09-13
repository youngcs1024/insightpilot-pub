"""CLI phase boundaries and fail-before-inference behavior, without live services."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import pytest

from evals import cli
from evals.harness import retrieval_cli, retrieval_runtime
from evals.harness.ablation import choose, evaluate
from app.retrieval.config import RetrievalSettings
from evals.harness.contracts import EvaluationError
from evals.harness.retrieval_contracts import Split
from tests.retrieval_eval_support import dataset, measurements


def test_cli_collect_retains_eight_arms_but_control_is_pending(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = measurements(control=False)
    monkeypatch.setattr(retrieval_cli, "load_dataset", lambda *args: dataset())
    monkeypatch.setattr(retrieval_cli, "provenance", lambda *args: raw.server)
    collect = AsyncMock(return_value=raw)
    monkeypatch.setattr(retrieval_cli, "collect", collect)
    path = tmp_path / "report.md"
    result = cli.main(["ablation", "--suite", "retrieval", "--arms", "all", "--report", str(path)])
    assert result == 2
    assert path.exists() and path.with_suffix(".raw.json").exists()
    assert collect.await_count == 1


def test_cli_selection_and_report_phases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(retrieval_cli, "load_dataset", lambda *args: dataset())
    raw = tmp_path / "raw.json"
    raw.write_text(measurements(control=False).model_dump_json())
    selection = tmp_path / "selected.yaml"
    report = tmp_path / "report.md"
    assert cli.main(["ablation", "--phase", "select", "--measurements", str(raw), "--selection", str(selection), "--report", str(report)]) == 0
    assert selection.exists()
    raw.write_text(measurements().model_dump_json())
    assert cli.main(["ablation", "--phase", "report", "--measurements", str(raw), "--report", str(report)]) == 0


def test_frozen_checks_selection_before_opening_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    load = AsyncMock()
    monkeypatch.setattr(retrieval_cli, "load_dataset", load)
    def fail(path: Path) -> None:
        raise EvaluationError("Selection is not committed")
    monkeypatch.setattr(retrieval_cli, "committed_selection", fail)
    assert cli.main(["ablation", "--split", "frozen"]) == 2
    load.assert_not_called()


def test_missing_provenance_and_invalid_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrieval_cli, "load_dataset", lambda *args: dataset())
    assert cli.main(["ablation"]) == 2
    assert cli.main(["ablation", "--threshold", "1.1"]) == 2


async def test_manifest_mismatch_fails_before_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(
        model_runtime=SimpleNamespace(precision="fp16"), database=object(),
        retrieval=RetrievalSettings(),
    )
    monkeypatch.setattr(retrieval_runtime.RetrievalProcessSettings, "load", lambda: settings)
    monkeypatch.setattr(retrieval_runtime, "Database", MagicMock())
    monkeypatch.setattr(retrieval_runtime, "ModelRuntimeClient", MagicMock())
    monkeypatch.setattr(retrieval_runtime, "HybridSearchStore", MagicMock())
    monkeypatch.setattr(retrieval_runtime, "source_identity", lambda: ("a" * 40, False))
    monkeypatch.setattr(retrieval_runtime, "validate_active", AsyncMock(side_effect=EvaluationError("manifest mismatch")))
    close = AsyncMock()
    retrieve = AsyncMock()
    monkeypatch.setattr(retrieval_runtime, "close_resources", close)
    monkeypatch.setattr(retrieval_runtime, "observe", retrieve)
    with pytest.raises(EvaluationError, match="manifest"):
        await retrieval_runtime.collect(dataset(), measurements().server)
    retrieve.assert_not_called()
    close.assert_awaited_once()


def test_select_phase_cannot_accept_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = dataset()
    selection = choose(evaluate(measurements(control=False), source, require_control=False), source)
    monkeypatch.setattr(retrieval_cli, "committed_selection", lambda *args: (selection, "b" * 40))
    monkeypatch.setattr(retrieval_cli, "load_dataset", lambda *args: dataset(Split.FROZEN))
    raw = tmp_path / "raw.json"
    raw.write_text(measurements(control=False, split=Split.FROZEN).model_dump_json())
    assert cli.main(["ablation", "--phase", "select", "--split", "frozen", "--measurements", str(raw), "--report", str(tmp_path / "report.md")]) == 2
