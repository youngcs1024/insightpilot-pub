"""Real routing implementation with deterministic provider and Git substitutes."""

# ruff: noqa: PLR2004 -- known provider token counts and bounded run grids.

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
import yaml
from pydantic import ValidationError

from app.agents.contracts import Route, RouterInput
from app.agents.nodes.router import classify_question, route_question
from app.core.config_models import LLMSettings, RouterSettings
from app.core.errors import ConflictError, LlmUnavailableError
from app.core.routing import RoutingStrategy
from app.services.llm.contracts import Usage
from app.services.llm.usage import collect_usage, record_attempt, record_usage
from evals.harness import routing_cli, routing_runtime
from evals.harness.contracts import EvaluationError
from evals.harness.routing import evaluate
from evals.harness.routing_contracts import Options, Selection, Snapshot, Split
from evals.harness.routing_dataset import load_cases
from evals.harness.routing_runtime import committed_selection, observe
from scripts.dev_route import RouteProcessSettings
from tests.eval.routing_support import measurements, snapshot
from tests.llm_support import URL, response, service
from tests.router_support import BOTH_QUESTION, decision, runtime


async def test_three_arms_share_production_classifier_without_label_input() -> None:
    inputs = RouterInput(question="2026年6月GMV")
    rules = runtime()
    hybrid = runtime()
    llm = runtime([decision(Route.DATA_ONLY)])
    assert (await observe(inputs, rules, RoutingStrategy.PREFILTER_ONLY)).tokens == 0
    assert (await observe(inputs, hybrid, RoutingStrategy.HYBRID)).decision.route is Route.DATA_ONLY
    assert (await observe(inputs, llm, RoutingStrategy.LLM_ONLY)).decision.route is Route.DATA_ONLY
    assert not rules.llm.calls and not hybrid.llm.calls
    assert len(llm.llm.calls) == 1
    assert set(json.loads(llm.llm.calls[0].messages[-1].content)) == {"question", "routing_context"}


async def test_prefilter_abstains_without_model_call() -> None:
    ctx = runtime()
    result = await observe(RouterInput(question=BOTH_QUESTION), ctx, RoutingStrategy.PREFILTER_ONLY)
    assert result.decision is None and result.failure_code is None and result.tokens == 0
    assert not ctx.llm.calls


async def test_llm_only_preserves_confidence_gate() -> None:
    ctx = runtime([decision(Route.DATA_ONLY, confidence=0.4)])
    result = await classify_question(RouterInput(question="2026年6月GMV"), ctx, RoutingStrategy.LLM_ONLY)
    assert result.route is Route.CLARIFY


async def test_production_strategy_selects_same_llm_path() -> None:
    ctx = replace(runtime([decision(Route.DATA_ONLY)]), settings=RouterSettings(strategy=RoutingStrategy.LLM_ONLY))
    result = await route_question(RouterInput(question="2026年6月GMV"), ctx)
    assert result.route is Route.DATA_ONLY and len(ctx.llm.calls) == 1
    with pytest.raises(ValidationError):
        RouterSettings(strategy=RoutingStrategy.PREFILTER_ONLY)


async def test_impossible_production_nondecision_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.agents.nodes.router.classify_question", AsyncMock(return_value=None))
    with pytest.raises(ConflictError):
        await route_question(RouterInput(question="test"), runtime())


async def test_failed_provider_is_recorded_not_relabelled_clarify() -> None:
    result = await observe(RouterInput(question=BOTH_QUESTION), runtime([LlmUnavailableError()]), RoutingStrategy.LLM_ONLY)
    assert result.decision is None
    assert result.failure_code == LlmUnavailableError.code
    assert result.tokens is None


async def test_nested_provider_usage_reaches_evaluation(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(URL).mock(return_value=response(decision().model_dump_json()))
    async with service() as llm:
        result = await observe(RouterInput(question=BOTH_QUESTION), replace(runtime(), llm=llm), RoutingStrategy.HYBRID)
    assert result.tokens == 20
    assert result.latency_ms > 0


async def test_unreported_provider_usage_is_not_zero(respx_mock: respx.MockRouter) -> None:
    payload = json.loads(response(decision().model_dump_json()).content)
    payload.pop("usage")
    respx_mock.post(URL).mock(return_value=httpx.Response(200, json=payload))
    async with service() as llm:
        result = await observe(RouterInput(question=BOTH_QUESTION), replace(runtime(), llm=llm), RoutingStrategy.LLM_ONLY)
    assert result.tokens is None


def test_nested_usage_aggregates_once_on_exception() -> None:
    with collect_usage() as outer:
        record_attempt()
        record_usage(Usage(prompt_tokens=1, completion_tokens=2))
        with pytest.raises(LlmUnavailableError), collect_usage():
            record_attempt()
            record_usage(Usage(prompt_tokens=3, completion_tokens=4))
            raise LlmUnavailableError()
    assert outer.attempts == outer.reported == 2
    assert outer.total == 10
    with collect_usage() as fresh:
        assert fresh.attempts == 0


def test_snapshot_exports_only_public_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    def git(args: list[str]) -> bytes:
        if args[0] == "rev-parse":
            return b"a" * 40
        if args[0] == "ls-files":
            return b"app/agents/nodes/router.py\n"
        return b""

    monkeypatch.setattr(routing_runtime, "git_bytes", git)
    settings = RouteProcessSettings(
        _env_file=None,
        llm=LLMSettings(base_url="https://secret-endpoint.invalid", model="test", api_key="secret-token"),
    )
    config = routing_runtime.snapshot(settings)
    text = config.model_dump_json()
    assert "secret" not in text
    assert config.model.model == "test"
    assert config.prompt_hashes.keys() == {"router.md", "clarify.md", "structured_json.md", "structured_repair.md"}
    assert config.fingerprint() == config.model_copy(update={"git_sha": "b" * 40}).fingerprint()


def selection(config: Snapshot | None = None) -> Selection:
    config = config or snapshot()
    return Selection(
        arm=RoutingStrategy.HYBRID, development_run_id="development",
        development_sha="a" * 40, development_report_hash="hash", fingerprint=config.fingerprint(),
    )


@pytest.mark.parametrize("defect", ["none", "uncommitted", "fingerprint", "strategy", "dirty"])
def test_committed_selection_prevents_frozen_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str) -> None:
    config = snapshot()
    selected = selection(config)
    content = yaml.safe_dump(selected.model_dump(mode="json"))
    path = tmp_path / "selected.yaml"
    path.write_text(content)
    monkeypatch.setattr(routing_runtime, "ROOT", tmp_path)

    def git(args: list[str]) -> bytes:
        if args[0] == "show":
            return (content if defect != "uncommitted" else "different").encode()
        return b"a" * 40

    monkeypatch.setattr(routing_runtime, "git_bytes", git)
    if defect == "fingerprint":
        config.dataset_hash = "changed"
    elif defect == "strategy":
        config.production_strategy = RoutingStrategy.LLM_ONLY
    elif defect == "dirty":
        config.source_dirty = True
    if defect == "none":
        assert committed_selection(config, path) == (selected, "a" * 40)
    else:
        with pytest.raises(EvaluationError):
            committed_selection(config, path)


async def test_frozen_checks_selection_before_dataset_or_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RouteProcessSettings, "load", lambda: RouteProcessSettings(_env_file=None))
    monkeypatch.setattr(routing_runtime, "snapshot", lambda settings: snapshot())

    def reject(config: Snapshot) -> None:
        raise EvaluationError("No committed selection")

    monkeypatch.setattr(routing_runtime, "committed_selection", reject)
    loader = AsyncMock()
    monkeypatch.setattr(routing_runtime, "load_cases", loader)
    with pytest.raises(EvaluationError):
        await routing_runtime.collect(Options(split=Split.FROZEN))
    loader.assert_not_called()


def test_selection_recomputes_metrics_and_requires_current_development(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = measurements()
    report = evaluate(raw, load_cases())
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json())
    target = tmp_path / "selected.yaml"
    monkeypatch.setattr(routing_cli, "SELECTION", target)
    monkeypatch.setattr(RouteProcessSettings, "load", lambda: RouteProcessSettings(_env_file=None))
    monkeypatch.setattr(routing_cli, "snapshot", lambda settings: raw.config)
    assert routing_cli.select(path) == 0
    assert yaml.safe_load(target.read_text())["arm"] == "hybrid"
    raw.split = Split.FROZEN
    path.write_text(evaluate(raw, load_cases()).model_dump_json())
    with pytest.raises(EvaluationError):
        routing_cli.select(path)
