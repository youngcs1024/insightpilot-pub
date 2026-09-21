"""CLI uses production routing while owning only lazily initialized LLM resources."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from app.agents.contracts import Route, RouterInput
from app.core.config_models import LLMSettings, RouterSettings, Settings
from app.core.errors import LlmConfigurationError, LlmRequestError
from app.core.llm_config import ModelRole
from scripts import dev_route
from tests.router_support import BOTH_QUESTION, decision


@pytest.fixture(autouse=True)
def isolated_route_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in tuple(os.environ):
        if key.upper().startswith("IP_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(dev_route.RouteProcessSettings, "project_root", tmp_path)


def configured() -> dev_route.RouteProcessSettings:
    return dev_route.RouteProcessSettings(
        _env_file=None,
        llm=LLMSettings(
            base_url="https://provider.invalid/v1",
            model="test-model",
            api_key="test-only",
        ),
    )


async def test_simple_demo_constructs_no_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock(side_effect=AssertionError("must remain lazy"))
    monkeypatch.setattr(dev_route, "LlmService", factory)
    result = await dev_route.run(
        RouterInput(question="8月的GMV是多少?"),
        dev_route.RouteProcessSettings(_env_file=None, router={"strategy": "hybrid"}),
    )
    assert result.route is Route.DATA_ONLY
    assert result.decided_by == "prefilter"
    factory.assert_not_called()


async def test_both_demo_uses_shared_classifier_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = AsyncMock()
    llm.generate_structured.return_value = decision()
    monkeypatch.setattr(dev_route, "LlmService", Mock(return_value=llm))
    result = await dev_route.run(RouterInput(question=BOTH_QUESTION), configured())
    assert result.route is Route.BOTH
    assert result.data_intent != result.knowledge_intent
    assert result.data_intent != BOTH_QUESTION
    llm.start.assert_awaited_once()
    llm.aclose.assert_awaited_once()
    assert llm.generate_structured.call_args.args[0] is ModelRole.ROUTER


@pytest.mark.parametrize("when", ["start", "generate_structured"])
async def test_failed_cli_initialization_or_call_closes_client(
    monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    llm = AsyncMock()
    getattr(llm, when).side_effect = LlmRequestError("private-provider-error")
    monkeypatch.setattr(dev_route, "LlmService", Mock(return_value=llm))
    with pytest.raises(LlmRequestError):
        await dev_route.run(RouterInput(question=BOTH_QUESTION), configured())
    llm.aclose.assert_awaited_once()


async def test_cancellation_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = AsyncMock()
    llm.generate_structured.side_effect = asyncio.CancelledError()
    monkeypatch.setattr(dev_route, "LlmService", Mock(return_value=llm))
    with pytest.raises(asyncio.CancelledError):
        await dev_route.run(RouterInput(question=BOTH_QUESTION), configured())
    llm.aclose.assert_awaited_once()


async def test_missing_model_config_fails_only_when_model_needed() -> None:
    with pytest.raises(LlmConfigurationError):
        await dev_route.run(
            RouterInput(question=BOTH_QUESTION), dev_route.RouteProcessSettings(_env_file=None)
        )


def test_cli_prints_prefilter_result(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IP_ROUTER__STRATEGY", "hybrid")
    assert dev_route.main(["8月的GMV是多少?"]) == 0
    printed = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert printed["route"] == "data_only"
    assert printed["decided_by"] == "prefilter"


def test_cli_missing_llm_config_has_safe_nonzero_exit(capsys: pytest.CaptureFixture[str]) -> None:
    assert dev_route.main([BOTH_QUESTION]) == 1
    assert "LLM_CONFIGURATION" in capsys.readouterr().err


def test_cli_summary_is_passed_as_routing_context(monkeypatch: pytest.MonkeyPatch) -> None:
    operation = AsyncMock(return_value=decision(Route.CLARIFY))
    monkeypatch.setattr(dev_route, "run", operation)
    assert dev_route.main(["那个", "--summary", "此前讨论GMV"]) == 0
    assert operation.call_args.args[0].routing_context.summary == "此前讨论GMV"


def test_cli_invalid_input_has_safe_nonzero_exit(capsys: pytest.CaptureFixture[str]) -> None:
    assert dev_route.main(["x" * 32_001]) == 1
    assert capsys.readouterr().err == "Invalid routing input or configuration.\n"


def test_cli_cleanup_timeout_has_safe_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(dev_route, "run", AsyncMock(side_effect=TimeoutError("private-cleanup")))
    assert dev_route.main([BOTH_QUESTION]) == 1
    assert capsys.readouterr().err == "Routing cleanup timed out.\n"


@pytest.mark.parametrize("invalid", [-0.01, 1.01, float("nan"), float("inf")])
def test_confidence_setting_is_finite_and_bounded(invalid: float) -> None:
    with pytest.raises(ValidationError):
        RouterSettings(min_confidence=invalid)


def test_process_override_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert dev_route.RouteProcessSettings(_env_file=None).router.min_confidence == 0.6  # noqa: PLR2004 -- specified default.
    monkeypatch.setenv("IP_ROUTER__MIN_CONFIDENCE", "0.8")
    assert dev_route.RouteProcessSettings.load().router.min_confidence == 0.8  # noqa: PLR2004 -- explicit environment override.


def test_api_settings_support_router_override(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IP_ROUTER__MIN_CONFIDENCE", "0.75")
    values = settings.model_dump(exclude={"router"})
    assert Settings(_env_file=None, **values).router.min_confidence == 0.75  # noqa: PLR2004 -- explicit environment override.
