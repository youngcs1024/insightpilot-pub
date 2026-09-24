"""Candidate preview uses production extraction without opening any database."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.config_models import Settings
from app.core.errors import LlmUnavailableError
from app.core.llm_config import ModelRole
from app.schemas.memory_extraction import MemoryExtraction
from scripts import dev_extract
from tests.memory_extraction_support import DURABLE, TRANSIENT, candidate


@pytest.mark.parametrize(("message", "candidates"), [(DURABLE, True), (TRANSIENT, False)])
async def test_preview_uses_production_gates_and_closes_client(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, message: str, candidates: bool
) -> None:
    llm = AsyncMock()
    llm.generate_structured.return_value = MemoryExtraction(
        candidates=[candidate()] if candidates else []
    )
    monkeypatch.setattr(dev_extract, "LlmService", Mock(return_value=llm))
    config = dev_extract.ExtractionProcessSettings(_env_file=None, llm=settings.llm)
    result = await dev_extract.run(message, config)
    assert bool(result.candidates) is candidates
    assert llm.generate_structured.call_args.args[0] is ModelRole.MEMORY_EXTRACT
    llm.start.assert_awaited_once()
    llm.aclose.assert_awaited_once()


@pytest.mark.parametrize("error", [LlmUnavailableError(), asyncio.CancelledError(), TimeoutError()])
async def test_preview_failure_closes_client(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    llm = AsyncMock()
    llm.generate_structured.side_effect = error
    monkeypatch.setattr(dev_extract, "LlmService", Mock(return_value=llm))
    config = dev_extract.ExtractionProcessSettings(_env_file=None, llm=settings.llm)
    with pytest.raises(type(error)):
        await dev_extract.run(DURABLE, config)
    llm.aclose.assert_awaited_once()


@pytest.mark.parametrize("show", [False, True])
def test_preview_cli_output(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    show: bool,
) -> None:
    config = dev_extract.ExtractionProcessSettings(_env_file=None, llm=settings.llm)
    monkeypatch.setattr(dev_extract.ExtractionProcessSettings, "load", lambda: config)
    monkeypatch.setattr(dev_extract, "run", AsyncMock(return_value=MemoryExtraction()))
    assert dev_extract.main([TRANSIENT, *(["--show-candidates"] if show else [])]) == 0
    output = capsys.readouterr().out
    if show:
        assert json.loads(output)["candidates"] == []
    else:
        assert "preview only; no writes" in output


@pytest.mark.parametrize("error", [LlmUnavailableError("private-upstream-detail"), TimeoutError()])
def test_preview_cli_failure_is_nonzero_and_safe(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    config = dev_extract.ExtractionProcessSettings(_env_file=None, llm=settings.llm)
    monkeypatch.setattr(dev_extract.ExtractionProcessSettings, "load", lambda: config)
    monkeypatch.setattr(dev_extract, "run", AsyncMock(side_effect=error))
    assert dev_extract.main([DURABLE, "--show-candidates"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "private-upstream-detail" not in output.err
