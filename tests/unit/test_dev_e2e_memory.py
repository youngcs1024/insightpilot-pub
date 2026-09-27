"""The diagnostic entry point selects the cloud scenario and never hides a failing exit."""

# ruff: noqa: PLR2004 -- fixed acceptance counts and retry/timeout boundaries.

from unittest.mock import Mock

import pytest

from scripts import dev_e2e_memory


@pytest.mark.parametrize("exit_code", [0, 1])
def test_same_scenario_and_exit_status(monkeypatch: pytest.MonkeyPatch, exit_code: int) -> None:
    call = Mock(return_value=Mock(returncode=exit_code))
    monkeypatch.setattr(dev_e2e_memory.subprocess, "run", call)
    assert dev_e2e_memory.main() == exit_code
    args, kwargs = call.call_args
    assert dev_e2e_memory.CASE in args[0]
    assert kwargs["cwd"] == dev_e2e_memory.ROOT
    assert kwargs["timeout"] == 600
    assert "--no-cov" in args[0]
