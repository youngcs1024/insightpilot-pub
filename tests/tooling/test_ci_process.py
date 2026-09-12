"""Container failure evidence survives timeout and independent finalizer failures."""

import secrets
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import SecretStr

from scripts.ci_process import (
    CommandEvidence,
    CommandRecorder,
    CommandState,
    retain_primary_failure,
)
from scripts.deployment import DeploymentError


@pytest.mark.parametrize("state", list(CommandState))
def test_all_process_outcomes_are_recorded_and_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: CommandState,
) -> None:
    secret = secrets.token_urlsafe(24)
    result = subprocess.CompletedProcess(
        ["docker"],
        0 if state is CommandState.SUCCESS else 7,
        stdout=secret,
        stderr="failure " + secret,
    )
    run = Mock(return_value=result)
    if state is CommandState.TIMEOUT:
        run.side_effect = subprocess.TimeoutExpired(
            ["docker"], 1, output=secret.encode(), stderr=b"partial"
        )
    elif state is CommandState.UNAVAILABLE:
        run.side_effect = OSError("private diagnostic " + secret)
    monkeypatch.setattr("scripts.ci_process.subprocess.run", run)
    recorder = CommandRecorder(
        directory=tmp_path / "evidence",
        cwd=tmp_path,
        environment={"PASSWORD": secret},
        secrets=[SecretStr(secret)],
    )
    if state is CommandState.SUCCESS:
        recorder.run("build", ["docker", "build"], timeout=1)
    else:
        with pytest.raises(DeploymentError):
            recorder.run("build", ["docker", "build"], timeout=1)
    report = CommandEvidence.model_validate_json(
        next(recorder.directory.glob("*.json")).read_text()
    )
    assert report.state is state
    assert report.elapsed_s >= 0
    for path in recorder.directory.iterdir():
        assert secret not in path.read_text()
        assert "PASSWORD" not in path.read_text()
    assert run.call_count == 1
    assert run.call_args.kwargs["timeout"] == 1
    assert secret not in repr(recorder)


@pytest.mark.parametrize("primary_fails", [False, True])
def test_every_finalizer_runs_without_replacing_primary(primary_fails: bool) -> None:
    primary = DeploymentError("original")
    logs = Mock(side_effect=DeploymentError("logs"))
    stop = Mock(side_effect=DeploymentError("stop"))

    def body() -> None:
        with retain_primary_failure([logs, stop]):
            if primary_fails:
                raise primary

    with pytest.raises(DeploymentError) as raised:
        body()
    assert (raised.value is primary) is primary_fails
    logs.assert_called_once()
    stop.assert_called_once()
    if primary_fails:
        assert raised.value.__notes__


def test_successful_finalizers_do_not_hide_body_failure() -> None:
    stop = Mock()
    primary = DeploymentError("original")
    with pytest.raises(DeploymentError) as raised, retain_primary_failure([stop]):
        raise primary
    assert raised.value is primary
    stop.assert_called_once()


@pytest.mark.parametrize("uid", ["10001", "0"])
def test_runtime_user_probe_requests_pid_and_checks_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uid: str,
) -> None:
    from tests.milvus_support import MilvusStack, record_storage_users  # noqa: PLC0415 -- fixture boundary.

    recorder = CommandRecorder(directory=tmp_path, cwd=tmp_path, environment={})

    def command(_self: CommandRecorder, stage: str, args: list[str]) -> CommandEvidence:
        if stage == "container-ids":
            output = "a\nb\nc\n"
        elif stage == "configured-user":
            output = "10001:10001"
        else:
            assert args[-2:] == ["-eo", "pid,uid"]
            output = f"PID UID\n12 {uid}\n"
        return CommandEvidence(stage=stage, state=CommandState.SUCCESS, elapsed_s=0, stdout=output)

    monkeypatch.setattr(CommandRecorder, "run", command)
    monkeypatch.setattr("tests.milvus_support.EVIDENCE", tmp_path)
    stack = MilvusStack(uri="http://127.0.0.1:19530", command=["docker"], recorder=recorder)
    if uid == "0":
        with pytest.raises(AssertionError):
            record_storage_users(stack)
    else:
        record_storage_users(stack)
        assert (tmp_path / "runtime-users.json").is_file()
