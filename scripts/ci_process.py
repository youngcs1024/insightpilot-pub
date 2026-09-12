"""Bounded subprocess evidence for isolated CI containers, including failure paths."""

import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, PrivateAttr, SecretStr

from scripts.deployment import DeploymentError


class CommandState(StrEnum):
    """Classify process outcomes without parsing exception prose."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


class CommandEvidence(BaseModel):
    """Only sanitized output and fixed stage identifiers are written to artifacts."""

    stage: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    state: CommandState
    exit_code: int | None = None
    elapsed_s: float = Field(ge=0)
    stdout: str = ""
    stderr: str = ""


class CommandRecorder(BaseModel):
    """Keep the environment private and record every completed or timed-out call."""

    directory: Path
    cwd: Path
    environment: dict[str, str] = Field(repr=False, exclude=True)
    secrets: list[SecretStr] = Field(default_factory=list, repr=False, exclude=True)
    _sequence: int = PrivateAttr(default=0)

    def redact(self, output: str | bytes | None) -> str:
        """Remove explicitly supplied synthetic credentials before any persistence."""
        text = output.decode(errors="replace") if isinstance(output, bytes) else output or ""
        for secret in sorted(
            self.secrets, key=lambda item: len(item.get_secret_value()), reverse=True
        ):
            value = secret.get_secret_value()
            if value:
                text = text.replace(value, "[REDACTED]")
        return text

    def run(self, stage: str, command: list[str], *, timeout: int = 30) -> CommandEvidence:
        """Capture partial timeout output; commands and environments never enter evidence."""
        evidence = CommandEvidence(stage=stage, state=CommandState.UNAVAILABLE, elapsed_s=0)
        began = time.monotonic()
        try:
            result = subprocess.run(  # noqa: S603 -- structured fixed commands, never shell input.
                command,
                cwd=self.cwd,
                env=self.environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            evidence.exit_code = result.returncode
            evidence.state = CommandState.FAILURE if result.returncode else CommandState.SUCCESS
            evidence.stdout = self.redact(result.stdout)
            evidence.stderr = self.redact(result.stderr)
        except subprocess.TimeoutExpired as exc:
            evidence.state = CommandState.TIMEOUT
            evidence.stdout = self.redact(exc.stdout)
            evidence.stderr = self.redact(exc.stderr)
        except OSError:
            evidence.state = CommandState.UNAVAILABLE
        evidence.elapsed_s = time.monotonic() - began
        self.directory.mkdir(parents=True, exist_ok=True)
        self._sequence += 1
        prefix = self.directory / f"{self._sequence:03d}-{stage}"
        prefix.with_suffix(".json").write_text(evidence.model_dump_json(indent=2) + "\n")
        prefix.with_suffix(".log").write_text(evidence.stdout + evidence.stderr)
        if evidence.state is not CommandState.SUCCESS:
            raise DeploymentError("CI container command failed.", stage=stage, state=evidence.state)
        return evidence


@contextmanager
def retain_primary_failure(finalizers: list[Callable[[], object]]) -> Iterator[None]:
    """Attempt every finalizer; never replace the primary setup or body failure."""
    primary: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        primary = exc
        raise
    finally:
        failures: list[Exception] = []
        for finalize in finalizers:
            try:
                finalize()
            except Exception as exc:
                failures.append(exc)
        if failures:
            if primary is not None:
                primary.add_note(
                    "Additional container finalizer failures are recorded in evidence."
                )
            else:
                raise DeploymentError(
                    "Container evidence collection or stop failed."
                ) from failures[0]
