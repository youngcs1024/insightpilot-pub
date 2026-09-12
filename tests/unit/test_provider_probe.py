"""Verify provider experiments offline; no credentials or live services are needed."""

import asyncio
import json
import os
from pathlib import Path
from time import monotonic

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from spikes.provider.cli import main
from spikes.provider.experiment import SAMPLE_COUNTS, make_request, summarize
from spikes.provider.models import (
    MODEL,
    Observation,
    Outcome,
    ProbeKind,
    ProviderSettings,
    Report,
    Settings,
)
from spikes.provider.reporting import metrics, percentile, render, write_report
from spikes.provider.runner import measure, run

SAMPLE = {"metric": "退款率", "count": 7, "region": {"name": "华东"}}
SAMPLE_AUTH = "test-only-secret-not-a-real-key"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolate_probe_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep offline tests independent of any configured live credentials."""
    for name in tuple(os.environ):
        if name.startswith("IP_SPIKE_PROVIDER"):
            monkeypatch.delenv(name)


@pytest.fixture
def settings() -> ProviderSettings:
    """Return an isolated synthetic workspace with bounded calls."""
    return ProviderSettings(workspace_id="llm-test", api_key=SecretStr(SAMPLE_AUTH))


def response(content: str = json.dumps(SAMPLE), **message: object) -> httpx.Response:
    """Construct a minimal provider response with actual token usage fields."""
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content, **message}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8100, "completion_tokens": 500},
        },
    )


def tool(name: str, arguments: str = json.dumps(SAMPLE)) -> dict[str, object]:
    """Create an OpenAI-compatible function call for a synthetic record."""
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


async def observe(
    settings: ProviderSettings, reply: httpx.Response, kind: ProbeKind = ProbeKind.NATIVE
) -> Observation:
    """Measure one sample against an in-memory HTTP transport."""
    row = Observation(probe=kind, sample=1, request=make_request(kind))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: reply)) as client:
        await measure(client, settings, row, monotonic() + 10)
    return row


@pytest.mark.parametrize("kind", [ProbeKind.NATIVE, ProbeKind.PROMPTED, ProbeKind.CHINESE])
async def test_strict_json_success(settings: ProviderSettings, kind: ProbeKind) -> None:
    """JSON must contain the exact nested Chinese record."""
    row = await observe(settings, response(), kind)
    assert row.outcome == Outcome.SUPPORTED
    assert row.attempts[0].usage is not None
    assert row.attempts[0].output_sha256


@pytest.mark.parametrize(
    "content",
    [
        "I ignored your schema.",
        '```json\n{"metric":"退款率"}\n```',
        json.dumps(SAMPLE | {"count": "7"}),
        json.dumps(SAMPLE | {"region": {"name": "乱码"}}),
        json.dumps(SAMPLE | {"extra": True}),
    ],
)
async def test_http_200_is_not_schema_support(settings: ProviderSettings, content: str) -> None:
    """Reject unconstrained prose, wrong types, extra fields and corrupted Chinese."""
    row = await observe(settings, response(content))
    assert row.outcome == Outcome.SCHEMA_INVALID
    assert len(row.attempts) == 1


@pytest.mark.parametrize(
    ("kind", "calls", "expected"),
    [
        (ProbeKind.TOOL, [tool("record")], Outcome.SUPPORTED),
        (ProbeKind.TOOL, [tool("wrong")], Outcome.SCHEMA_INVALID),
        (ProbeKind.TOOL, [tool("record", "{}")], Outcome.SCHEMA_INVALID),
        (ProbeKind.TOOL, [tool("record"), tool("record")], Outcome.SCHEMA_INVALID),
        (ProbeKind.PARALLEL, [tool("record_primary")], Outcome.SCHEMA_INVALID),
        (
            ProbeKind.PARALLEL,
            [tool("record_secondary"), tool("record_primary")],
            Outcome.SUPPORTED,
        ),
    ],
)
async def test_tool_validation(
    settings: ProviderSettings, kind: ProbeKind, calls: list[dict[str, object]], expected: Outcome
) -> None:
    """Validate every argument and exact requested tool multiplicity."""
    row = await observe(settings, response("", tool_calls=calls), kind)
    assert row.outcome == expected
    assert row.attempts[0].tool_count == len(calls)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, Outcome.REJECTED),
        (401, Outcome.AUTH),
        (403, Outcome.AUTH),
        (429, Outcome.RATE_LIMITED),
        (501, Outcome.SERVER_ERROR),
    ],
)
async def test_nonretryable_http_statuses(
    settings: ProviderSettings, status: int, expected: Outcome
) -> None:
    """Do not retry a 4xx or classify errors from secret-bearing response text."""
    row = await observe(settings, httpx.Response(status, text=SAMPLE_AUTH))
    assert row.outcome == expected
    assert len(row.attempts) == 1
    assert SAMPLE_AUTH not in row.model_dump_json()


@pytest.mark.parametrize("fault", ["server", "connection"])
async def test_retry_keeps_failed_attempt(settings: ProviderSettings, fault: str) -> None:
    """Retry only classified transient failures and retain failed attempts."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            if fault == "connection":
                raise httpx.ConnectError(SAMPLE_AUTH, request=request)
            return httpx.Response(503, text=SAMPLE_AUTH)
        return response()

    row = Observation(probe=ProbeKind.NATIVE, sample=1, request=make_request(ProbeKind.NATIVE))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await measure(client, settings, row, monotonic() + 10)
    assert row.outcome == Outcome.SUPPORTED
    assert [attempt.number for attempt in row.attempts] == [1, 2]
    assert row.attempts[0].outcome in {Outcome.CONNECTION, Outcome.SERVER_ERROR}
    assert SAMPLE_AUTH not in row.model_dump_json()


async def test_retries_are_bounded(settings: ProviderSettings) -> None:
    """A permanently failing upstream cannot turn the spike into an unbounded job."""
    row = await observe(settings, httpx.Response(503))
    assert len(row.attempts) == settings.max_attempts
    assert row.outcome == Outcome.SERVER_ERROR


async def test_timeout_is_not_retried(settings: ProviderSettings) -> None:
    """A timed-out generation may already be billed and is not blindly repeated."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(SAMPLE_AUTH, request=request)

    row = Observation(probe=ProbeKind.NATIVE, sample=1, request=make_request(ProbeKind.NATIVE))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await measure(client, settings, row, monotonic() + 10)
    assert row.outcome == Outcome.TIMEOUT
    assert len(row.attempts) == 1


async def test_deadline_interrupts_http_and_prevents_more_calls(settings: ProviderSettings) -> None:
    """The total deadline cancels an in-flight call and skips later samples."""
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)
        return response()

    row = Observation(probe=ProbeKind.NATIVE, sample=1, request=make_request(ProbeKind.NATIVE))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        deadline = monotonic() + 0.02
        await measure(client, settings, row, deadline)
        skipped = row.model_copy(update={"attempts": []})
        await measure(client, settings, skipped, deadline)
    assert row.outcome == Outcome.DEADLINE
    assert row.attempts[0].outcome == Outcome.DEADLINE
    assert not skipped.attempts
    assert calls == 1


async def test_auth_failure_skips_remaining_samples(settings: ProviderSettings) -> None:
    """Do not send eighteen requests with a known-invalid key."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(401, text=SAMPLE_AUTH))
    ) as client:
        report = await run(settings, client)
    assert sum(len(row.attempts) for row in report.observations) == 1
    assert len(report.observations) == sum(SAMPLE_COUNTS.values())
    assert report.recommended_tier is None
    assert not report.execution_complete
    assert SAMPLE_AUTH not in report.model_dump_json()


@pytest.mark.parametrize(
    "reply", [httpx.Response(200, text="bad JSON"), httpx.Response(200, json={})]
)
async def test_malformed_response_is_not_capability_rejection(
    settings: ProviderSettings, reply: httpx.Response
) -> None:
    """Invalid envelopes remain an inconclusive protocol failure."""
    assert (await observe(settings, reply)).outcome == Outcome.PROTOCOL


async def test_long_context_requires_markers_and_measured_tokens(
    settings: ProviderSettings,
) -> None:
    """Do not assert truncation from failed marker recall or missing usage."""
    markers = json.dumps({"first": "ip_begin_7491", "last": "ip_end_2863"})
    assert (await observe(settings, response(markers), ProbeKind.LONG)).outcome == Outcome.SUPPORTED
    assert (await observe(settings, response("{}"), ProbeKind.LONG)).outcome == Outcome.UNKNOWN
    payload = json.loads(response(markers).content)
    payload.pop("usage")
    row = await observe(settings, httpx.Response(200, json=payload), ProbeKind.LONG)
    assert row.outcome == Outcome.UNKNOWN
    payload["usage"] = {"prompt_tokens": 100, "completion_tokens": 10}
    row = await observe(settings, httpx.Response(200, json=payload), ProbeKind.LONG)
    assert row.outcome == Outcome.UNKNOWN


@pytest.mark.parametrize(
    ("native", "tool_outcome", "prompted", "tier"),
    [
        (Outcome.SUPPORTED, Outcome.SUPPORTED, Outcome.SUPPORTED, 1),
        (Outcome.REJECTED, Outcome.SUPPORTED, Outcome.SUPPORTED, 2),
        (Outcome.REJECTED, Outcome.REJECTED, Outcome.SUPPORTED, 3),
        (Outcome.AUTH, Outcome.AUTH, Outcome.AUTH, None),
        (Outcome.REJECTED, Outcome.REJECTED, Outcome.SCHEMA_INVALID, None),
    ],
)
def test_tier_selection(
    native: Outcome, tool_outcome: Outcome, prompted: Outcome, tier: int | None
) -> None:
    """The capability ladder requires schema evidence, not transport success."""
    report = Report(evidence_source="live_api")
    for kind, outcome in (
        (ProbeKind.NATIVE, native),
        (ProbeKind.TOOL, tool_outcome),
        (ProbeKind.PROMPTED, prompted),
    ):
        for sample in range(1, SAMPLE_COUNTS[kind] + 1):
            report.observations.append(
                Observation(probe=kind, sample=sample, request=make_request(kind), outcome=outcome)
            )
    summarize(report)
    assert report.recommended_tier == tier


async def test_complete_experiment_and_generated_artifacts(
    settings: ProviderSettings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise all requests and verify reports reflect measurements, including variance."""
    zero_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal zero_calls
        payload = json.loads(request.content)
        assert payload["model"] == MODEL
        assert payload["enable_thinking"] is False
        assert request.headers["authorization"] == "Bearer " + SAMPLE_AUTH
        if "tools" in payload:
            return response("", tool_calls=[tool(t["function"]["name"]) for t in payload["tools"]])
        content = payload["messages"][0]["content"]
        if "FIRST_MARKER=" in content:
            return response(json.dumps({"first": "ip_begin_7491", "last": "ip_end_2863"}))
        if "one short Chinese sentence" in content:
            zero_calls += 1
            return response(str(zero_calls))
        return response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run(settings, client)
    assert report.execution_complete
    assert report.recommended_tier == 1
    assert "identical outputs = false" in "\n".join(metrics(report))
    output = tmp_path / "capabilities.json"
    write_report(report, output)
    restored = Report.model_validate_json(output.read_text())
    assert restored == report
    markdown = output.with_name("PROVIDER_CAPABILITIES.md").read_text()
    assert markdown == render(restored)
    assert "5/5 successful samples" in markdown
    assert SAMPLE_AUTH not in output.read_text() + markdown + capsys.readouterr().out
    assert settings.workspace_id not in output.read_text()


def test_percentiles_and_pending_report() -> None:
    """Use explicit small-sample interpolation and never turn no data into zero latency."""
    assert percentile([], 0.5) is None
    assert percentile([50, 10, 40, 20, 30], 0.5) == 30  # noqa: PLR2004
    assert percentile([10, 20, 30, 40, 50], 0.95) == 48  # noqa: PLR2004
    pending = Report(evidence_source="not_run")
    summarize(pending)
    assert not pending.execution_complete
    assert pending.recommended_tier is None
    assert "not run" in render(pending)
    with pytest.raises(ValidationError):
        Report.model_validate(pending.model_dump() | {"schema_version": 2})


def test_settings_isolation_and_bounds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only explicit project settings are read; secret input is hidden in errors."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", SAMPLE_AUTH)
    (tmp_path / ".env").write_text(f"IP_SPIKE_PROVIDER__API_KEY={SAMPLE_AUTH}\n")
    with pytest.raises(ValidationError) as missing:
        Settings()
    assert SAMPLE_AUTH not in str(missing.value)
    env_file = tmp_path / "explicit.env"
    env_file.write_text(
        f"IP_SPIKE_PROVIDER__WORKSPACE_ID=llm-test\nIP_SPIKE_PROVIDER__API_KEY={SAMPLE_AUTH}\n"
    )
    configured = Settings(_env_file=env_file)
    assert configured.provider.api_key.get_secret_value() == SAMPLE_AUTH
    assert SAMPLE_AUTH not in repr(configured)
    with pytest.raises(ValidationError) as invalid:
        ProviderSettings(
            workspace_id="../../evil", api_key=SecretStr(SAMPLE_AUTH), max_attempts=100
        )
    assert SAMPLE_AUTH not in str(invalid.value)


def test_cli_missing_config_preserves_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A missing explicit configuration fails before touching the evidence files."""
    evidence = tmp_path / "docs/provider_capabilities.json"
    evidence.parent.mkdir()
    evidence.write_text('{"synthetic":true}\n')
    before = evidence.read_bytes()
    monkeypatch.setattr("spikes.provider.cli.ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["provider_probe", "--env-file", ".env.missing-provider-test"])
    assert main() == 2  # noqa: PLR2004 -- configuration error exit.
    assert "must exist" in capsys.readouterr().out
    assert evidence.read_bytes() == before


@pytest.mark.parametrize("tokens", [0, 30, 399, 651])
async def test_latency_requires_representative_output(
    settings: ProviderSettings, tokens: int
) -> None:
    """A fast tiny completion must not pass the approximately-500-token latency probe."""
    payload = json.loads(response("short").content)
    payload["usage"]["completion_tokens"] = tokens
    row = await observe(settings, httpx.Response(200, json=payload), ProbeKind.LATENCY)
    assert row.outcome == Outcome.UNKNOWN


async def test_latency_missing_usage_is_unknown(settings: ProviderSettings) -> None:
    """Do not substitute a character count for missing provider token accounting."""
    payload = json.loads(response().content)
    payload.pop("usage")
    row = await observe(settings, httpx.Response(200, json=payload), ProbeKind.LATENCY)
    assert row.outcome == Outcome.UNKNOWN


async def test_truncated_tool_output_is_unknown(settings: ProviderSettings) -> None:
    """Even parseable arguments do not establish complete tool output after a length stop."""
    payload = json.loads(response("", tool_calls=[tool("record")]).content)
    payload["choices"][0]["finish_reason"] = "length"
    row = await observe(settings, httpx.Response(200, json=payload), ProbeKind.TOOL)
    assert row.outcome == Outcome.UNKNOWN


def test_cli_cannot_write_reference_tree(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A custom output path cannot overwrite vendored reference material."""
    monkeypatch.setattr("sys.argv", ["provider_probe", "--out", "RagMate/capabilities.json"])
    assert main() == 2  # noqa: PLR2004 -- configuration error exit.
    assert "docs directory" in capsys.readouterr().out


def test_cli_invalid_secret_setting_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Even nested settings parse failures must not echo secret-bearing input."""
    monkeypatch.setenv("IP_SPIKE_PROVIDER", SAMPLE_AUTH)
    monkeypatch.setattr("sys.argv", ["provider_probe"])
    assert main() == 2  # noqa: PLR2004 -- configuration error exit.
    assert SAMPLE_AUTH not in capsys.readouterr().out


def test_template_covers_all_settings() -> None:
    """Every nested process setting has a discoverable local template entry."""
    template = (ROOT / "spikes/provider/.env.example").read_text()
    for name in ProviderSettings.model_fields:
        assert f"IP_SPIKE_PROVIDER__{name.upper()}=" in template
