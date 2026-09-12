"""Render human-readable evidence from the same versioned JSON contract."""

import math
from collections import Counter
from pathlib import Path

from spikes.provider.experiment import SAMPLE_COUNTS
from spikes.provider.models import Outcome, ProbeKind, Report


def percentile(values: list[float], quantile: float) -> float | None:
    """Return the linearly interpolated empirical percentile (R-7)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metrics(report: Report) -> list[str]:
    """Summarize sample reliability without hiding failed calls or retries."""
    prompted = [row for row in report.observations if row.probe == ProbeKind.PROMPTED]
    success = sum(row.outcome == Outcome.SUPPORTED for row in prompted)
    zero = [row for row in report.observations if row.probe == ProbeKind.ZERO]
    identical = "unknown"
    if len(zero) == SAMPLE_COUNTS[ProbeKind.ZERO] and all(
        row.outcome == Outcome.SUPPORTED and row.attempts for row in zero
    ):
        identical = str(len({row.attempts[-1].output_sha256 for row in zero}) == 1).lower()
    latency = [row for row in report.observations if row.probe == ProbeKind.LATENCY]
    durations = [
        row.attempts[-1].elapsed_ms
        for row in latency
        if row.outcome == Outcome.SUPPORTED and row.attempts
    ]
    output_tokens = [
        row.attempts[-1].usage.completion_tokens
        for row in latency
        if row.attempts and row.attempts[-1].usage is not None
    ]
    return [
        f"- Prompted JSON: {success}/{len(prompted)} successful samples (planned: 5).",
        f"- Temperature zero: identical outputs = {identical} (planned: 3).",
        f"- Latency: {len(durations)}/5 successful samples; "
        f"p50 = {percentile(durations, 0.5)} ms; p95 = {percentile(durations, 0.95)} ms.",
        f"- Latency completion token counts: {output_tokens} (target: approximately 500).",
        "- Percentiles use linear interpolation (R-7) on successful final HTTP attempts, "
        "excluding retry backoff. A qualifying sample needs 400-650 reported output tokens; "
        "missing usage or shorter output is inconclusive. All failed attempts remain in JSON; five samples are "
        "descriptive evidence, not a production latency benchmark.",
    ]


def render(report: Report) -> str:
    """Build the matrix and runbook without reading credentials or external state."""
    lines = [
        "# Provider capabilities — Step 0.4",
        "",
        f"Evidence source: **{report.evidence_source}**. "
        f"Execution complete: **{str(report.execution_complete).lower()}**.",
        "",
        f"Model: `{report.model}`. Provider: Alibaba Cloud Bailian. Region: `{report.region}`.",
        f"Started: {report.started_at}. Finished: {report.finished_at}.",
        f"Recommended structured-output tier: **{report.recommended_tier or 'undetermined'}**.",
        "",
        "Only this model is in the approved scope; deepseek-chat, qwen-max and glm-4 "
        "from the original example were not probed. No model-role mapping is assumed. "
        "Phase 1 Step 1.8 reads this per-model evidence through its own role registry.",
        "",
        "| Probe | Observed outcomes |",
        "| --- | --- |",
    ]
    for kind in ProbeKind:
        counts = Counter(row.outcome.value for row in report.observations if row.probe == kind)
        summary = (
            ", ".join(f"{name}: {count}" for name, count in sorted(counts.items())) or "not run"
        )
        lines.append(f"| {kind.value} | {summary} |")
    lines.extend(["", *metrics(report), "", "## Interpretation", ""])
    lines.extend(
        [
            "Native/tool support requires strict validation of the three-field nested schema. "
            "A schema-invalid HTTP 200 is recorded as invalid or ignored; it does not prove "
            "the provider's internal decoding mechanism. Tier 3 is recommended only when "
            "all five prompted JSON samples pass; otherwise the tier remains undetermined "
            "unless tier 1 or 2 passed. Repair belongs to Step 1.8, not this experiment.",
            "",
            "Chinese values are checked using prompted JSON independently of API shaping. "
            "Parallel calls require both named tools with valid arguments; each attempt "
            "records the actual tool count. Non-determinism at temperature zero is an "
            "observation, not an execution failure.",
            "",
            "Long-context input contains 8,000 repeated English words plus first/last markers. "
            "Success requires exact marker recall and provider usage of 7,500-9,000 input "
            "tokens. Missing usage or markers is unknown, not proof of input truncation. "
            "Actual usage and finish reasons are retained for every valid response.",
            "",
            "Authentication, network, rate-limit and malformed-response failures leave "
            "acceptance pending. HTTP request rejection is an observed result, not proof "
            "that every deployment of this model lacks the capability. A complete run can "
            "contain unsupported capabilities. All requests set enable_thinking=false.",
            "",
            "## Reproduce",
            "",
            "Create `.env.provider-probe` in the InsightPilot root using "
            "`spikes/provider/.env.example`. Fill in the workspace ID and API key locally. "
            "The file is gitignored. Do not paste keys into the command line or commit them.",
            "",
            "```bash",
            "bash scripts/uv_project.sh api run --locked --no-env-file python "
            "spikes/provider_probe.py --models qwen3.6-flash-2026-04-16 "
            "--env-file .env.provider-probe --out docs/provider_capabilities.json",
            "cat docs/PROVIDER_CAPABILITIES.md",
            "make lint",
            "make typecheck",
            "make test",
            "```",
            "",
            "Exit codes: 0 = measurements complete; 1 = live evidence inconclusive; "
            "2 = configuration or output-path error. Configuration errors do not overwrite "
            "existing evidence. JSON and this Markdown are generated together. "
            "An artifact with evidence_source=not_run is a pending record, not a mock run.",
            "",
            "Default limits: 60 seconds per HTTP call, 900 seconds for the run, at most two "
            "attempts per sample (one-second backoff). There are 18 planned samples. "
            "Only connection establishment failures and HTTP 500/502/503/504 are retried; "
            "timeouts, 4xx and schema failures are retained without retry. Authentication "
            "failure skips subsequent calls; skipped samples have no attempts.",
            "",
            "Endpoint: `https://{workspace_id}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`. "
            "See [regional workspace endpoints](https://www.alibabacloud.com/help/en/model-studio/regions) "
            "and [model documentation](https://help.aliyun.com/zh/model-studio/qwen3-6-flash). "
            "Documentation claims do not substitute for real measurements.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: Report, output: Path) -> None:
    """Persist the JSON contract and derived Markdown to the chosen directory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    output.with_name("PROVIDER_CAPABILITIES.md").write_text(render(report), encoding="utf-8")
