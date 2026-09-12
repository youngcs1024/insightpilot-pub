"""Human-readable and machine-readable reports from the same typed measurements."""

from pathlib import Path

from evals.harness.contracts import Report, Score, Summary


def _score(score: Score) -> str:
    return f"{score.passed}/{score.total} ({score.rate:.1%})" if score.rate is not None else "N/A"


def _row(name: str, summary: Summary) -> str:
    return (
        "| "
        + " | ".join(
            [
                name,
                _score(summary.execution_match),
                _score(summary.result_accuracy),
                _score(summary.metric_resolution_accuracy),
                _score(summary.unsafe_sql_block_rate),
            ]
        )
        + " |"
    )


def markdown(report: Report) -> str:
    """Render only controlled identifiers and public configuration in the summary."""
    lines = [
        "# NL2SQL v1 evaluation",
        "",
        f"Run: `{report.run_id}`; source: `{report.config.git_sha}`",
        f"Evidence valid: {report.evidence_valid}; source dirty: {report.config.source_dirty}",
        "",
        "## Configuration snapshot",
        "",
        "```json",
        report.config.model_dump_json(indent=2),
        "```",
        "",
        "## Scores",
        "",
        "| Group | Execution match | Result accuracy | Metric resolution | Unsafe SQL block |",
        "| --- | --- | --- | --- | --- |",
        _row("All", report.summary),
        *[_row(trap, summary) for trap, summary in report.per_trap.items()],
        *[_row(f"Repeat {n}", summary) for n, summary in report.per_repeat.items()],
        "",
        "Unsafe SQL uses fixed validator probes, not model refusal. Failed attempts stay in denominators.",
        "Repeated scores above describe variability; no best-attempt selection is performed.",
        "",
        "## Attempts",
        "",
        "| Case | Repeat | Execution | Result | Metrics | Blocked | Valid | Corrections | Failure |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        corrections = r.observation.corrections if r.observation else 0
        lines.append(
            f"| {r.case_id} | {r.repeat} | {r.execution_match} | {r.result_accuracy} "
            f"| {r.metric_resolution_accuracy} | {r.unsafe_sql_blocked} "
            f"| {r.evidence_valid} | {corrections} | {r.failure_code or ''} |"
        )
    rates = [
        rate
        for summary in report.per_repeat.values()
        if (rate := summary.result_accuracy.rate) is not None
    ]
    if rates:
        lines.extend(["", f"Result accuracy across repeats: {min(rates):.1%}-{max(rates):.1%}."])
    execution = report.summary.execution_match.rate
    accuracy = report.summary.result_accuracy.rate
    if execution is not None and accuracy is not None:
        lines.extend(["", f"Execution minus result accuracy: {execution - accuracy:.1%}."])
    return "\n".join(lines) + "\n"


def write_report(report: Report, directory: Path) -> None:
    """Retain immutable run files and convenient latest copies; delete nothing."""
    directory.mkdir(parents=True, exist_ok=True)
    for suffix, payload in (
        ("json", report.model_dump_json(indent=2) + "\n"),
        ("md", markdown(report)),
    ):
        with (directory / f"nl2sql_{report.run_id}.{suffix}").open("x", encoding="utf-8") as target:
            target.write(payload)
        (directory / f"nl2sql_latest.{suffix}").write_text(payload, encoding="utf-8")


def exit_code(report: Report, threshold: float | None) -> int:
    """Missing attempts, invalid evidence and any unblocked adversary fail closed."""
    safety = report.summary.unsafe_sql_block_rate.rate
    accuracy = report.summary.result_accuracy.rate
    complete = (
        report.evidence_valid
        and not report.config.source_dirty
        and len(report.results) == report.expected_attempts
        and len({(r.case_id, r.repeat) for r in report.results}) == report.expected_attempts
        and all(r.evidence_valid for r in report.results)
    )
    return int(
        not complete
        or safety != 1.0
        or accuracy is None
        or (threshold is not None and accuracy < threshold)
    )
