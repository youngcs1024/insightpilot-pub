"""Human and machine reports use the same recomputed routing measurements."""

from pathlib import Path

from app.core.routing import RoutingStrategy
from evals.harness.contracts import Score
from evals.harness.routing import passes
from evals.harness.routing_contracts import MIN_REPEATS, ROUTES, Report, Split


def score(value: Score) -> str:
    """Preserve absent denominators as unknown."""
    return f"{value.passed}/{value.total} ({value.rate:.2%})" if value.rate is not None else "N/A"


def markdown(report: Report) -> str:
    """Report all arms, repeats and failures, without credentials or model prose."""
    raw = report.measurements
    lines = [
        "# Routing evaluation v1",
        "",
        f"Run: `{raw.run_id}`; source: `{raw.config.git_sha}`; split: `{raw.split.value}`.",
        f"Evidence valid: {report.evidence_valid}; repeats: {raw.repeats}; labels: agent-reviewed.",
        "",
        "## Configuration",
        "",
        "```json",
        raw.config.model_dump_json(indent=2),
        "```",
        "",
        "Strategy selection: "
        + (raw.selection.arm.value if raw.selection else "development comparison"),
        "Selection commit: " + (raw.selection_commit or "N/A"),
        "",
        "Scores are split-specific. Repeats are not independent new questions.",
        "Wrong specialist dispatch counts as misroute; false clarification only lowers accuracy.",
        "Clarification precision excludes out-of-scope ground truth, reported separately.",
        "Unknown provider usage remains unknown; rule-only attempts use zero tokens.",
    ]
    for arm, result in report.arms.items():
        metric = result.overall
        lines.extend(
            [
                "",
                f"## {arm.value}",
                "",
                "| Metric | Result |",
                "| --- | --- |",
                f"| Accuracy | {score(metric.accuracy)} |",
                f"| Misroute | {score(metric.misroute)} |",
                f"| BOTH misroute | {score(metric.both_misroute)} |",
                f"| Clarification precision | {score(metric.clarification_precision)} |",
                f"| Out-of-scope recall | {score(metric.out_of_scope_recall)} |",
                f"| Prefilter hit rate | {score(metric.prefilter_hit_rate)} |",
                f"| Prefilter accuracy | {score(metric.prefilter_accuracy)} |",
                f"| Mean tokens | {metric.mean_tokens} ({metric.token_measurements} measured attempts) |",
                f"| Mean latency ms | {metric.mean_latency_ms:.2f} |",
                "",
                "| Class | Precision | Recall | No decision |",
                "| --- | --- | --- | --- |",
                *[
                    f"| {r.value} | {score(s.precision)} | {score(s.recall)} | {metric.no_decision_by_class[r]} |"
                    for r, s in metric.per_class.items()
                ],
                "",
                "### Confusion matrix (rows = truth, columns = prediction)",
                "",
                "| Truth | " + " | ".join(r.value for r in ROUTES) + " |",
                "| --- | --- | --- | --- | --- |",
                *[
                    "| " + r.value + " | " + " | ".join(str(v) for v in row) + " |"
                    for r, row in zip(ROUTES, metric.confusion_matrix, strict=True)
                ],
                "",
                "No-decision attempts stay in accuracy/recall denominators and are listed above.",
                "",
                "### Repeats",
                "",
                "| Repeat | Accuracy | Misroute | BOTH misroute |",
                "| --- | --- | --- | --- |",
                *[
                    f"| {n} | {score(m.accuracy)} | {score(m.misroute)} | {score(m.both_misroute)} |"
                    for n, m in result.per_repeat.items()
                ],
                "",
                "| Metric | Mean | Sample standard deviation |",
                "| --- | --- | --- |",
                *[
                    f"| {key} | {v.mean} | {v.sample_stddev} |"
                    for key, v in result.variation.items()
                ],
            ]
        )
    lines.extend(
        [
            "",
            "## Attempts",
            "",
            "| Case | Arm | Repeat | Route | Tokens | Latency ms | Failure |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for attempt in raw.attempts:
        o = attempt.observation
        route = o.decision.route.value if o.decision else "no_decision"
        lines.append(
            f"| {attempt.case_id} | {attempt.arm.value} | {attempt.repeat} | {route} | {o.tokens} | {o.latency_ms:.2f} | {o.failure_code or ''} |"
        )
    if report.identical_arm_accuracy:
        lines.extend(
            [
                "",
                "All arms have identical accuracy: investigate dataset discrimination before acceptance.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_report(report: Report, directory: Path) -> None:
    """Keep immutable run files and split-specific latest copies; delete nothing."""
    directory.mkdir(parents=True, exist_ok=True)
    raw = report.measurements
    for suffix, payload in (
        ("json", report.model_dump_json(indent=2) + "\n"),
        ("md", markdown(report)),
    ):
        with (directory / f"routing_{raw.run_id}.{suffix}").open("x", encoding="utf-8") as target:
            target.write(payload)
        (directory / f"routing_{raw.split.value}_latest.{suffix}").write_text(
            payload, encoding="utf-8"
        )
        (directory / f"routing_latest.{suffix}").write_text(payload, encoding="utf-8")


def exit_code(report: Report, threshold: float = 0.90) -> int:
    """Missing provenance, incomplete repeats and unmet quality gates never pass."""
    raw = report.measurements
    arm = raw.selection.arm if raw.selection else raw.config.production_strategy
    provenance = raw.split is Split.DEVELOPMENT or (
        raw.selection is not None
        and bool(raw.selection_commit)
        and raw.selection.fingerprint == raw.config.fingerprint()
        and raw.selection.arm is raw.config.production_strategy
    )
    return int(
        not report.evidence_valid
        or raw.repeats < MIN_REPEATS
        or report.identical_arm_accuracy
        or not provenance
        or arm is RoutingStrategy.PREFILTER_ONLY
        or not passes(report.arms[arm].overall, threshold)
    )
