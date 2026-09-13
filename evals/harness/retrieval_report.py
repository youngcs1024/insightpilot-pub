"""Private Markdown/JSON evidence with explicit incomplete and negative outcomes."""

from pathlib import Path

from evals.harness.retrieval_contracts import AblationReport, Arm


def markdown(report: AblationReport) -> str:
    """Keep pre-filter quality, production filtering and replay latency distinct."""
    run = report.measurements
    lines = [
        "# Retrieval ablation — " + run.split.value,
        "",
        "Evidence valid: " + str(report.evidence_valid),
        "Client SHA: " + run.client_sha,
        "Dataset: " + run.dataset_identity,
        "Corpus: " + run.corpus_version,
        "Judgments: agent-reviewed under explicit user authorization; not human-reviewed.",
        "",
        "IR averages use answerable queries only; negative controls are reported separately.",
        "Arm scores are recorded: search timings include diagnostic single-arm requests.",
        "FP32 total is candidate-replay wall time, NOT a new end-to-end retrieval time.",
        "No missing/failed/degraded attempt counts as successful evidence.",
        "",
        "| Arm | Valid/attempted | R@5 | R@10 | nDCG@10 | MRR@10 | P@5 | Filtered nDCG | Correct abstain/negative | False evidence | Client p50/p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.summaries:
        score = row.ranking
        latency = row.latency_ms["total_ms"]
        lines.append(
            f"| {row.arm.value} | {row.valid}/{row.attempted} | {score.recall_5:.4f} | {score.recall_10:.4f} | {score.ndcg_10:.4f} | {score.mrr_10:.4f} | {score.precision_5:.4f} | {row.filtered.ndcg_10:.4f} | {row.correct_abstentions}/{row.negative_count} | {row.false_evidence} | {latency.p50}/{latency.p95} |"
        )
    lines.extend(["", "## Stage latency (client milliseconds)", ""])
    for row in report.summaries:
        for stage, value in row.latency_ms.items():
            lines.append(
                f"- {row.arm.value} {stage}: n={value.count}, p50={value.p50}, p95={value.p95}"
            )
    lines.extend(contributions(report))
    if report.selection:
        lines.extend(
            [
                "",
                "## Development-selected configuration",
                "",
                "```json",
                report.selection.model_dump_json(indent=2),
                "```",
            ]
        )
    if report.issues:
        lines.extend(
            ["", "## Missing or invalid evidence", "", *["- " + item for item in report.issues]]
        )
    lines.extend(
        [
            "",
            "The +15% nDCG improvement and 3.5s p50 targets are later-stage goals; negative results remain reportable.",
            "",
        ]
    )
    return "\n".join(lines)


def contributions(report: AblationReport) -> list[str]:
    """State stage contributions as measured deltas, never presumed improvement."""
    if not report.evidence_valid:
        return ["", "No accepted stage-contribution conclusion: evidence is incomplete."]
    rows = {item.arm: item for item in report.summaries}
    lines = ["", "## Paired stage contributions", ""]
    pairs = [(Arm.A, Arm.B), (Arm.A, Arm.C), (Arm.B, Arm.D)]
    pairs.extend(
        [
            (Arm.A, Arm.A_RERANK),
            (Arm.B, Arm.B_RERANK),
            (Arm.C, Arm.C_RERANK),
            (Arm.D, Arm.D_RERANK),
            (Arm.D_RERANK, Arm.D_FP32),
        ]
    )
    for left, right in pairs:
        if left not in rows or right not in rows:
            continue
        delta = rows[right].ranking.ndcg_10 - rows[left].ranking.ndcg_10
        recall = rows[right].ranking.recall_10 - rows[left].ranking.recall_10
        lines.append(
            f"- {left.value} → {right.value}: ΔnDCG@10={delta:+.4f}, ΔRecall@10={recall:+.4f}; negative/zero is retained."
        )
    indexed = {(item.query_id, item.arm): item for item in report.results}
    changes = sum(
        row.abstained != indexed[(row.query_id, Arm.D_FP32)].abstained
        for row in report.results
        if row.arm is Arm.D_RERANK and (row.query_id, Arm.D_FP32) in indexed
    )
    lines.append(f"- FP16/FP32 abstention decisions changed: {changes} queries.")
    attempts = {(item.query_id, item.arm): item for item in report.measurements.attempts}
    pairs_observed = [
        (item.observed, attempts[(item.query_id, Arm.D_FP32)].observed)
        for item in report.measurements.attempts
        if item.arm is Arm.D_RERANK and (item.query_id, Arm.D_FP32) in attempts
    ]
    floor_changes = sum(
        left.result.meets_floor != right.result.meets_floor
        for left, right in pairs_observed
        if left is not None and right is not None
    )
    lines.append(f"- FP16/FP32 absolute-floor decisions changed: {floor_changes} queries.")
    return lines


def write_report(report: AblationReport, path: Path) -> None:
    """Retain full attempts and snapshots alongside the human-readable table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    path.write_text(markdown(report), encoding="utf-8")


def exit_code(report: AblationReport, threshold: float | None) -> int:
    """2 means invalid evidence; 1 is a measured metric below an explicit gate."""
    if not report.evidence_valid:
        return 2
    if threshold is not None:
        arm = (
            report.selection.arm
            if report.selection
            else max(
                (item for item in report.summaries if item.arm is not Arm.D_FP32),
                key=lambda item: item.ranking.ndcg_10,
            ).arm
        )
        quality = next(item.ranking.ndcg_10 for item in report.summaries if item.arm is arm)
        if quality < threshold:
            return 1
    return 0
