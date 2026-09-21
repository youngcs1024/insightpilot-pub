"""Pure Suite C scoring, retaining failures and rules-only abstentions."""

from statistics import mean, stdev
from typing import Literal

from app.agents.contracts import Route
from app.core.routing import RoutingStrategy
from evals.harness.contracts import Score
from evals.harness.routing_contracts import (
    MAX_BOTH_MISROUTE,
    ROUTES,
    ArmReport,
    Attempt,
    Case,
    ClassScore,
    Distribution,
    Measurements,
    Metrics,
    Report,
)


def _predicted(attempt: Attempt) -> Route | None:
    observation = attempt.observation
    return (
        observation.decision.route
        if observation.decision and not observation.failure_code
        else None
    )


def metrics(attempts: list[Attempt], cases: list[Case]) -> Metrics:
    """A wrong specialist dispatch is a misroute; false clarification is accuracy loss."""
    expected = {case.id: case for case in cases}
    pairs = [(expected[a.case_id], _predicted(a), a.observation) for a in attempts]
    matrix = [
        [sum(c.expected is truth and pred is guess for c, pred, _ in pairs) for guess in ROUTES]
        for truth in ROUTES
    ]
    tokens = [o.tokens for _, _, o in pairs if o.tokens is not None]
    return Metrics(
        accuracy=Score(passed=sum(c.expected is p for c, p, _ in pairs), total=len(pairs)),
        misroute=Score(
            passed=sum(
                p is not None and p is not Route.CLARIFY and p is not c.expected
                for c, p, _ in pairs
            ),
            total=len(pairs),
        ),
        both_misroute=Score(
            passed=sum(
                c.expected is Route.BOTH and p in {Route.DATA_ONLY, Route.KNOWLEDGE_ONLY}
                for c, p, _ in pairs
            ),
            total=sum(c.expected is Route.BOTH for c, _, _ in pairs),
        ),
        clarification_precision=Score(
            passed=sum(
                p is Route.CLARIFY and c.clarification_kind == "ambiguous" for c, p, _ in pairs
            ),
            total=sum(
                p is Route.CLARIFY and c.clarification_kind != "out_of_scope" for c, p, _ in pairs
            ),
        ),
        out_of_scope_recall=Score(
            passed=sum(
                c.clarification_kind == "out_of_scope" and p is Route.CLARIFY for c, p, _ in pairs
            ),
            total=sum(c.clarification_kind == "out_of_scope" for c, _, _ in pairs),
        ),
        prefilter_hit_rate=Score(
            passed=sum(o.prefilter_hit for _, _, o in pairs), total=len(pairs)
        ),
        prefilter_accuracy=Score(
            passed=sum(o.prefilter_hit and c.expected is p for c, p, o in pairs),
            total=sum(o.prefilter_hit for _, _, o in pairs),
        ),
        per_class={
            route: ClassScore(
                precision=Score(passed=matrix[i][i], total=sum(row[i] for row in matrix)),
                recall=Score(
                    passed=matrix[i][i], total=sum(c.expected is route for c, _, _ in pairs)
                ),
            )
            for i, route in enumerate(ROUTES)
        },
        confusion_matrix=matrix,
        no_decision_by_class={
            r: sum(c.expected is r and p is None for c, p, _ in pairs) for r in ROUTES
        },
        mean_tokens=mean(tokens) if tokens and len(tokens) == len(pairs) else None,
        token_measurements=len(tokens),
        mean_latency_ms=mean(o.latency_ms for _, _, o in pairs) if pairs else 0,
    )


def _values(metric: Metrics) -> dict[str, float | None]:
    values = {
        name: getattr(metric, name).rate
        for name in (
            "accuracy",
            "misroute",
            "both_misroute",
            "clarification_precision",
            "out_of_scope_recall",
            "prefilter_hit_rate",
            "prefilter_accuracy",
        )
    }
    values.update(mean_tokens=metric.mean_tokens, mean_latency_ms=metric.mean_latency_ms)
    for route, score in metric.per_class.items():
        values[f"{route.value}_precision"] = score.precision.rate
        values[f"{route.value}_recall"] = score.recall.rate
    return values


def evaluate(raw: Measurements, cases: list[Case]) -> Report:
    """Derive rather than trust stored aggregates; incomplete grids fail closed."""
    selected = [c for c in cases if c.split is raw.split]
    expected_ids = {c.id for c in selected}
    grid = {
        (c.id, arm, n)
        for c in selected
        for arm in RoutingStrategy
        for n in range(1, raw.repeats + 1)
    }
    actual = [(a.case_id, a.arm, a.repeat) for a in raw.attempts]
    valid = (
        raw.complete
        and not raw.config.source_dirty
        and set(raw.case_ids) == expected_ids
        and len(raw.case_ids) == len(expected_ids)
        and set(actual) == grid
        and len(actual) == len(grid)
    )
    arms = {}
    for arm in RoutingStrategy:
        attempts = [a for a in raw.attempts if a.arm is arm and a.case_id in expected_ids]
        repeats = {
            n: metrics([a for a in attempts if a.repeat == n], selected)
            for n in range(1, raw.repeats + 1)
        }
        values = [_values(value) for value in repeats.values()]
        variation = {}
        for key in _values(metrics(attempts, selected)):
            measured = [value for v in values if (value := v[key]) is not None]
            complete = len(measured) == raw.repeats
            variation[key] = Distribution(
                mean=mean(measured) if measured and complete else None,
                sample_stddev=stdev(measured) if len(measured) > 1 and complete else None,
            )
        arms[arm] = ArmReport(
            overall=metrics(attempts, selected), per_repeat=repeats, variation=variation
        )
    return Report(
        measurements=raw,
        arms=arms,
        evidence_valid=valid,
        identical_arm_accuracy=len({a.overall.accuracy.rate for a in arms.values()}) == 1,
    )


def passes(metric: Metrics, threshold: float = 0.90) -> bool:
    """Routing acceptance includes the overview's BOTH safety ceiling."""
    accuracy, misroute = metric.accuracy.rate, metric.both_misroute.rate
    return (
        accuracy is not None
        and misroute is not None
        and accuracy >= threshold
        and misroute <= MAX_BOTH_MISROUTE
    )


def choose(report: Report) -> Literal[RoutingStrategy.HYBRID, RoutingStrategy.LLM_ONLY]:
    """Accuracy-first development decision; ties retain the hybrid default."""
    hybrid = report.arms[RoutingStrategy.HYBRID].overall
    llm = report.arms[RoutingStrategy.LLM_ONLY].overall
    if (
        passes(llm)
        and llm.accuracy.rate is not None
        and hybrid.accuracy.rate is not None
        and llm.accuracy.rate > hybrid.accuracy.rate
        and llm.misroute.rate is not None
        and hybrid.misroute.rate is not None
        and llm.misroute.rate <= hybrid.misroute.rate
        and llm.both_misroute.rate is not None
        and hybrid.both_misroute.rate is not None
        and llm.both_misroute.rate <= hybrid.both_misroute.rate
    ):
        return RoutingStrategy.LLM_ONLY
    return RoutingStrategy.HYBRID
