"""Static case loading is safe during pytest collection."""

from pathlib import Path

import yaml  # type: ignore[import-untyped]  # PyYAML ships no typing marker.
from pydantic import TypeAdapter, ValidationError

from app.core.errors import InvalidMetricPatchError, SchemaMetadataError
from app.services.metric_patch_sql import canonical_filters
from data.seed.schema_metadata_loader import check_keys
from evals.harness.adversarial import CASES as ADVERSARIAL_CASES
from evals.harness.adversarial import load_adversaries
from evals.harness.contracts import CASES, Case, EvaluationError

ORDINARY_COUNT = 26


def load_cases(path: Path = CASES, adversarial_path: Path = ADVERSARIAL_CASES) -> list[Case]:
    """Reject incomplete or ambiguous suites before opening a service."""
    try:
        source = path.read_text()
        node = yaml.compose(source)
        if node is not None:
            check_keys(node)
        cases = TypeAdapter(list[Case]).validate_python(yaml.safe_load(source))
        if len(cases) != ORDINARY_COUNT or any(case.adversarial_sql for case in cases):
            raise EvaluationError("Ordinary accuracy denominator changed")
        for case in cases:
            for binding in case.expected_bindings:
                canonical_filters(binding.filters)
    except (
        OSError,
        yaml.YAMLError,
        ValidationError,
        SchemaMetadataError,
        InvalidMetricPatchError,
    ) as exc:
        raise EvaluationError("Invalid static evaluation dataset") from exc
    cases.extend(
        case.suite_a_case() for case in load_adversaries(adversarial_path) if case.expected_reasons
    )
    ids = [case.id for case in cases]
    if not cases or len(set(ids)) != len(ids):
        raise EvaluationError("Empty suite or duplicate case IDs")
    return cases
