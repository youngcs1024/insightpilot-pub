"""Static dataset loading and quota checks do no network or database work."""

from collections import Counter
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import TypeAdapter, ValidationError

from app.agents.contracts import Route
from app.core.errors import SchemaMetadataError
from data.seed.schema_metadata_loader import check_keys
from evals.harness.contracts import EvaluationError
from evals.harness.routing_contracts import CASES, Case, Split

SPLIT_SIZE = 40
CLASS_SIZE = 10


def load_cases(path: Path = CASES) -> list[Case]:
    """Reject duplicates, cross-split paraphrase groups and incomplete v1 suites."""
    try:
        source = path.read_text(encoding="utf-8")
        node = yaml.compose(source)
        if node is not None:
            check_keys(node)
        cases = TypeAdapter(list[Case]).validate_python(yaml.safe_load(source))
    except (OSError, ValidationError, yaml.YAMLError, SchemaMetadataError) as exc:
        raise EvaluationError("Invalid routing dataset") from exc
    identities = [(c.question.strip(), c.routing_context.model_dump_json()) for c in cases]
    if len({c.id for c in cases}) != len(cases) or len(set(identities)) != len(cases):
        raise EvaluationError("Duplicate routing case or input")
    groups: dict[str, Split] = {}
    for case in cases:
        if groups.setdefault(case.semantic_group_id, case.split) is not case.split:
            raise EvaluationError("Semantic group crosses development and frozen")
    for split in Split:
        selected = [c for c in cases if c.split is split]
        counts = Counter(c.expected for c in selected)
        if len(selected) != SPLIT_SIZE or counts != dict.fromkeys(Route, CLASS_SIZE):
            raise EvaluationError("Routing v1 requires forty balanced cases per split")
        if sum(c.deliberately_ambiguous for c in selected) < CLASS_SIZE:
            raise EvaluationError("Insufficient deliberately ambiguous routing cases")
    return cases
