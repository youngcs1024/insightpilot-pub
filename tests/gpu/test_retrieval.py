"""Explicit Suite B live evidence acceptance; excluded before ordinary collection."""

from pathlib import Path

import pytest

from evals.harness.retrieval_acceptance import AcceptanceBundle, issues

pytestmark = [pytest.mark.gpu, pytest.mark.external]


def test_complete_final_revision_retrieval_ablation(request: pytest.FixtureRequest) -> None:
    path = request.config.getoption("--retrieval-evidence")
    assert path, "New nine-arm development/frozen GPU evidence is required"
    bundle = AcceptanceBundle.model_validate_json(Path(path).read_text())
    assert not issues(bundle, request.config.getoption("--client-sha"))
