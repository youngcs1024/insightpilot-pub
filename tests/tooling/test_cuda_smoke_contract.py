"""Validate probe evidence without importing torch, downloading models or accessing a GPU."""

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from spikes.model_runtime.cuda_smoke import (
    DENSE_DIMENSION,
    PAIR_COUNT,
    TEXT_COUNT,
    Outputs,
    valid_snapshot,
)


def test_valid_probe_outputs() -> None:
    """The fixed workload accepts valid dense, sparse and normalized rerank output."""
    result = Outputs(
        dense=[[0.1] * DENSE_DIMENSION for _ in range(TEXT_COUNT)],
        sparse=[{1: 0.5} for _ in range(TEXT_COUNT)],
        scores=[0.5] * PAIR_COUNT,
    )
    assert len(result.dense) == TEXT_COUNT


@pytest.mark.parametrize("failure", ["dimension", "nan", "sparse", "count", "range"])
def test_invalid_probe_outputs(failure: str) -> None:
    """Malformed output can never become successful CUDA evidence."""
    dense = [[0.1] * DENSE_DIMENSION for _ in range(TEXT_COUNT)]
    sparse = [{1: 0.5} for _ in range(TEXT_COUNT)]
    scores = [0.5] * PAIR_COUNT
    if failure == "dimension":
        dense[0] = [0.1]
    elif failure == "nan":
        dense[0][0] = math.nan
    elif failure == "sparse":
        sparse[0] = {}
    elif failure == "count":
        scores = [0.5]
    else:
        scores[0] = 1.5
    with pytest.raises(ValidationError):
        Outputs(dense=dense, sparse=sparse, scores=scores)


def test_mutable_or_missing_snapshot_rejected(tmp_path: Path) -> None:
    """A branch name or missing cache configuration is not a revision-pinned input."""
    mutable = tmp_path / "main"
    mutable.mkdir()
    (mutable / "config.json").write_text("{}")
    assert not valid_snapshot(mutable)
    immutable = tmp_path / ("a" * 40)
    immutable.mkdir()
    assert not valid_snapshot(immutable)
    (immutable / "config.json").write_text("{}")
    assert valid_snapshot(immutable)
