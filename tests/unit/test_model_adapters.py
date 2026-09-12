"""Adapter recovery policy and sparse/score semantics without ML imports."""
# ruff: noqa: PLR2004 -- exact protocol dimensions, scores and deadlines are test expectations.

import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from model_runtime.config import ModelServerSettings
from model_runtime.embedder import lexical_weights
from model_runtime.errors import ModelDeadlineError, ModelOOMError
from model_runtime.models import CudaModels
from model_runtime.reranker import sigmoid


class FakeOOMError(RuntimeError):
    pass


def models() -> CudaModels:
    backend = CudaModels(
        ModelServerSettings(
            auth_token="synthetic",  # noqa: S106 -- public fixture credential.
            embed_revision="a" * 40,
            rerank_revision="b" * 40,
        )
    )
    backend.torch = SimpleNamespace(
        cuda=SimpleNamespace(
            OutOfMemoryError=FakeOOMError,
            reset_peak_memory_stats=lambda: None,
            synchronize=lambda: None,
            empty_cache=lambda: None,
            max_memory_allocated=lambda: 123,
        ),
        inference_mode=nullcontext,
    )
    return backend


def test_oom_halves_once_without_partial_outputs() -> None:
    calls = []

    def operation(batch: int) -> list[int]:
        calls.append(batch)
        if batch == 16:
            raise FakeOOMError()
        return [42]

    assert models()._recover(operation, 16, time.monotonic() + 1) == ([42], 8)
    assert calls == [16, 8]


def test_repeated_oom_nonretryable_and_other_runtime_errors_not_recovered() -> None:
    calls = []

    def operation(batch: int) -> None:
        calls.append(batch)
        raise FakeOOMError()

    with pytest.raises(ModelOOMError) as caught:
        models()._recover(operation, 16, time.monotonic() + 1)
    assert calls == [16, 8]
    assert not caught.value.retryable

    def incompatible(batch: int) -> None:
        raise RuntimeError("unrelated model defect")

    with pytest.raises(RuntimeError, match="unrelated"):
        models()._recover(incompatible, 16, time.monotonic() + 1)


def test_recovery_cannot_renew_deadline() -> None:
    with pytest.raises(ModelDeadlineError):
        models()._recover(lambda _: None, 16, time.monotonic() - 1)


def test_sparse_max_weight_and_special_token_filtering() -> None:
    assert lexical_weights([0.2, 0.8, 1, -1], [42, 42, 0, 9], {0}) == {42: 0.8}


def test_sigmoid_handles_extreme_logits() -> None:
    assert sigmoid(1000) == 1
    assert sigmoid(-1000) == 0
    assert sigmoid(0) == 0.5
