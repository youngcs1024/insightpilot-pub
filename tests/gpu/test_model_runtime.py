"""Dedicated live HTTP acceptance; never imported by ordinary test collection."""
# ruff: noqa: PLR2004 -- exact protocol dimensions, scores and deadlines are test expectations.

import time

import pytest
from pydantic import SecretStr

from app.clients.model_runtime import ModelRuntimeClient
from app.core.deadline import Deadline
from app.schemas.model_runtime import EmbedMode
from model_runtime.errors import ModelAuthError
from scripts.model_diagnostics_settings import ModelDiagnosticsSettings

pytestmark = [pytest.mark.gpu, pytest.mark.external]


async def test_authenticated_ready_requires_service_token() -> None:
    settings = ModelDiagnosticsSettings.load().model_runtime
    client = ModelRuntimeClient(settings)
    invalid = ModelRuntimeClient(
        settings.model_copy(update={"auth_token": SecretStr("invalid-acceptance-token")})
    )
    try:
        ready = await client.ready(deadline=Deadline(time.monotonic() + 2))
        assert ready.ready
        assert ready.metadata.precision == settings.precision
        with pytest.raises(ModelAuthError):
            await invalid.ready(deadline=Deadline(time.monotonic() + 2))
    finally:
        await client.aclose()
        await invalid.aclose()


async def test_embed_returns_1024_dense_and_nonempty_sparse() -> None:
    client = ModelRuntimeClient(ModelDiagnosticsSettings.load().model_runtime)
    try:
        result = await client.embed(
            ["七天无理由退货政策", "SKU-A1023 退款"],
            EmbedMode.QUERY,
            deadline=Deadline(time.monotonic() + 20),
        )
        assert len(result.dense) == len(result.sparse) == 2
        assert all(len(row) == 1024 for row in result.dense)
        assert all(row for row in result.sparse)
    finally:
        await client.aclose()


async def test_query_and_document_modes_share_encoding_semantics() -> None:
    client = ModelRuntimeClient(ModelDiagnosticsSettings.load().model_runtime)
    try:
        query = await client.embed(
            ["七天无理由退货政策"], EmbedMode.QUERY, deadline=Deadline(time.monotonic() + 20)
        )
        document = await client.embed(
            ["七天无理由退货政策"], EmbedMode.DOCUMENT, deadline=Deadline(time.monotonic() + 20)
        )
        assert query.dense == document.dense
        assert query.sparse == document.sparse
    finally:
        await client.aclose()


async def test_rerank_relevant_passage_higher_and_preserves_input_order() -> None:
    client = ModelRuntimeClient(ModelDiagnosticsSettings.load().model_runtime)
    try:
        passages = ["签收七天内未使用的商品可以申请退货。", "北京明天天气晴朗。"]
        first = await client.rerank(
            "七天退货政策是什么?", passages, deadline=Deadline(time.monotonic() + 30)
        )
        second = await client.rerank(
            "七天退货政策是什么?",
            list(reversed(passages)),
            deadline=Deadline(time.monotonic() + 30),
        )
        assert 0 <= first.scores[1] < first.scores[0] <= 1
        assert second.scores == pytest.approx(list(reversed(first.scores)), abs=1e-3)
    finally:
        await client.aclose()
