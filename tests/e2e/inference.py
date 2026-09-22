"""CPU-only HTTP substitutes; deliberately separate from production image contents."""

import asyncio
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.schemas.model_runtime import (
    EMBED_REVISION, RERANK_REVISION, EmbedMode, EmbedRequest, EmbedResult, ModelMetadata,
    ReadyResult, RerankRequest, RerankResult,
)
from app.services.llm.contracts import CompletionRequest
from tests.e2e.contracts import Scenario, ScriptRequest, ScriptStatus
from tests.e2e.scripts import Script


def metadata() -> ModelMetadata:
    return ModelMetadata(embed_revision=EMBED_REVISION, rerank_revision=RERANK_REVISION,
                         precision="fp16", embed_batch=16, rerank_batch=16,
                         embed_max_length=512, rerank_max_length=320)


def application() -> FastAPI:
    app = FastAPI()
    script = Script(Scenario.CLARIFY)
    release = asyncio.Event()
    release.set()

    @app.get("/ready")
    async def ready() -> ReadyResult:
        return ReadyResult(request_id=uuid4().hex, metadata=metadata())

    @app.post("/_e2e/script")
    async def configure(request: ScriptRequest) -> ScriptStatus:
        nonlocal script
        script = Script(request.scenario)
        release.clear() if request.scenario is Scenario.CHAOS else release.set()
        return script.snapshot()

    @app.get("/_e2e/status")
    async def status() -> ScriptStatus:
        return script.snapshot()

    @app.post("/_e2e/release")
    async def unblock() -> ScriptStatus:
        release.set()
        return script.snapshot()

    @app.post("/v1/chat/completions")
    async def completion(request: CompletionRequest) -> JSONResponse:
        try:
            if (script.scenario is Scenario.CHAOS and request.response_format is not None
                and request.response_format["json_schema"]["schema"]["title"] == "SqlGeneratorOutput"):
                script.status.sql_waiting = True
                async with asyncio.timeout(45):
                    await release.wait()
            content = await script.complete(request)
        except Exception as exc:
            script.status.errors.append(type(exc).__name__)
            return JSONResponse({"error": {"code": "e2e_script_mismatch"}}, status_code=400)
        return JSONResponse({"choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": content}}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})

    @app.post("/v1/embed")
    async def embed(request: EmbedRequest) -> EmbedResult:
        if request.mode is EmbedMode.QUERY:
            script.status.embed_query += 1
        else:
            script.status.embed_document += 1
        return EmbedResult(request_id=uuid4().hex, ms=0, queue_ms=0, inference_ms=0,
                           metadata=metadata(), dense=[[1.0, *([0.0]*1023)] for _ in request.texts],
                           sparse=[{42: 1.0} for _ in request.texts])

    @app.post("/v1/rerank")
    async def rerank(request: RerankRequest) -> RerankResult:
        script.status.rerank += 1
        return RerankResult(request_id=uuid4().hex, ms=0, queue_ms=0, inference_ms=0,
                            metadata=metadata(), scores=[0.9 for _ in request.passages])

    return app
