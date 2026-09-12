"""Scripted provider replies and bounded real LLM service lifecycle."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from app.core.config_models import LLMSettings
from app.services.llm.registry import CapabilityReport, ModelRegistry, StructuredTier
from app.services.llm.service import LlmService

URL = "https://provider.invalid/v1/chat/completions"
GOOD = '{"count":7,"region":{"name":"华东"}}'


def response(content: str = GOOD, *, tool_name: str | None = None) -> httpx.Response:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_name is not None:
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": tool_name, "arguments": content},
            }
        ]
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": message, "finish_reason": "tool_calls" if tool_name else "stop"}
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 9},
        },
    )


def report(model: str, tier: StructuredTier) -> CapabilityReport:
    return CapabilityReport(
        schema_version=1,
        evidence_source="live_api",
        model=model,
        execution_complete=True,
        recommended_tier=tier,
    )


@asynccontextmanager
async def service(
    tier: StructuredTier = StructuredTier.NATIVE,
    *,
    fallback: bool = False,
) -> AsyncIterator[LlmService]:
    config = LLMSettings(
        base_url="https://provider.invalid/v1",
        model="primary",
        api_key="test-key-only",
        roles={"sql": {"fallback_models": ["backup"] if fallback else []}},
    )
    registry = ModelRegistry(config, [report("primary", tier), report("backup", tier)])
    llm = LlmService(config, registry=registry)
    await llm.start()
    try:
        yield llm
    finally:
        await llm.aclose()
