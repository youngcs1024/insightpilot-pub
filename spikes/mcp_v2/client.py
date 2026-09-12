"""Executable acceptance checks for MCP v2 and a minimal LangChain adapter."""

import asyncio
import hashlib
import json
from http import HTTPStatus
from importlib.metadata import version

import httpx2
import structlog
from langchain_core.tools import StructuredTool
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import BaseModel, ConfigDict, Field

from spikes.mcp_v2.settings import ENDPOINT, Settings

logger = structlog.get_logger()
CALL_COUNT = 3


class EchoOutput(BaseModel):
    """Typed interpretation of the SDK's scalar return wrapper."""

    model_config = ConfigDict(extra="forbid")
    result: str


class WireEvidence(BaseModel):
    """Observe session reuse without retaining bearer tokens or raw session IDs."""

    initialize_requests: int = 0
    call_sessions: list[str] = Field(default_factory=list)
    issued_sessions: list[str] = Field(default_factory=list)

    async def request(self, request: httpx2.Request) -> None:
        """Count protocol operations at the HTTP boundary."""
        if request.method != "POST":
            return
        payload = json.loads(request.content)
        if payload.get("method") == "initialize":
            self.initialize_requests += 1
        if payload.get("method") == "tools/call":
            session_id = request.headers.get("mcp-session-id", "")
            require(bool(session_id), "tool_call_missing_session")
            self.call_sessions.append(fingerprint(session_id))

    async def response(self, response: httpx2.Response) -> None:
        """Record only a digest of server-issued session identifiers."""
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self.issued_sessions.append(fingerprint(session_id))


def fingerprint(value: str) -> str:
    """Return a one-way session identifier for the acceptance transcript."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def require(condition: bool, event: str) -> None:
    """Fail the executable acceptance check without exposing request secrets."""
    if not condition:
        raise SystemExit(event)


def check_result(result: CallToolResult, expected: str) -> EchoOutput:
    """Verify both structured and textual content, then print the wire model."""
    require(not result.is_error, "echo_tool_error")
    output = EchoOutput.model_validate(result.structured_content)
    require(output.result == expected, "structured_echo_mismatch")
    require(
        any(isinstance(item, TextContent) and item.text == expected for item in result.content),
        "text_echo_mismatch",
    )
    print(result.model_dump_json(by_alias=True, exclude_none=True, exclude_unset=True))
    return output


def adapt_echo(descriptor: Tool, session: ClientSession) -> StructuredTool:
    """Adapt the discovered echo descriptor; this is not a general-purpose adapter."""

    async def invoke(text: str) -> EchoOutput:
        result = await session.call_tool(descriptor.name, arguments={"text": text})
        require(isinstance(result, CallToolResult), "unexpected_tool_result")
        return check_result(result, text)

    return StructuredTool.from_function(
        coroutine=invoke,
        name=descriptor.name,
        description=descriptor.description or "Echo text",
        args_schema=descriptor.input_schema,
    )


async def check_auth(settings: Settings) -> None:
    """Prove missing and incorrect bearer tokens fail before MCP dispatch."""
    async with httpx2.AsyncClient(timeout=settings.mcp.timeout_seconds, trust_env=False) as http:
        for label, headers in (
            ("missing", {}),
            (
                "incorrect",
                {"Authorization": "Bearer " + settings.mcp.auth_token.get_secret_value() + "x"},
            ),
        ):
            response = await http.get(ENDPOINT, headers=headers)
            require(response.status_code == HTTPStatus.UNAUTHORIZED, "auth_not_rejected")
            logger.info("auth_rejected", case=label, status=response.status_code)


async def check_session(settings: Settings) -> None:
    """Initialize once and execute all three tool calls in that same session."""
    evidence = WireEvidence()
    async with (
        httpx2.AsyncClient(
            headers={"Authorization": "Bearer " + settings.mcp.auth_token.get_secret_value()},
            timeout=settings.mcp.timeout_seconds,
            trust_env=False,
            event_hooks={"request": [evidence.request], "response": [evidence.response]},
        ) as http,
        streamable_http_client(ENDPOINT, http_client=http) as (read, write),
        ClientSession(read, write, read_timeout_seconds=settings.mcp.timeout_seconds) as session,
    ):
        initialized = await session.initialize()
        print(initialized.model_dump_json(by_alias=True, exclude_none=True, exclude_unset=True))
        listing = await session.list_tools()
        print(listing.model_dump_json(by_alias=True, exclude_none=True, exclude_unset=True))
        require(len(listing.tools) == 1 and listing.tools[0].name == "echo", "unexpected_tools")
        for text in ("first echo", "第二次 echo"):
            result = await session.call_tool("echo", arguments={"text": text})
            require(isinstance(result, CallToolResult), "unexpected_tool_result")
            check_result(result, text)
        tool = adapt_echo(listing.tools[0], session)
        adapted = await tool.ainvoke({"text": "LangChain echo"})
        require(
            isinstance(adapted, EchoOutput) and adapted.result == "LangChain echo", "adapter_failed"
        )
    require(evidence.initialize_requests == 1, "multiple_initializations")
    require(len(evidence.call_sessions) == CALL_COUNT, "unexpected_call_count")
    require(len(set(evidence.call_sessions)) == 1, "session_not_reused")
    require(
        set(evidence.call_sessions) == set(evidence.issued_sessions), "session_not_server_issued"
    )
    print(evidence.model_dump_json())
    logger.info("spike_passed", calls=CALL_COUNT, langchain_call=True, session_reused=True)


async def main() -> None:
    """Run bounded live checks with no automatic retries."""
    settings = Settings()
    logger.info("probe_versions", mcp=version("mcp"), langchain_core=version("langchain-core"))
    async with asyncio.timeout(settings.mcp.timeout_seconds * 8):
        await check_auth(settings)
        await check_session(settings)


if __name__ == "__main__":
    asyncio.run(main())
