"""Measure the current tool surface without starting MCP or model services."""

import asyncio
import json
from typing import cast, get_args

from pydantic import BaseModel, Field, SecretStr

from app.agents.tools.native import NativeKind, definitions
from app.services.schema_tokens import SchemaTokenCounter
from mcp_server.config import AuditSettings, BusinessSettings, McpServerSettings, ServerSettings
from mcp_server.server import create_server

RETRIEVAL_TOP_K = 3


class ToolSurfaceAudit(BaseModel):
    """Reproducible capability inventory and model-visible prompt estimate."""

    mcp_tool_names: list[str]
    model_visible_tool_names: list[str]
    model_visible_tool_count: int = Field(ge=0)
    model_visible_tool_block_tokens: int = Field(ge=0)
    tokenizer: str = "cl100k_base"
    retrieval_top_k: int = Field(gt=0)
    top_k_prunable_tool_count: int = Field(ge=0)


def _offline_server_settings() -> McpServerSettings:
    """Build typed dummy settings without dotenv or environment discovery."""
    return McpServerSettings.model_construct(
        business=BusinessSettings(password=SecretStr("offline-audit-business")),
        audit=AuditSettings(password=SecretStr("offline-audit-audit")),
        mcp=ServerSettings(auth_token=SecretStr("offline-audit-token-0000000000000000")),
    )


async def audit() -> ToolSurfaceAudit:
    """Read registered descriptors; only native tools enter the model payload."""
    server = create_server(_offline_server_settings())
    mcp_names = sorted(tool.name for tool in await server.list_tools())
    visible = definitions(cast("list[NativeKind]", list(get_args(NativeKind.__value__))))
    wire_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in visible
    ]
    tool_block = json.dumps(wire_tools, ensure_ascii=False, separators=(",", ":"))
    return ToolSurfaceAudit(
        mcp_tool_names=mcp_names,
        model_visible_tool_names=[tool.name for tool in visible],
        model_visible_tool_count=len(visible),
        model_visible_tool_block_tokens=SchemaTokenCounter().count(tool_block),
        retrieval_top_k=RETRIEVAL_TOP_K,
        top_k_prunable_tool_count=max(0, len(visible) - RETRIEVAL_TOP_K),
    )


def main() -> int:
    """Print one machine-readable offline observation."""
    print(asyncio.run(audit()).model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
