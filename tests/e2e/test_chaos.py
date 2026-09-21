"""Stop a real, fixture-owned MCP container during a BOTH HTTP turn."""

from __future__ import annotations

import asyncio
import json
import secrets
import socket
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr

from app.agents.contracts import Route
from app.clients.mcp_client import McpClient
from app.core.config_models import DatabaseSettings, MCPSettings
from app.services.graph import GraphService
from scripts.ci_process import CommandRecorder, retain_primary_failure
from scripts.deployment import ROOT, process_environment
from tests.agents.parent_support import parent_context
from tests.agents.synthesis_support import synthesis_draft
from tests.api.chat_support import Harness, chat
from tests.database_support import DatabaseStack, foreign_snapshot
from tests.fakes.chat_model import FakeChatModel
from tests.integration.checkpoint_support import checkpoint_setup, graph_database
from tests.retrieval_support import RetrievalHarness, harness

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import httpx

    from app.core.deadline import Deadline
    from app.db.session import Database
    from app.schemas.mcp import QueryArguments, QueryResultPayload
    from app.schemas.retrieval import RetrievalQuery, RetrievalResult
    from tests.milvus_support import MilvusStack

pytestmark = [pytest.mark.e2e, pytest.mark.integration, pytest.mark.storage]
__all__ = ["chat", "checkpoint_setup", "graph_database", "harness"]


@dataclass
class ChaosMcp:
    command: list[str]
    recorder: CommandRecorder
    settings: MCPSettings
    directory: Path


@pytest.fixture
def chaos_mcp(
    migration_stack: DatabaseStack,
    checkpoint_setup: None,
    milvus_stack: MilvusStack,
    tmp_path: Path,
) -> Iterator[ChaosMcp]:
    settings = migration_stack.settings.model_copy(
        update={"mcp_auth_token": SecretStr(secrets.token_urlsafe(24))}
    )
    before = foreign_snapshot(migration_stack.docker)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    overlay = tmp_path / "mcp-chaos.yml"
    overlay.write_text(
        "services:\n  mcp:\n"
        f"    image: {settings.compose_project_name}-mcp:step410\n"
        f"    ports: ['127.0.0.1:{port}:8001']\n"
        "    networks: [backend, egress]\n"
    )
    command = [
        migration_stack.docker,
        "compose",
        "-p",
        settings.compose_project_name,
        "--project-directory",
        str(ROOT),
        "--env-file",
        str(ROOT / ".env.deployment.example"),
        "-f",
        str(ROOT / "docker-compose.yml"),
        "-f",
        str(overlay),
        "--profile",
        "core",
    ]
    directory = milvus_stack.directory / "step410-chaos"
    recorder = CommandRecorder(
        directory=directory / "commands",
        cwd=ROOT,
        environment=process_environment(settings),
        secrets=[
            settings.postgres_superuser_password,
            settings.bootstrap.app_password,
            settings.bootstrap.mcp_password,
            settings.bootstrap.etl_password,
            settings.mcp_auth_token,
        ],
    )
    with retain_primary_failure(
        [
            lambda: recorder.run("logs", [*command, "logs", "--no-color", "mcp"]),
            lambda: recorder.run("stop", [*command, "stop", "-t", "1", "mcp"]),
        ]
    ):
        recorder.run("build-mcp", [*command, "build", "mcp"], timeout=300)
        recorder.run(
            "start-mcp",
            [*command, "up", "-d", "--no-deps", "--wait", "--wait-timeout", "90", "mcp"],
            timeout=120,
        )
        yield ChaosMcp(
            command,
            recorder,
            MCPSettings(
                base_url=f"http://127.0.0.1:{port}/mcp", auth_token=settings.mcp_auth_token
            ),
            directory,
        )
    assert foreign_snapshot(migration_stack.docker) == before


async def test_mcp_killed_midturn(
    chat: Harness,
    chaos_mcp: ChaosMcp,
    harness: RetrievalHarness,
    graph_database: tuple[Database, DatabaseSettings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, settings = graph_database
    graph = GraphService(settings)
    await graph.start()
    mcp = McpClient(chaos_mcp.settings)
    await mcp.connect()
    ctx = parent_context(Route.BOTH, settings=chat.app.state.settings)
    llm = FakeChatModel(list(ctx.llm._responses)[:-1])
    pipeline = harness.pipeline()
    data_entered = asyncio.Event()
    knowledge_ready = asyncio.Event()
    stopped = asyncio.Event()
    original_call = mcp.call_tool
    original_retrieve = pipeline.retrieve

    async def call(name: str, args: QueryArguments, *, deadline: Deadline) -> QueryResultPayload:
        data_entered.set()
        await stopped.wait()
        return await original_call(name, args, deadline=deadline)

    async def retrieve(query: RetrievalQuery, *, deadline: Deadline) -> RetrievalResult:
        result = await original_retrieve(query, deadline=deadline)
        assert result.candidates
        llm.enqueue(synthesis_draft(data=False, knowledge_id=result.candidates[0].chunk_uuid))
        knowledge_ready.set()
        return result

    monkeypatch.setattr(mcp, "call_tool", call)
    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    chat.app.state.mcp = mcp
    chat.app.state.llm = llm
    chat.app.state.retrieval = pipeline
    chat.app.state.chat.graph = graph
    try:
        async with asyncio.timeout(60), asyncio.TaskGroup() as tasks:
            request = tasks.create_task(
                chat.client.post(chat.url, json={"content": "请分析2026年8月经营数据与退款政策"})
            )
            await data_entered.wait()
            await knowledge_ready.wait()
            await asyncio.to_thread(
                chaos_mcp.recorder.run,
                "kill-midturn",
                [*chaos_mcp.command, "stop", "-t", "1", "mcp"],
            )
            stopped.set()
        await assert_partial(chat, request.result(), chaos_mcp.directory)
    finally:
        await mcp.aclose()
        await graph.aclose()


async def assert_partial(chat: Harness, response: httpx.Response, directory: Path) -> None:
    assert response.status_code == HTTPStatus.OK, response.text
    result = response.json()
    assert result["status"] == "degraded"
    assert result["answer"]["degraded_components"] == ["data"]
    assert "数据源当前不可用" in result["answer"]["markdown"]
    assert result["answer"]["citations"]
    assert result["evidence_refs"]["data_snapshot_id"] is None
    assert result["evidence_refs"]["knowledge_snapshot_id"]
    assert (await chat.stored())[-1]["answer"] == result["answer"]
    evidence = await chat.client.get(
        f"/api/v1/conversations/{chat.cid}/turns/{result['id']}/evidence"
    )
    assert evidence.status_code == HTTPStatus.OK
    assert evidence.json()["knowledge"]["id"] == result["evidence_refs"]["knowledge_snapshot_id"]
    directory.joinpath("response.json").write_text(
        json.dumps({"response": result, "evidence": evidence.json()}, ensure_ascii=False, indent=2)
    )
