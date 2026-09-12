"""Generate and display SQL without executing the generated query or persisting a turn."""

import argparse
import asyncio
import time
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from uuid import uuid4

from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import Field

from app.agents.contracts import TurnIdentity
from app.agents.data.nodes.generate_sql import build_messages, generate_sql
from app.agents.data.nodes.resolve_metrics import resolve_metrics
from app.agents.data.nodes.select_schema import select_schema
from app.agents.data.state import DataAgentInput, DataAgentState
from app.agents.runtime import RuntimeContext
from app.clients.mcp_client import McpClient
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.errors import InsightPilotError, ValidationError
from app.db.session import Database
from app.schemas.mcp import Contract
from app.schemas.metric_resolution import RegionScope
from app.services.conversations import ConversationService
from app.services.evidence import EvidenceService
from app.services.llm.service import LlmService
from app.services.metrics import MetricService
from app.services.regions import RegionService
from app.services.schema_catalog import SchemaCatalogService
from app.services.schema_tokens import SchemaTokenCounter


class SqlDemoResult(Contract):
    """Printable demonstration artifact; never evidence of query execution."""

    prompt: list[str] = Field(default_factory=list)
    sql: str


def apply_update(state: DataAgentState, command: Command[str]) -> DataAgentState:
    """Validate each partial node update before invoking its dependent node."""
    if not isinstance(command.update, dict):
        raise ValidationError("Expected a state update from the demonstration node.")
    return DataAgentState.model_validate({**state.model_dump(), **command.update})


async def demonstrate(inputs: DataAgentInput, ctx: RuntimeContext) -> SqlDemoResult:
    """Run the three isolated nodes with injectable production or scripted services."""
    state = DataAgentState.model_validate(inputs.model_dump())
    runtime = Runtime(context=ctx)
    for node in (select_schema, resolve_metrics):
        command = await node(state, runtime)
        state = apply_update(state, command)
    if state.clarification is not None:
        raise ValidationError("The demonstration inputs require clarification.")
    messages = build_messages(state, ctx)
    command = await generate_sql(state, runtime)
    final = apply_update(state, command)
    return SqlDemoResult(
        prompt=[str(message.content) for message in messages], sql=final.generated_sql
    )


async def run(inputs: DataAgentInput) -> SqlDemoResult:
    """Compose only application-role resources; no business database credential."""
    settings = Settings.load()
    database = Database(settings.database)
    mcp = McpClient(settings.mcp)
    llm = LlmService(settings.llm)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(database.aclose)
        stack.push_async_callback(mcp.aclose)
        stack.push_async_callback(llm.aclose)
        database.start()
        await llm.start()
        ctx = RuntimeContext(
            llm=llm,
            mcp=mcp,
            evidence=EvidenceService(database),
            conversations=ConversationService(database),
            settings=settings,
            deadline=Deadline(time.monotonic() + settings.http.request_timeout_s),
            identity=TurnIdentity(user_id=uuid4(), conversation_id=uuid4(), turn_id=uuid4()),
            trace_id=uuid4().hex,
            schema_catalog=SchemaCatalogService(database, mcp, settings.schema_catalog),
            schema_token_counter=await asyncio.to_thread(SchemaTokenCounter),
            metrics=MetricService(database, settings.database),
            regions=RegionService(mcp),
            now=datetime.now(UTC),
        )
        return await demonstrate(inputs, ctx)


def main() -> int:
    """Region names use read-only catalog lookup; optional IDs supply an upstream default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--region-id", type=int, action="append", default=[])
    args = parser.parse_args()
    inputs = DataAgentInput(
        question=args.question,
        region_scope=RegionScope(region_ids=args.region_id) if args.region_id else None,
    )
    try:
        result = asyncio.run(run(inputs))
    except InsightPilotError as exc:
        print(exc.code + ": " + exc.user_message)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
