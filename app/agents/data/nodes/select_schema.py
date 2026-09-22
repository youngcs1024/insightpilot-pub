"""Full-schema selection with reproducible measurement."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.data.state import DataAgentState
from app.agents.runtime import RuntimeContext
from app.core.observability import TraceMetadata, update_current_observation
from app.schemas.schema_tools import GetSchemaArgs


async def select_schema(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Render complete catalog semantics through the injected service."""
    ctx = runtime.context
    ctx.deadline.check("select_schema")
    response = await ctx.mcp.get_schema(
        GetSchemaArgs(include_samples=True), deadline=ctx.deadline
    )
    block = response.rendered
    tables = [table.table_name for table in response.tables]
    tokens = ctx.schema_token_counter.count(block)
    update_current_observation(
        TraceMetadata(
            schema_strategy=ctx.settings.data_agent.schema_strategy,
            schema_tables=len(tables),
            schema_tokens=tokens,
            schema_tokenizer=ctx.schema_token_counter.name,
            schema_utf8_bytes=len(block.encode("utf-8")),
        )
    )
    return Command(update={"schema_block": block, "schema_tables": tables})
