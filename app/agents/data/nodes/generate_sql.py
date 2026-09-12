"""Generate a single candidate from immutable resolved metric inputs."""

import json
import re

import sqlglot
import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import Scope, traverse_scope

from app.agents.contracts import SqlGeneratorOutput
from app.agents.data.state import DataAgentState
from app.agents.multiturn import prior_queries
from app.agents.prompts import SQL_GENERATE
from app.agents.runtime import RuntimeContext
from app.core.errors import SqlGenerationError
from app.core.llm_config import ModelRole
from app.services.periods import build_date_context

logger = structlog.get_logger(__name__)


def build_messages(state: DataAgentState, ctx: RuntimeContext) -> list[BaseMessage]:
    """Render snapshots only; no fresh catalog read can change resolved meaning."""
    expected = {(b.metric_key, b.definition_version) for b in state.metric_bindings}
    actual = {(e.metric_key, e.definition_version) for e in state.metric_examples}
    if (
        state.clarification is not None
        or not state.schema_block.strip()
        or not state.schema_tables
        or not expected
        or actual != expected
        or len(actual) != len(state.metric_examples)
    ):
        raise SqlGenerationError("Missing or inconsistent SQL generation context.")
    prompt = SQL_GENERATE.format(
        date_context=build_date_context(now=ctx.now),
        schema_block=state.schema_block,
        bindings=json.dumps(
            [b.model_dump(mode="json") for b in state.metric_bindings], ensure_ascii=False
        ),
        examples=json.dumps(
            [e.model_dump(mode="json") for e in state.metric_examples], ensure_ascii=False
        ),
    )
    return [
        SystemMessage(content=prompt),
        HumanMessage(
            content=json.dumps(
                {
                    "question": state.question,
                    "prior_queries_for_reference": prior_queries(state.prior_sql),
                },
                ensure_ascii=False,
            )
        ),
    ]


def clean_sql(query: str) -> str:
    """Strip only outer fencing and whitespace; preserve literal and comment bytes."""
    cleaned = query.strip()
    match = re.fullmatch(r"```(?:sql)?[ \t]*\r?\n(.*?)\r?\n```", cleaned, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else cleaned


def physical_tables(sql: str) -> list[str]:
    """Cross-check physical sources; this is not the MCP security validator."""
    try:
        parsed = sqlglot.parse(sql, read="postgres")
        queries = [item for item in parsed if item is not None]
        if len(queries) != 1 or not isinstance(queries[0], exp.Query):
            raise SqlGenerationError("Expected exactly one query.")
        query = normalize_identifiers(queries[0].copy(), dialect="postgres")
        return sorted(_scope_tables(traverse_scope(query)))
    except (SqlglotError, RecursionError) as exc:
        raise SqlGenerationError("The generated query could not be parsed.") from exc


def _scope_tables(scopes: list[Scope]) -> set[str]:
    tables: set[str] = set()
    for scope in scopes:
        _ = scope.selected_sources
        for source in scope.tables:
            if not isinstance(source.this, exp.Identifier):
                continue
            if not source.db and isinstance(scope.sources.get(source.alias_or_name), Scope):
                continue
            table = exp.Table(
                this=source.this.copy(),
                db=source.args.get("db"),
                catalog=source.args.get("catalog"),
            )
            tables.add(table.sql(dialect="postgres"))
    return tables


async def generate_sql(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Use the existing structured-output ladder; execution belongs to later nodes."""
    ctx = runtime.context
    ctx.deadline.check("generate_sql")
    output = await ctx.llm.generate_structured(
        ModelRole.SQL, build_messages(state, ctx), SqlGeneratorOutput, deadline=ctx.deadline
    )
    ctx.deadline.check("generate_sql_complete")
    sql = clean_sql(output.sql)
    if not sql:
        raise SqlGenerationError("The generated SQL is empty.")
    tables = physical_tables(sql)
    if set(output.tables_used) != set(tables):
        logger.warning(
            "sql_tables_mismatch", declared_count=len(output.tables_used), parsed_count=len(tables)
        )
    logger.info("sql_generated", table_count=len(tables))
    return Command(
        update={
            "generated_sql": sql,
            "tables_used": tables,
            "messages": [*state.messages, AIMessage(content=sql, name="sql_generator")],
        }
    )
