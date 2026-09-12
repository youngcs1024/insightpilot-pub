"""Credential-scoped live composition; all business data access goes through MCP."""

import asyncio
import hashlib
import subprocess
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from uuid import uuid4

from app.agents.contracts import TurnIdentity
from app.agents.runtime import RuntimeContext
from app.clients.mcp_client import McpClient
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.llm_config import ModelRole
from app.db.session import Database
from app.schemas.mcp import QueryArguments, QueryResultPayload
from app.services.conversations import ConversationService
from app.services.evidence import EvidenceService
from app.services.llm.service import LlmService
from app.services.metrics import MetricService
from app.services.regions import RegionService
from app.services.schema_catalog import SchemaCatalogService
from app.services.schema_tokens import SchemaTokenCounter
from data.seed.contracts import TABLE_NAMES, Manifest, Parameters
from evals.harness.contracts import CASES, REFERENCE_TIME, ConfigSnapshot, EvaluationError, Options

PROJECT = CASES.parents[3]


def digest(value: bytes) -> str:
    """Stable content identity independent of local paths."""
    return hashlib.sha256(value).hexdigest()


def git_value(*arguments: str) -> str:
    """Bounded read-only metadata lookup; never print environment or credentials."""
    try:
        return subprocess.run(  # noqa: S603 -- fixed metadata arguments from this module.
            ["git", *arguments],  # noqa: S607 -- repository tooling requires Git on PATH.
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvaluationError("Cannot identify evaluated source") from exc


async def snapshot(ctx: RuntimeContext, options: Options) -> ConfigSnapshot:
    """Snapshot public inputs; the operator provisions the matching seed export."""
    manifest = Manifest.model_validate_json(
        await asyncio.to_thread(options.seed_manifest.read_bytes)
    )
    if (
        manifest.parameters != Parameters()
        or manifest.format_version != 1
        or manifest.design_version != "2026-09-07-step2.1"
        or tuple(entry.table for entry in manifest.files) != TABLE_NAMES
    ):
        raise EvaluationError("Suite v1 requires seed 42 / 50000 orders / 18 months")
    await verify_seed_counts(ctx, manifest)
    definitions = await ctx.metrics.list_active(deadline=ctx.deadline)
    schema = await ctx.schema_catalog.snapshot(deadline=ctx.deadline)
    prompts = PROJECT / "app/agents/prompts"
    role = ctx.settings.llm.for_role(ModelRole.SQL)
    return ConfigSnapshot(
        model=role.model or ctx.settings.llm.model,
        sql_role=role,
        prompt_hashes={
            name: digest(await asyncio.to_thread((prompts / name).read_bytes))
            for name in (
                "metric_intent.md",
                "sql_generate.md",
                "sql_correct.md",
                "schema.md",
                "structured_json.md",
                "structured_repair.md",
            )
        },
        catalog_versions={definition.key: definition.version for definition in definitions},
        catalog_hash=digest("\n".join(d.model_dump_json() for d in definitions).encode()),
        schema_hash=digest(schema.model_dump_json().encode()),
        dataset_hash=digest(await asyncio.to_thread(CASES.read_bytes)),
        seed_manifest=manifest,
        git_sha=await asyncio.to_thread(git_value, "rev-parse", "HEAD"),
        source_dirty=bool(await asyncio.to_thread(git_value, "status", "--porcelain")),
    )


async def verify_seed_counts(ctx: RuntimeContext, manifest: Manifest) -> None:
    """Check database counts against the operator's export, without an operator credential."""
    # Fixed identifiers are independent of CLI/YAML input.
    sql = """SELECT
        (SELECT COUNT(*) FROM biz.regions) AS regions,
        (SELECT COUNT(*) FROM biz.promotions) AS promotions,
        (SELECT COUNT(*) FROM biz.products) AS products,
        (SELECT COUNT(*) FROM biz.customers) AS customers,
        (SELECT COUNT(*) FROM biz.orders) AS orders,
        (SELECT COUNT(*) FROM biz.order_items) AS order_items,
        (SELECT COUNT(*) FROM biz.refunds) AS refunds,
        (SELECT COUNT(*) FROM biz.inventory) AS inventory"""
    actual = await execute(ctx, sql)
    if actual.result_truncated or actual.rows != [[entry.rows for entry in manifest.files]]:
        raise EvaluationError("Database row counts differ from the provisioned seed manifest")


def fresh(ctx: RuntimeContext) -> RuntimeContext:
    """Every independent case receives one fresh request budget and identity."""
    return replace(
        ctx,
        deadline=Deadline(time.monotonic() + ctx.settings.http.request_timeout_s),
        identity=TurnIdentity(user_id=uuid4(), conversation_id=uuid4(), turn_id=uuid4()),
        trace_id=uuid4().hex,
    )


async def execute(ctx: RuntimeContext, sql: str) -> QueryResultPayload:
    """Use the same MCP query cap as production; a truncated oracle is invalid."""
    return await ctx.mcp.call_tool(
        "execute_readonly_query", QueryArguments(sql=sql), deadline=fresh(ctx).deadline
    )


@asynccontextmanager
async def live_context(settings: Settings) -> AsyncIterator[RuntimeContext]:
    """Release all resources even after failed evaluation."""
    database = Database(settings.database)
    mcp = McpClient(settings.mcp)
    llm = LlmService(settings.llm)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(database.aclose)
        stack.push_async_callback(mcp.aclose)
        stack.push_async_callback(llm.aclose)
        database.start()
        await llm.start()
        yield RuntimeContext(
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
            now=REFERENCE_TIME,
        )
