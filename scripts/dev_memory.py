"""Read-only operator explanation of the production memory eligibility rules."""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from time import monotonic
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import Field, ValidationError

from app.agents.contracts import Route, RouteDecision, RouterInput, RoutingContext
from app.agents.nodes.router import route_question
from app.agents.runtime import RoutingRuntime
from app.core.config_models import DatabaseSettings, LLMSettings, RouterSettings
from app.core.deadline import Deadline
from app.core.errors import InsightPilotError
from app.core.settings_base import ProcessSettings
from app.db.session import Database
from app.schemas.intent import MetricIntentInput
from app.schemas.mcp import Contract
from app.schemas.memory import FormatPreferenceContent, TerminologyContent, TerminologyProjection
from app.schemas.memory_retrieval import MemoryReadRequest, MemorySelection, MemoryStage
from app.services.memory.service import MemoryService
from app.services.metric_intent import MetricIntentService
from app.services.metrics import MetricService
from app.services.schema_tokens import SchemaTokenCounter
from scripts.dev_route import LazyRoutingLlm


class MemoryProcessSettings(ProcessSettings):
    """No business database credential, GPU or write service is constructed."""

    process_name: ClassVar[str] = "memory"
    database: DatabaseSettings
    llm: LLMSettings | None = None
    router: RouterSettings = Field(default_factory=RouterSettings)
    timeout_s: float = Field(default=90, ge=0.01, le=300)


class MemoryExplanation(Contract):
    """Provisional selection and the production restart decision, without execution."""

    schema_version: Literal[1] = 1
    route: RouteDecision
    preparation: MemorySelection
    final: MemorySelection
    restart_required: bool = False


async def run(user: UUID, question: str, settings: MemoryProcessSettings) -> MemoryExplanation:
    """Explain candidates without dispatching specialists, extracting or writing memory."""
    database = Database(settings.database)
    llm = LazyRoutingLlm(settings.llm)
    database.start()
    deadline = Deadline(monotonic() + settings.timeout_s)
    counter = SchemaTokenCounter()
    memories = MemoryService(database, settings.database)
    try:
        before = await memories.retrieve(
            MemoryReadRequest(user_id=user, question=question, stage=MemoryStage.PREPARE),
            deadline=deadline, counter=counter,
        )
        context = RoutingContext(
            terminology=[r.content for r in before.selected if isinstance(r.content, TerminologyContent)],
            format_preference=next((r.content for r in before.selected if isinstance(r.content, FormatPreferenceContent)), None),
        )
        route = await route_question(RouterInput(question=question, routing_context=context),
                                    RoutingRuntime(llm=llm, settings=settings.router, deadline=deadline))
        request = MemoryReadRequest(
            user_id=user, question=question, stage=MemoryStage.FINALIZE,
            data_route=route.route in {Route.DATA_ONLY, Route.BOTH},
            clarify=route.route is Route.CLARIFY, region_mentioned=route.region_mentioned,
        )
        if route.region.names or route.region.all_regions:
            request.region_mentioned = True
        if request.data_route:
            intent = await MetricIntentService(llm, MetricService(database, settings.database)).interpret(
                MetricIntentInput(question=question, data_intent=route.data_intent, metric_hints=route.metric_hints,
                                  terminology=[TerminologyProjection(**term.model_dump()) for term in context.terminology]),
                deadline=deadline, now=datetime.now(UTC),
            )
            request.metric_keys = list(intent.metric_keys)
            request.explicit_patch = intent.explicit_patch.model_copy(deep=True)
            request.region_mentioned = intent.region_mentioned or bool(intent.region.names) or intent.region.all_regions
        final = (MemorySelection(failed=True, failure_code=before.failure_code) if before.failed
                 else await memories.retrieve(request, deadline=deadline, counter=counter))
        used = {r.id for r in before.selected if isinstance(r.content, TerminologyContent)}
        kept = {r.id for r in final.selected if isinstance(r.content, TerminologyContent)}
        # This diagnostic stops before dispatch. Report the production restart decision,
        # and never misrepresent provisional candidates as the effective final selection.
        restart = bool(used - kept)
        if restart:
            final = final.model_copy(update={"selected": [], "tokens": 0})
        return MemoryExplanation(route=route, preparation=before, final=final, restart_required=restart)
    finally:
        try:
            await llm.aclose()
        finally:
            async with asyncio.timeout(10):
                await database.aclose()


def main(argv: list[str] | None = None) -> int:
    """Print only requested operator diagnostics; typed failures have nonzero exit codes."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--user", required=True, type=UUID)
    retrieve.add_argument("--question", required=True)
    retrieve.add_argument("--explain", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = asyncio.run(run(args.user, args.question, MemoryProcessSettings.load()))
        if args.explain:
            print(report.model_dump_json())
        else:
            print(f"Selected: {len(report.final.selected)}; tokens: {report.final.tokens}")
        return 1 if report.final.failed else 0
    except ValidationError:
        print("Invalid memory retrieval input or configuration.", file=sys.stderr)
    except InsightPilotError as exc:
        print(exc.code + ": " + exc.user_message, file=sys.stderr)
    except TimeoutError:
        print("Memory retrieval timed out.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
