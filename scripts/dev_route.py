"""Inspect routing without dispatching specialists or opening database connections."""

import argparse
import asyncio
import sys
import time
from typing import ClassVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ValidationError

from app.agents.contracts import RouteDecision, RouterInput, RoutingContext
from app.agents.nodes.router import route_question
from app.agents.runtime import RoutingRuntime
from app.core.config_models import LLMSettings, RouterSettings
from app.core.deadline import Deadline
from app.core.errors import InsightPilotError, LlmConfigurationError
from app.core.llm_config import ModelRole
from app.core.settings_base import ProcessSettings
from app.services.llm.service import LlmService


class RouteProcessSettings(ProcessSettings):
    """No database, MCP, retrieval, GPU or application identity is required."""

    process_name: ClassVar[str] = "route"
    llm: LLMSettings | None = None
    router: RouterSettings = Field(default_factory=RouterSettings)
    timeout_s: float = Field(default=90, ge=0.01, le=300)


class LazyRoutingLlm:
    """Prefilter decisions need neither a provider credential nor an HTTP client."""

    def __init__(self, settings: LLMSettings | None) -> None:
        self.settings = settings
        self.service: LlmService | None = None

    async def generate_structured[T: BaseModel](
        self, role: ModelRole, messages: list[BaseMessage], schema: type[T], *, deadline: Deadline
    ) -> T:
        """Initialize once on the first classifier call and preserve its deadline."""
        deadline.check("router_llm_start")
        if self.settings is None:
            raise LlmConfigurationError()
        if self.service is None:
            self.service = LlmService(self.settings)
            await self.service.start()
        deadline.check("router_llm_started")
        return await self.service.generate_structured(role, messages, schema, deadline=deadline)

    async def aclose(self) -> None:
        """Release resources even if initialization or classification failed."""
        if self.service is not None:
            async with asyncio.timeout(10):
                await self.service.aclose()


async def run(inputs: RouterInput, settings: RouteProcessSettings) -> RouteDecision:
    """Use the exact production classifier with minimal process-owned dependencies."""
    llm = LazyRoutingLlm(settings.llm)
    try:
        return await route_question(
            inputs,
            RoutingRuntime(
                llm=llm,
                settings=settings.router,
                deadline=Deadline(time.monotonic() + settings.timeout_s),
            ),
        )
    finally:
        await llm.aclose()


def main(argv: list[str] | None = None) -> int:
    """Print a route decision or a safe typed error; clarification is a valid result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--summary", default="")
    args = parser.parse_args(argv)
    try:
        inputs = RouterInput(
            question=args.question, routing_context=RoutingContext(summary=args.summary)
        )
        result = asyncio.run(run(inputs, RouteProcessSettings.load()))
        print(result.model_dump_json())
        return 0
    except ValidationError:
        print("Invalid routing input or configuration.", file=sys.stderr)
    except InsightPilotError as exc:
        print(exc.code + ": " + exc.user_message, file=sys.stderr)
    except TimeoutError:
        print("Routing cleanup timed out.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
