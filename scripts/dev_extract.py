"""Preview production memory extraction with no database connection or writes."""

import argparse
import asyncio
import sys
from time import monotonic
from typing import ClassVar

from pydantic import Field, ValidationError

from app.agents.contracts import Answer, EvidenceRefs
from app.core.config_models import LLMSettings
from app.core.deadline import Deadline
from app.core.errors import InsightPilotError
from app.core.settings_base import ProcessSettings
from app.db.models.turn import TurnRole, TurnStatus
from app.schemas.memory_extraction import MemoryExtraction, MemoryExtractionInput
from app.services.llm.service import LlmService
from app.services.memory.extract import extract


class ExtractionProcessSettings(ProcessSettings):
    """Only provider credentials are available to this read-only diagnostic."""

    process_name: ClassVar[str] = "extract"
    llm: LLMSettings
    timeout_s: float = Field(default=90, ge=0.01, le=300)


async def run(message: str, settings: ExtractionProcessSettings) -> MemoryExtraction:
    """A synthetic eligible envelope previews candidates; it is never persisted."""
    inputs = MemoryExtractionInput(
        role=TurnRole.ASSISTANT,
        status=TurnStatus.SUCCEEDED,
        user_message=message,
        answer=Answer(
            markdown="Diagnostic preview; no assistant answer supplied.",
            confidence=1,
            assumptions=[],
            sql="",
            evidence_refs=EvidenceRefs(),
            trace_id="memory-extraction-preview",
        ),
    )
    llm = LlmService(settings.llm)
    try:
        async with asyncio.timeout(settings.timeout_s):
            await llm.start()
            return await extract(inputs, llm, deadline=Deadline(monotonic() + settings.timeout_s))
    finally:
        async with asyncio.timeout(10):
            await llm.aclose()


def main(argv: list[str] | None = None) -> int:
    """Print candidates only on explicit request; failures never masquerade as empty output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message")
    parser.add_argument("--show-candidates", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(run(args.message, ExtractionProcessSettings.load()))
        if args.show_candidates:
            print(result.model_dump_json())
        else:
            print(f"Accepted candidates: {len(result.candidates)} (preview only; no writes)")
        return 0
    except ValidationError:
        print("Invalid extraction input or configuration.", file=sys.stderr)
    except InsightPilotError as exc:
        print(exc.code + ": " + exc.user_message, file=sys.stderr)
    except TimeoutError:
        print("Memory extraction timed out.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
