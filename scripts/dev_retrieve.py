"""Inspect registered retrieval candidates without generating answers or changing corpus data."""

import argparse
import asyncio
import sys
import time
from datetime import date, datetime
from typing import ClassVar
from zoneinfo import ZoneInfo

from pydantic import Field, ValidationError

from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import DatabaseSettings, ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.core.errors import InsightPilotError, PeriodUnresolved, RetrievalConfigurationError
from app.core.settings_base import ProcessSettings
from app.db.session import Database
from app.retrieval.config import RetrievalConfig, RetrievalSettings
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.search_store import HybridSearchStore
from app.schemas.retrieval import PolicyPeriod, PointTimeScope, RangeTimeScope, RetrievalQuery


class RetrievalProcessSettings(ProcessSettings):
    """The diagnostic process accepts only app DML, retrieval and optional model settings."""

    process_name: ClassVar[str] = "retrieve"
    database: DatabaseSettings
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    model_runtime: ModelRuntimeClientSettings | None = None
    timeout_s: float = Field(default=90, ge=0.01, le=300)


def parse_query(args: argparse.Namespace, now: datetime) -> RetrievalQuery:
    """CLI time is explicit ISO dates; a missing date records the Shanghai default."""
    try:
        if args.period:
            periods = []
            for value in args.period:
                start, end = value.split(":")
                periods.append(PolicyPeriod(start=date.fromisoformat(start), end=date.fromisoformat(end), label=value))
            return RetrievalQuery(standalone=args.query, time_scope=RangeTimeScope(periods=periods))
        day = date.fromisoformat(args.as_of) if args.as_of else now.astimezone(ZoneInfo("Asia/Shanghai")).date()
        assumptions = [] if args.as_of else [f"未指定日期，按 Asia/Shanghai 的 {day.isoformat()} 查询。"]
        return RetrievalQuery(standalone=args.query, time_scope=PointTimeScope(as_of=day), assumptions=assumptions)
    except (ValueError, ValidationError) as exc:
        raise PeriodUnresolved() from exc


def search_config(args: argparse.Namespace, configured: RetrievalConfig) -> RetrievalConfig:
    """Explicit flags override typed settings; diagnostics record arm scores by default."""
    values = configured.model_dump()
    if args.arms is not None:
        arms = args.arms.split(",")
        if not arms or len(set(arms)) != len(arms) or set(arms) - {"dense", "sparse", "bm25"}:
            raise RetrievalConfigurationError(reason="invalid_arms")
        values.update(use_dense="dense" in arms, use_sparse_learned="sparse" in arms, use_bm25="bm25" in arms)
    if args.record_arm_scores is not None:
        values["record_arm_scores"] = args.record_arm_scores
    elif "record_arm_scores" not in configured.model_fields_set:
        values["record_arm_scores"] = True
    return RetrievalConfig.model_validate(values)


async def run(settings: RetrievalProcessSettings, query: RetrievalQuery) -> int:
    """Close all process-owned resources after success, failure or cancellation."""
    database = Database(settings.database)
    database.start()
    config = settings.retrieval.search
    needs_model = config.use_dense or config.use_sparse_learned
    model = ModelRuntimeClient(settings.model_runtime) if needs_model and settings.model_runtime else None
    try:
        async with HybridSearchStore(settings.retrieval.milvus) as store:
            result = await RetrievalPipeline(database, store, model, settings.retrieval).retrieve(
                query, deadline=Deadline(time.monotonic() + settings.timeout_s),
            )
            print(result.model_dump_json())
            return 0
    finally:
        try:
            if model is not None:
                async with asyncio.timeout(10):
                    await model.aclose()
        finally:
            async with asyncio.timeout(10):
                await database.aclose()


def parser() -> argparse.ArgumentParser:
    """Expose only typed arm/date choices, never raw Milvus expressions."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("query")
    result.add_argument("--arms")
    dates = result.add_mutually_exclusive_group()
    dates.add_argument("--as-of")
    dates.add_argument("--period", action="append")
    result.add_argument("--record-arm-scores", action=argparse.BooleanOptionalAction, default=None)
    return result


def main(argv: list[str] | None = None) -> int:
    """Print fixed safe errors with nonzero exits; empty candidate pools succeed."""
    args = parser().parse_args(argv)
    try:
        query = parse_query(args, datetime.now(ZoneInfo("Asia/Shanghai")))
        settings = RetrievalProcessSettings.load()
        settings.retrieval.search = search_config(args, settings.retrieval.search)
        return asyncio.run(run(settings, query))
    except ValidationError:
        print("Invalid retrieval configuration.", file=sys.stderr)
    except InsightPilotError as exc:
        print(exc.code + ": " + exc.user_message, file=sys.stderr)
    except TimeoutError:
        print("Retrieval cleanup timed out.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
