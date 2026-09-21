"""Credential-scoped live routing with committed development provenance."""

import asyncio
import hashlib
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from uuid import uuid4

import structlog

from app.agents.contracts import RouterInput
from app.agents.nodes.router import classify_question
from app.agents.runtime import RoutingRuntime
from app.core.deadline import Deadline
from app.core.errors import InsightPilotError, LlmConfigurationError
from app.core.llm_config import ModelRole
from app.core.routing import RoutingStrategy
from app.services.corpus_sources import parse_yaml
from app.services.llm.usage import collect_usage
from evals.harness.contracts import EvaluationError
from evals.harness.routing_contracts import (
    CASES,
    CONCURRENCY,
    ROOT,
    SELECTION,
    Attempt,
    Case,
    Measurements,
    Observation,
    Options,
    Selection,
    Snapshot,
    Split,
)
from evals.harness.routing_dataset import load_cases
from scripts.ci_changes import git_bytes
from scripts.dev_route import LazyRoutingLlm, RouteProcessSettings

logger = structlog.get_logger(__name__)


def digest(value: bytes) -> str:
    """Content identity is independent of machine paths."""
    return hashlib.sha256(value).hexdigest()


def snapshot(settings: RouteProcessSettings) -> Snapshot:
    """Capture source, dataset and public configuration before any provider calls."""
    if settings.llm is None:
        raise LlmConfigurationError()
    paths = (
        git_bytes(
            [
                "ls-files",
                "app",
                "evals/harness",
                "evals/cli.py",
                "scripts/dev_route.py",
                "uv.lock",
                "pyproject.toml",
            ]
        )
        .decode()
        .splitlines()
    )
    source = hashlib.sha256()
    for relative in sorted(paths):
        source.update(relative.encode() + b"\0" + (ROOT / relative).read_bytes() + b"\0")
    return Snapshot(
        git_sha=git_bytes(["rev-parse", "HEAD"]).decode().strip(),
        source_dirty=bool(git_bytes(["status", "--porcelain", "--untracked-files=normal"])),
        dataset_hash=digest(CASES.read_bytes()),
        source_hash=source.hexdigest(),
        prompt_hashes={
            name: digest((ROOT / "app/agents/prompts" / name).read_bytes())
            for name in ("router.md", "clarify.md", "structured_json.md", "structured_repair.md")
        },
        capability_hashes=[digest(path.read_bytes()) for path in settings.llm.capabilities_paths],
        model=settings.llm.for_role(ModelRole.ROUTER),
        min_confidence=settings.router.min_confidence,
        timeout_s=settings.timeout_s,
        production_strategy=settings.router.strategy,
    )


def committed_selection(config: Snapshot, path: Path = SELECTION) -> tuple[Selection, str]:
    """Validate the saved choice before loading frozen cases or making calls."""
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise EvaluationError("Selection must be a committed repository file")
    relative = resolved.relative_to(ROOT).as_posix()
    content = path.read_text(encoding="utf-8")
    if git_bytes(["show", f"HEAD:{relative}"]).decode() != content:
        raise EvaluationError("Selection differs from committed source")
    selected = parse_yaml(content, Selection)
    commit = git_bytes(["log", "-1", "--format=%H", "--", relative]).decode().strip()
    git_bytes(["merge-base", "--is-ancestor", selected.development_sha, commit])
    if (
        selected.fingerprint != config.fingerprint()
        or selected.arm is not config.production_strategy
    ):
        raise EvaluationError("Committed strategy or evaluated inputs differ from development")
    if config.source_dirty:
        raise EvaluationError("Frozen evaluation requires clean committed source")
    return selected, commit


async def observe(inputs: RouterInput, ctx: RoutingRuntime, arm: RoutingStrategy) -> Observation:
    """Nested provider collectors propagate actual usage, including retries and repairs."""
    start = time.monotonic()
    result = Observation(latency_ms=0)
    with collect_usage() as usage:
        try:
            result.decision = await classify_question(inputs, ctx, arm)
            result.prefilter_hit = (
                result.decision is not None and result.decision.decided_by == "prefilter"
            )
        except InsightPilotError as exc:
            result.failure_code = exc.code
        finally:
            result.latency_ms = (time.monotonic() - start) * 1000
    result.tokens = (
        0
        if usage.attempts == 0 and (result.prefilter_hit or arm is RoutingStrategy.PREFILTER_ONLY)
        else usage.total
    )
    return result


async def _attempts(
    raw: Measurements,
    cases: list[Case],
    settings: RouteProcessSettings,
    llm: LazyRoutingLlm,
    directory: Path,
) -> None:
    """Rotate arm order across repeats; never share model outputs between arms."""

    async def run_case(case: Case, repeat: int) -> None:
        arms = list(RoutingStrategy)
        rotated = arms[(repeat - 1) % len(arms) :] + arms[: (repeat - 1) % len(arms)]
        for arm in rotated:
            ctx = RoutingRuntime(
                llm=llm,
                settings=settings.router,
                deadline=Deadline(time.monotonic() + settings.timeout_s),
            )
            observation = await observe(
                RouterInput(
                    question=case.question,
                    routing_context=case.routing_context.model_copy(deep=True),
                ),
                ctx,
                arm,
            )
            raw.attempts.append(
                Attempt(case_id=case.id, arm=arm, repeat=repeat, observation=observation)
            )
            logger.info(
                "routing_evaluation_attempt",
                case_id=case.id,
                arm=arm.value,
                repeat=repeat,
                failure_code=observation.failure_code,
            )

    for repeat, offset in product(range(1, raw.repeats + 1), range(0, len(cases), CONCURRENCY)):
        async with asyncio.TaskGroup() as group:
            for case in cases[offset : offset + CONCURRENCY]:
                group.create_task(run_case(case, repeat))
        await asyncio.to_thread(save_partial, raw, directory)


def save_partial(raw: Measurements, directory: Path) -> None:
    """Persist in-progress attempts without promoting them to latest acceptance evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"routing_{raw.run_id}.partial.json").write_text(
        raw.model_dump_json(indent=2), encoding="utf-8"
    )


async def collect(options: Options) -> Measurements:
    """Use only the route process credential projection; no application services."""
    settings = RouteProcessSettings.load()
    config = await asyncio.to_thread(snapshot, settings)
    selection, commit = None, None
    if options.split is Split.FROZEN:
        selection, commit = await asyncio.to_thread(committed_selection, config)
    cases = [c for c in await asyncio.to_thread(load_cases) if c.split is options.split]
    raw = Measurements(
        run_id=uuid4().hex,
        created_at=datetime.now(UTC),
        split=options.split,
        repeats=options.repeats,
        case_ids=[c.id for c in cases],
        config=config,
        attempts=[],
        selection=selection,
        selection_commit=commit,
        complete=False,
    )
    llm = LazyRoutingLlm(settings.llm)
    try:
        await _attempts(raw, cases, settings, llm, options.report)
    finally:
        try:
            await asyncio.to_thread(save_partial, raw, options.report)
        finally:
            await llm.aclose()
    final = await asyncio.to_thread(snapshot, settings)
    raw.complete = final == config and not final.source_dirty
    return raw
