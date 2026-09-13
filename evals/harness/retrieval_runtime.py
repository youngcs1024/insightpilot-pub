"""Explicit live lifecycle; no GPU/SSH operations occur on module import."""

import asyncio
from itertools import product
from pathlib import Path

from app.clients.model_runtime import ModelRuntimeClient
from app.db.session import Database
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.search_store import HybridSearchStore
from app.schemas.ingestion import canonical, digest
from app.services.corpus_sources import parse_yaml
from evals.harness.ablation import BASE_ARMS
from evals.harness.contracts import EvaluationError
from evals.harness.retrieval import observe, replay_fp32
from evals.harness.retrieval_contracts import Arm, Dataset, Measurements, Selection, arm_config
from evals.harness.retrieval_dataset import check_manifest
from scripts.ci_changes import git_bytes
from scripts.dev_retrieve import RetrievalProcessSettings, close_resources
from scripts.model_evidence import Provenance


def source_identity() -> tuple[str, bool]:
    """Capture tracked source and newly authored untracked public resources."""
    return git_bytes(["rev-parse", "HEAD"]).decode().strip(), bool(
        git_bytes(["status", "--porcelain", "--untracked-files=normal"])
    )


def committed_selection(path: Path) -> tuple[Selection, str]:
    """Check the committed selection before opening any frozen judgments."""
    root = Path(git_bytes(["rev-parse", "--show-toplevel"]).decode().strip())
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise EvaluationError("Selection must be a committed repository resource")
    relative = resolved.relative_to(root).as_posix()
    content = path.read_text(encoding="utf-8")
    if git_bytes(["show", f"HEAD:{relative}"]).decode() != content:
        raise EvaluationError("Selection has uncommitted changes")
    selection = parse_yaml(content, Selection)
    if selection.arm is Arm.D_FP32 or selection.config != arm_config(
        selection.arm, selection.config.filtering
    ):
        raise EvaluationError("Invalid selected configuration")
    commit = git_bytes(["log", "-1", "--format=%H", "--", relative]).decode().strip()
    git_bytes(["merge-base", "--is-ancestor", selection.development_sha, commit])
    return selection, commit


async def collect(
    dataset: Dataset,
    server: Provenance,
    selection: Selection | None = None,
    selection_commit: str | None = None,
) -> Measurements:
    """Execute all eight arms in one process with a common active corpus manifest."""
    settings = RetrievalProcessSettings.load()
    if settings.model_runtime is None or settings.model_runtime.precision != "fp16":
        raise EvaluationError("FP16 model client configuration is required")
    database = Database(settings.database)
    database.start()
    model = ModelRuntimeClient(settings.model_runtime)
    sha, dirty = await asyncio.to_thread(source_identity)
    result = Measurements(
        client_sha=sha,
        source_dirty=dirty,
        dataset_identity=dataset.identity,
        corpus_version=dataset.manifest.corpus_version,
        split=dataset.cases[0].split,
        server=server,
        attempts=[],
        selection_commit=selection_commit,
        selection_identity=digest(canonical(selection.model_dump(mode="json")))
        if selection
        else None,
    )
    try:
        async with HybridSearchStore(settings.retrieval.milvus) as store:
            await validate_active(dataset, database)
            for case, arm in product(dataset.cases, BASE_ARMS):
                effective = settings.retrieval.model_copy(deep=True)
                effective.search = (
                    selection.config.model_copy(deep=True)
                    if selection is not None and arm is selection.arm
                    else arm_config(arm)
                )
                pipeline = RetrievalPipeline(database, store, model, effective)
                result.attempts.append(await observe(case, pipeline, arm))
            await validate_active(dataset, database)
    finally:
        await close_resources(database, model)
    final_sha, final_dirty = await asyncio.to_thread(source_identity)
    result.source_dirty = result.source_dirty or final_dirty or final_sha != result.client_sha
    return result


async def validate_active(dataset: Dataset, database: Database) -> None:
    """Read registry identity before any expensive inference or retrieval."""
    async with database.session() as session, session.begin():
        active = await DocumentRepository(session).manifest()
        chunks = await ChunkRepository(session).list_all()
    check_manifest(dataset, active, {item.chunk_uuid for item in chunks})


async def control(raw: Measurements, server: Provenance) -> Measurements:
    """Operator switches only the authorized service, then invokes this replay phase."""
    sha, dirty = await asyncio.to_thread(source_identity)
    if raw.server != server or raw.client_sha != sha or raw.source_dirty or dirty:
        raise EvaluationError("Precision replay must use identical clean source and deployment")
    if any(item.arm is Arm.D_FP32 for item in raw.attempts):
        raise EvaluationError("Precision control is already present")
    settings = RetrievalProcessSettings.load()
    if settings.model_runtime is None:
        raise EvaluationError("Model client configuration is required")
    configured = settings.model_runtime.model_copy(update={"precision": "fp32"})
    model = ModelRuntimeClient(configured)
    result = raw.model_copy(deep=True)
    try:
        for original in raw.attempts:
            if original.arm is Arm.D_RERANK:
                result.attempts.append(await replay_fp32(original, model))
    finally:
        async with asyncio.timeout(10):
            await model.aclose()
    final_sha, final_dirty = await asyncio.to_thread(source_identity)
    result.source_dirty = result.source_dirty or final_dirty or final_sha != result.client_sha
    return result
