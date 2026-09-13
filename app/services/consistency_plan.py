"""Reconstruct only committed document versions, using their frozen splitter settings."""

import json
from pathlib import Path
from uuid import UUID

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from app.core.errors import CorpusValidationError, IngestionConfigurationError
from app.schemas.consistency import RegistrySnapshot, RepairBlock
from app.schemas.ingestion import PreparedDocument, canonical
from app.schemas.mcp import Contract
from app.services.chunking import prepare
from app.services.ingestion_config import IngestionSettings
from app.services.loaders import discover, load_source, snapshot


class RepairPlan(Contract):
    """Ephemeral prepared work; unfinished repairs remain detectable in durable stores."""

    prepared: list[PreparedDocument] = Field(default_factory=list)
    blocked: list[RepairBlock] = Field(default_factory=list)


def restore_splitter(raw: str, limits: IngestionSettings) -> IngestionSettings:
    """Accept only configurations reproducible by the currently installed implementation."""
    try:
        payload = TypeAdapter(dict[str, JsonValue]).validate_python(json.loads(raw))
        values = {
            key: payload[key]
            for key in ("child_size", "child_overlap", "parent_size", "parent_overlap", "table_size")
        }
        settings = IngestionSettings.model_validate(
            {**values, "timeout_s": limits.timeout_s, "max_source_bytes": limits.max_source_bytes}
        )
    except (ValidationError, ValueError, KeyError) as exc:
        raise IngestionConfigurationError() from exc
    if settings.splitter_config() != canonical(payload):
        raise IngestionConfigurationError()
    return settings


def stage_repair(
    root: Path, registry: RegistrySnapshot, identifiers: set[UUID], limits: IngestionSettings
) -> RepairPlan:
    """A changed/missing source is blocked without publishing or retiring any version."""
    result = RepairPlan()
    if not identifiers:
        return result
    try:
        entries = {entry.path: entry for entry in discover(root).documents}
    except CorpusValidationError as exc:
        result.blocked = [
            RepairBlock(document_id=identifier, code=exc.code)
            for identifier in sorted(identifiers)
        ]
        return result
    manifest = registry.manifest
    if manifest is None:
        raise IngestionConfigurationError()
    for document in registry.documents:
        if document.document_id not in identifiers:
            continue
        try:
            config = restore_splitter(
                manifest.splitter_configs[document.chunking_version], limits
            )
            entry = entries.get(document.source_path)
            if entry is None:
                raise CorpusValidationError("Committed source is not in the inventory.")
            source = snapshot(root, entry, limits.max_source_bytes)
            prepared = prepare(
                load_source(entry, source.raw, source.sidecar, source.fingerprint), config
            )
            if (
                prepared.document.document_id != document.document_id
                or prepared.document.document_version != document.document_version
                or prepared.document.chunking_version != document.chunking_version
            ):
                result.blocked.append(
                    RepairBlock(document_id=document.document_id, code="SOURCE_VERSION_CHANGED")
                )
            else:
                result.prepared.append(prepared)
        except (CorpusValidationError, IngestionConfigurationError) as exc:
            result.blocked.append(RepairBlock(document_id=document.document_id, code=exc.code))
    return result
