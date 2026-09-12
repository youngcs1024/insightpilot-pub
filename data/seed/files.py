"""Canonical JSONL export and validated import; existing files are never replaced."""

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from data.seed.contracts import (
    TABLE_NAMES,
    Dataset,
    FileManifest,
    Manifest,
    SeedConflictError,
    SeedError,
)
from data.seed.rows import ROW_MODELS, SeedRow
from data.seed.validation import validate


def encode_rows(rows: list[SeedRow]) -> bytes:
    """Serialize explicitly declared row fields in stable column order."""
    if not rows:
        return b""
    primary_key = next(iter(type(rows[0]).model_fields))
    ordered = sorted(rows, key=lambda row: getattr(row, primary_key))
    return ("".join(row.model_dump_json() + "\n" for row in ordered)).encode("utf-8")


def export(dataset: Dataset, directory: Path) -> Manifest:
    """Publish a manifest last; matching complete exports are reusable."""
    validate(dataset)
    payloads = [encode_rows(rows) for rows in dataset.tables]
    manifest = Manifest(
        parameters=dataset.parameters,
        files=[
            FileManifest(table=name, rows=len(rows), sha256=hashlib.sha256(payload).hexdigest())
            for name, rows, payload in zip(TABLE_NAMES, dataset.tables, payloads, strict=True)
        ],
    )
    if directory.exists() and any(directory.iterdir()):
        existing, _ = read(directory)
        if existing != manifest:
            raise SeedConflictError("Output differs; select a new output directory.")
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in zip(TABLE_NAMES, payloads, strict=True):
        with (directory / f"{name}.jsonl").open("xb") as target:
            target.write(payload)
    with (directory / "manifest.json").open("x", encoding="utf-8") as target:
        target.write(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def read(directory: Path) -> tuple[Manifest, Dataset]:
    """Validate manifest, file hashes, counts and typed rows before opening a database."""
    try:
        manifest = Manifest.model_validate_json((directory / "manifest.json").read_bytes())
        if manifest.format_version != 1 or manifest.design_version != "2026-09-07-step2.1":
            raise SeedError("Unsupported seed format or design version.")
        if tuple(f.table for f in manifest.files) != TABLE_NAMES:
            raise SeedError("Manifest must contain exactly the eight ordered tables.")
        tables = []
        for entry, model in zip(manifest.files, ROW_MODELS, strict=True):
            payload = (directory / f"{entry.table}.jsonl").read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry.sha256:
                raise SeedError("Seed file checksum mismatch.")
            rows = [model.model_validate_json(line) for line in payload.splitlines()]
            if len(rows) != entry.rows:
                raise SeedError("Seed row count mismatch.")
            tables.append(rows)
        dataset = Dataset(parameters=manifest.parameters, tables=tables)
        validate(dataset)
        return manifest, dataset
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        raise SeedError("Cannot read a complete typed seed export.") from exc
