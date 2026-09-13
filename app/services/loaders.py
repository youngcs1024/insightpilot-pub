"""Manifest-driven source snapshots with per-file extraction failures."""

import hashlib
from io import BytesIO
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import openpyxl  # type: ignore[import-untyped]
from openpyxl.utils.exceptions import InvalidFileException  # type: ignore[import-untyped]
from pydantic import Field, ValidationError
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.core.errors import CorpusValidationError
from app.schemas.corpus import CorpusEntry, CorpusFormat, CorpusManifest, CorpusMetadata
from app.schemas.ingestion import (
    LoadedSource,
    SourcePart,
    SourceSnapshot,
    canonical,
    digest,
    normalize,
)
from app.services.corpus_sources import (
    markdown_parts,
    parse_yaml,
    read_utf8,
    source_path,
    table_row,
)


class IngestionInventory(CorpusManifest):
    """An explicitly empty inventory retires the last source; authoring stays nonempty."""

    documents: list[CorpusEntry] = Field(min_length=0, max_length=100)


def discover(root: Path) -> CorpusManifest:
    """Only the inventory owns membership; unlisted files are never implicitly ingested."""
    manifest = parse_yaml(read_utf8(source_path(root, "MANIFEST.yaml")), IngestionInventory)
    paths = [normalize(entry.path) for entry in manifest.documents]
    if len(paths) != len(set(paths)):
        raise CorpusValidationError("Normalized paths collide.")
    return manifest


def read_bytes(root: Path, relative: str, limit: int) -> bytes:
    """Bound a snapshot read and refuse links before touching source bytes."""
    try:
        with source_path(root, relative).open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise CorpusValidationError("Source exceeds byte limit.", path=relative)
        return raw
    except OSError as exc:
        raise CorpusValidationError("Cannot read source.", path=relative) from exc


def snapshot(root: Path, entry: CorpusEntry, limit: int) -> SourceSnapshot:
    """Source and sidecar bytes are retained so extraction cannot race a second read."""
    raw = read_bytes(root, entry.path, limit)
    sidecar = read_bytes(root, entry.metadata_path, limit) if entry.metadata_path else b""
    fingerprint = digest(
        canonical(
            [
                hashlib.sha256(raw).hexdigest(),
                hashlib.sha256(sidecar).hexdigest(),
                entry.format.value,
            ]
        )
    )
    return SourceSnapshot(raw=raw, sidecar=sidecar, fingerprint=fingerprint)


def excel_parts(raw: bytes) -> list[SourcePart]:
    """Retain worksheet boundaries and self-describing tables, without pandas."""
    workbook = openpyxl.load_workbook(BytesIO(raw), read_only=True, data_only=False)
    try:
        parts = []
        for sheet in workbook.worksheets:
            rows = [
                [str(cell) if cell is not None else "" for cell in row]
                for row in sheet.iter_rows(values_only=True)
                if any(cell is not None for cell in row)
            ]
            if rows:
                header, *body = rows
                text = "\n".join(table_row(row) for row in [header, ["---"] * len(header), *body])
                parts.append(SourcePart(text=normalize(text), heading=normalize(sheet.title)))
        return parts
    finally:
        workbook.close()


def pdf_parts(raw: bytes) -> list[SourcePart]:
    """A blank, scanned or encrypted PDF fails as one file rather than vanishing."""
    with BytesIO(raw) as stream:
        reader = PdfReader(stream, strict=True)
        if reader.is_encrypted:
            raise CorpusValidationError("Encrypted PDF.")
        return [
            SourcePart(text=normalize(page.extract_text() or "").strip(), page=index)
            for index, page in enumerate(reader.pages, 1)
        ]


def load_source(entry: CorpusEntry, raw: bytes, sidecar: bytes, fingerprint: str) -> LoadedSource:
    """Extract one snapshot; convert parser failures into a typed per-file result."""
    try:
        if entry.format is CorpusFormat.MARKDOWN:
            metadata, body = markdown_parts(normalize(raw.decode("utf-8")))
            parts = [SourcePart(text=body)]
        else:
            metadata = parse_yaml(normalize(sidecar.decode("utf-8")), CorpusMetadata)
            parts = excel_parts(raw) if entry.format is CorpusFormat.EXCEL else pdf_parts(raw)
        # Business strings are normalized as part of the version contract too.
        metadata = CorpusMetadata.model_validate_json(normalize(metadata.model_dump_json()))
        return LoadedSource(
            entry=entry, metadata=metadata, source_fingerprint=fingerprint, parts=parts
        )
    except (
        OSError,
        UnicodeError,
        BadZipFile,
        PyPdfError,
        InvalidFileException,
        ParseError,
        ValidationError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        raise CorpusValidationError("Cannot extract document.", path=entry.path) from exc
