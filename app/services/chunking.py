"""Pure format-aware splitting with stable source and positional provenance."""

from uuid import uuid5

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import ValidationError

from app.core.errors import CorpusValidationError
from app.schemas.corpus import CorpusFormat
from app.schemas.ingestion import (
    LoadedSource,
    PreparedChunk,
    PreparedDocument,
    RegisteredDocument,
    SourcePart,
    canonical,
    digest,
    document_id,
    normalize,
)
from app.services.ingestion_config import IngestionSettings


def recursive(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """Pin otherwise implicit splitter defaults in the versioned contract."""
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,
        strip_whitespace=True,
        is_separator_regex=False,
    )


def table_fragments(text: str, limit: int) -> list[str]:
    """Keep complete rows, repeating both header rows even for an oversized row."""
    header, separator, *rows = text.splitlines()
    prefix = [header, separator]
    result: list[str] = []
    current: list[str] = []
    for row in rows:
        if current and len("\n".join([*prefix, *current, row])) > limit:
            result.append("\n".join([*prefix, *current]))
            current = []
        current.append(row)
    if current or not result:
        result.append("\n".join([*prefix, *current]))
    return result


def contexts(source: LoadedSource, config: IngestionSettings) -> list[SourcePart]:
    """Produce parent sections while preserving sheet names and one-based PDF pages."""
    if source.entry.format is CorpusFormat.EXCEL:
        return source.parts
    result: list[SourcePart] = []
    for part in source.parts:
        if source.entry.format is CorpusFormat.PDF:
            result.extend(
                SourcePart(text=text, page=part.page)
                for text in recursive(config.parent_size, config.parent_overlap).split_text(
                    part.text
                )
            )
        else:
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
            )
            for section in splitter.split_text(part.text):
                headers = [
                    "#" * level + " " + str(section.metadata[f"h{level}"])
                    for level in range(1, 4)
                    if f"h{level}" in section.metadata
                ]
                result.append(
                    SourcePart(
                        text="\n".join(headers) + "\n\n" + section.page_content
                        if headers
                        else section.page_content,
                        heading=" / ".join(
                            str(section.metadata[key])
                            for key in ("h1", "h2", "h3")
                            if key in section.metadata
                        ),
                    )
                )
    return result


def prepare(source: LoadedSource, config: IngestionSettings) -> PreparedDocument:
    """Hash content and parsed metadata, then assign globally ordered child identities."""
    text = normalize(
        "\n\n".join(
            ("## " + part.heading + "\n\n" if part.heading else "") + part.text
            for part in source.parts
        )
    )
    identifier = document_id(source.entry.path)
    version = digest(
        canonical({"content": text, "metadata": source.metadata.model_dump(mode="json")})
    )
    chunking = config.chunking_version()
    chunks: list[PreparedChunk] = []
    try:
        for parent in contexts(source, config):
            children = (
                table_fragments(parent.text, config.table_size)
                if source.entry.format is CorpusFormat.EXCEL
                else recursive(config.child_size, config.child_overlap).split_text(parent.text)
            )
            for content in children:
                identity = canonical(
                    {
                        "document_version": version,
                        "chunking_version": chunking,
                        "heading_path": parent.heading,
                        "ordinal": len(chunks),
                        "content_sha256": digest(content),
                    }
                )
                parent_text = (
                    ("## " + parent.heading + "\n\n" + parent.text)
                    if source.entry.format is CorpusFormat.EXCEL
                    else parent.text
                )
                validate_storage_text(content, parent_text, parent.heading)
                chunks.append(
                    PreparedChunk(
                        document_id=identifier,
                        document_version=version,
                        chunking_version=chunking,
                        chunk_uuid=uuid5(identifier, identity),
                        content_sha256=digest(content),
                        ordinal=len(chunks),
                        heading_path=parent.heading,
                        page=parent.page,
                        char_len=len(content),
                        content=content,
                        parent_content=parent_text,
                    )
                )
        return PreparedDocument(
            document=RegisteredDocument(
                document_id=identifier,
                source_path=normalize(source.entry.path),
                source_fingerprint=source.source_fingerprint,
                content_sha256=digest(text),
                document_version=version,
                chunking_version=chunking,
                metadata=source.metadata,
                chunk_count=len(chunks),
            ),
            chunks=chunks,
        )
    except ValidationError as exc:
        raise CorpusValidationError("Invalid split document.", path=source.entry.path) from exc


def validate_storage_text(content: str, parent: str, heading: str) -> None:
    """Milvus VARCHAR limits count UTF-8 bytes rather than Unicode characters."""
    if any(
        len(value.encode("utf-8")) > limit
        for value, limit in (
            (content, 65535),
            (parent, 65535),
            (heading, 512),
        )
    ):
        raise CorpusValidationError("Chunk exceeds storage byte limit.")
