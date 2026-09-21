"""Typed published-corpus fixtures for clarification capability tests."""

from datetime import date
from uuid import uuid4

from app.schemas.corpus import CorpusMetadata, DocumentType
from app.schemas.ingestion import (
    ActiveManifest,
    RegisteredDocument,
    VersionMember,
    canonical,
    digest,
)


def published_document(category: DocumentType = DocumentType.POLICY) -> RegisteredDocument:
    """Build a registry member without any source, model or vector operation."""
    return RegisteredDocument(
        document_id=uuid4(),
        document_version="a" * 64,
        chunking_version=digest("{}"),
        source_path="rule.md",
        source_fingerprint="b" * 64,
        content_sha256="c" * 64,
        metadata=CorpusMetadata(
            title="测试规则",
            doc_type=category,
            effective_from=date(2026, 1, 1),
            effective_to=None,
            supersedes=None,
        ),
        chunk_count=1,
    )


def published_manifest(document: RegisteredDocument) -> ActiveManifest:
    """Bind the manifest to the exact document and chunking revisions."""
    member = VersionMember(
        document_id=document.document_id,
        document_version=document.document_version,
        chunking_version=document.chunking_version,
    )
    return ActiveManifest(
        corpus_version=digest(
            canonical([(str(member.document_id), member.document_version, member.chunking_version)])
        ),
        members=[member],
        splitter_configs={document.chunking_version: "{}"},
        collection="test",
    )
