"""Runtime-injected packaging entry point, independently usable before graph assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.retrieval.evidence import package_evidence as package

if TYPE_CHECKING:
    from app.agents.runtime import RuntimeContext
    from app.schemas.knowledge import KnowledgeEvidence
    from app.schemas.retrieval import RetrievalResult


def package_evidence(result: RetrievalResult, context: RuntimeContext) -> KnowledgeEvidence:
    """Read only supplied state, settings and the already-initialized token counter."""
    context.deadline.check("package_knowledge_evidence")
    return package(result, context.settings.retrieval.evidence, context.schema_token_counter)
