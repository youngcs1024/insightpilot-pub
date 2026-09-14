"""Ephemeral dependency injection, excluded from checkpoint state and metadata."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from app.agents.contracts import DataEvidence, EvidenceSnapshot, PreparedContext, TurnIdentity
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.errors import PeriodUnresolved
from app.core.llm_config import ModelRole
from app.schemas.mcp import QueryArguments, QueryResultPayload
from app.schemas.metric_resolution import RegionReference, RegionScope
from app.schemas.metrics import MetricDefinition
from app.schemas.retrieval import RetrievalQuery, RetrievalResult
from app.schemas.schema_catalog import SchemaCatalog


class LlmPort(Protocol):
    """Existing LlmService's structured output boundary."""

    async def generate_structured[T: BaseModel](
        self, role: ModelRole, messages: list[BaseMessage], schema: type[T], *, deadline: Deadline
    ) -> T: ...


class RetrievalPort(Protocol):
    """One bounded retrieval service owns all storage and remote model operations."""

    async def retrieve(self, query: RetrievalQuery, *, deadline: Deadline) -> RetrievalResult: ...


class McpPort(Protocol):
    """Only the authenticated read-only tool is exposed to the graph."""

    async def call_tool(
        self, name: Literal["execute_readonly_query"], args: QueryArguments, *, deadline: Deadline
    ) -> QueryResultPayload: ...


class EvidencePort(Protocol):
    """Snapshots cross the boundary as versioned models."""

    async def find(
        self, identity: TurnIdentity, snapshot_id: UUID | None = None
    ) -> EvidenceSnapshot | None: ...

    async def commit(self, identity: TurnIdentity, data: DataEvidence) -> EvidenceSnapshot: ...


class ConversationPort(Protocol):
    """Owned application history is the source of truth."""

    async def prepare(self, identity: TurnIdentity) -> PreparedContext: ...


class SchemaCatalogPort(Protocol):
    """Render through the application catalog without exposing storage."""

    async def render(
        self, tables: list[str] | None = None, *, deadline: Deadline | None = None
    ) -> str: ...

    async def snapshot(self, *, deadline: Deadline | None = None) -> SchemaCatalog: ...


class MetricPort(Protocol):
    """Published catalog reads retain the request deadline and service retry policy."""

    async def list_active(self, *, deadline: Deadline | None = None) -> list[MetricDefinition]: ...

    async def get_active(
        self, key: str, *, deadline: Deadline | None = None
    ) -> MetricDefinition: ...


class RegionPort(Protocol):
    """Resolve explicit names through the business trust boundary."""

    async def resolve(
        self, reference: RegionReference, *, deadline: Deadline
    ) -> RegionScope | None: ...


class SchemaTokenPort(Protocol):
    """A preinitialized named tokenizer performs no node-time resource loading."""

    @property
    def name(self) -> Literal["cl100k_base"]: ...

    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class RuntimeContext:
    """Passed through LangGraph's native runtime context, never configurable metadata."""

    llm: LlmPort
    mcp: McpPort
    evidence: EvidencePort
    conversations: ConversationPort
    settings: Settings
    deadline: Deadline
    identity: TurnIdentity
    trace_id: str
    schema_catalog: SchemaCatalogPort
    schema_token_counter: SchemaTokenPort
    regions: RegionPort
    metrics: MetricPort
    now: datetime
    retrieval: RetrievalPort | None = None

    def __post_init__(self) -> None:
        """Reject a host-local or otherwise ambiguous reference instant."""
        if self.now.tzinfo is None or self.now.utcoffset() is None:
            raise PeriodUnresolved("Runtime requires an aware request instant.")
