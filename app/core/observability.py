"""Application-owned, best-effort Langfuse tracing with metadata-only callbacks."""

from __future__ import annotations

import asyncio
import base64
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, LangfuseGeneration
from langgraph.config import get_config
from langgraph.types import Command
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel, Field

from app.core.config_models import SchemaStrategy  # noqa: TC001 -- Pydantic resolves at runtime.
from app.core.masking import mask, mask_otel_spans
from app.core.trace_export import SafeSpanExporter
from app.schemas.sanity import SanityFlag  # noqa: TC001 -- Pydantic resolves at runtime.
from app.schemas.sql_correction import CorrectionStatus

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from langfuse import LangfuseSpan

    from app.core.config_models import Settings

logger = structlog.get_logger(__name__)
_current: ContextVar[Observation | None] = ContextVar("ip_observation", default=None)
_role: ContextVar[str | None] = ContextVar("ip_llm_role", default=None)
_NODES = frozenset(
    {
        "prepare",
        "rewrite_question",
        "answer_data",
        "answer_data_routed",
        "answer_knowledge",
        "persist_evidence",
        "format_answer",
        "select_schema",
        "sanity_check",
        "correct_sql",
        "resolve_metrics",
        "generate_sql",
        "validate_sql",
        "execute_sql",
        "package_evidence",
        "package_failure",
        "rewrite_query",
        "resolve_time_scope",
        "retrieve",
        "no_evidence",
        "finish_knowledge",
    }
)


class TraceMetadata(BaseModel):
    """Only explicitly reviewed diagnostic fields cross the tracing boundary."""

    referenced_prior_turn: bool | None = None
    unresolved_reference_count: int | None = Field(default=None, ge=0, le=20)
    schema_strategy: SchemaStrategy | None = None
    sanity_flags: list[SanityFlag] = Field(default_factory=list, max_length=8)
    sanity_check_failed: bool | None = None
    schema_tables: int | None = Field(default=None, ge=0)
    schema_tokens: int | None = Field(default=None, ge=0)
    schema_tokenizer: Literal["cl100k_base"] | None = None
    schema_utf8_bytes: int | None = Field(default=None, ge=0)
    user_id: str | None = None
    conversation_id: str | None = None
    turn_id: str | None = None
    request_id: str | None = None
    route: Literal["DATA_ONLY", "data_only", "knowledge_only", "both", "clarify"] | None = None
    original_route: Literal["data_only", "knowledge_only", "both", "clarify"] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    decided_by: Literal["prefilter", "llm"] | None = None
    prefilter_hit: bool | None = None
    router_tokens: int | None = Field(default=None, ge=0)
    projection_specialist: Literal["data", "knowledge"] | None = None
    projection_tokens: int | None = Field(default=None, ge=0)
    projection_tokenizer: Literal["cl100k_base"] | None = None
    status: str | None = None
    degraded_components: list[str] = Field(default_factory=list)
    role: str | None = None
    model: str | None = None
    attempt: int | None = None
    structured_tier: int | None = None
    fallback_index: int | None = None
    execution_ms: int | None = None
    mcp_call_id: str | None = None
    tool: str | None = None
    row_count: int | None = None
    code: str | None = None
    cost_status: Literal["inferred_by_langfuse_or_unknown"] | None = None


@dataclass
class Observation:
    """A safe handle; telemetry operations cannot change business control flow."""

    owner: Observability
    span: LangfuseSpan | LangfuseGeneration | None = None
    metadata: TraceMetadata = field(default_factory=TraceMetadata)
    nodes: dict[str, Observation] = field(default_factory=dict)

    def update(self, metadata: TraceMetadata) -> None:
        """Merge reviewed fields, never overwrite identity with null defaults."""
        self.metadata = self.metadata.model_copy(update=metadata.model_dump(exclude_unset=True))
        if self.span is not None:
            try:
                self.span.update(metadata=self.metadata.model_dump(exclude_none=True))
            except Exception:
                logger.exception("trace_update_failed")

    def usage(self, prompt: int, completion: int) -> None:
        """Record provider-reported usage without inventing prices or counts."""
        if isinstance(self.span, LangfuseGeneration):
            try:
                self.span.update(usage_details={"input": prompt, "output": completion})
            except Exception:
                logger.exception("trace_usage_failed")

    def evidence(self, data: BaseModel) -> None:
        """Sanitize before SDK serialization or media processing can occur."""
        if self.span is not None:
            try:
                self.span.update(output=mask(data))
            except Exception:
                logger.exception("trace_evidence_failed")

    def end(self) -> None:
        """Ending a span is best effort, including on cancellation."""
        if self.span is not None:
            try:
                self.span.end()
            except Exception:
                logger.exception("trace_end_failed")


class Observability:
    """One client per application; disabled tracing does no network work."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Langfuse | None = None

    async def start(self) -> None:
        """Initialize off the event loop, with no synchronous auth probe."""
        if self.client is not None or not self.settings.observability.langfuse_enabled:
            return
        try:
            self.client = await asyncio.to_thread(self._create_client)
        except Exception:
            logger.exception("tracing_unavailable")

    def _create_client(self) -> Langfuse:
        config = self.settings.observability
        public_key = (
            config.langfuse_public_key.get_secret_value() if config.langfuse_public_key else ""
        )
        secret_key = (
            config.langfuse_secret_key.get_secret_value() if config.langfuse_secret_key else ""
        )
        authorization = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        exporter = SafeSpanExporter(
            OTLPSpanExporter(
                endpoint=str(config.langfuse_base_url).rstrip("/") + "/api/public/otel/v1/traces",
                headers={
                    "Authorization": "Basic " + authorization,
                    "x-langfuse-sdk-name": "python",
                    "x-langfuse-sdk-version": "4.15.1",
                    "x-langfuse-public-key": public_key,
                    "x-langfuse-ingestion-version": "4",
                },
                timeout=5,
            )
        )
        try:
            return Langfuse(
                public_key=config.langfuse_public_key.get_secret_value()
                if config.langfuse_public_key
                else None,
                secret_key=config.langfuse_secret_key.get_secret_value()
                if config.langfuse_secret_key
                else None,
                base_url=str(config.langfuse_base_url),
                environment=self.settings.environment.value,
                timeout=5,
                flush_at=32,
                flush_interval=1.0,
                tracing_enabled=True,
                sample_rate=1.0,
                tracer_provider=TracerProvider(
                    resource=Resource({"service.name": "insightpilot"}),
                    shutdown_on_exit=False,
                ),
                span_exporter=exporter,
                mask=lambda *, data, **kwargs: mask(data),
                mask_otel_spans=mask_otel_spans,
                should_export_span=lambda span: (
                    span.instrumentation_scope is not None
                    and span.instrumentation_scope.name == "langfuse-sdk"
                ),
            )
        except Exception:
            exporter.shutdown()
            raise

    async def aclose(self) -> None:
        """Flush only at shutdown, outside the event loop and within its budget."""
        client, self.client = self.client, None
        if client is None:
            return
        try:
            async with asyncio.timeout(self.settings.http.shutdown_timeout_s):
                await asyncio.to_thread(client.shutdown)
        except Exception:
            logger.exception("trace_shutdown_failed")

    def begin(
        self,
        name: str,
        metadata: TraceMetadata,
        *,
        generation: bool = False,
        trace_id: str | None = None,
    ) -> Observation:
        """Start from the explicit request-local parent; never export input text."""
        result = Observation(self, metadata=metadata)
        if self.client is None:
            return result
        try:
            parent = _parent()
            if trace_id is not None:
                result.span = self.client.start_observation(
                    name=name,
                    trace_context={"trace_id": trace_id},
                    metadata=metadata.model_dump(exclude_none=True),
                )
            elif parent is not None and parent.span is not None:
                if generation:
                    result.span = parent.span.start_observation(
                        name=name,
                        as_type="generation",
                        model=metadata.model,
                        metadata=metadata.model_dump(exclude_none=True),
                    )
                else:
                    result.span = parent.span.start_observation(
                        name=name,
                        as_type="span",
                        metadata=metadata.model_dump(exclude_none=True),
                    )
        except Exception:
            logger.exception("trace_start_failed")
        return result

    @contextmanager
    def turn(self, trace_id: str, metadata: TraceMetadata) -> Iterator[Observation]:
        """Keep the root alive through answer commit and cancellation cleanup."""
        observation = self.begin("turn", metadata, trace_id=trace_id)
        token = _current.set(observation)
        try:
            yield observation
        finally:
            _current.reset(token)
            observation.end()


@contextmanager
def observe(
    name: str, metadata: TraceMetadata, *, generation: bool = False
) -> Iterator[Observation | None]:
    """Services inherit the executing node, without global client lookup."""
    parent = _parent()
    if parent is None:
        yield None
        return
    observation = parent.owner.begin(name, metadata, generation=generation)
    token = _current.set(observation)
    try:
        yield observation
    except BaseException:
        observation.update(TraceMetadata(status="failed"))
        raise
    else:
        observation.update(TraceMetadata(status="succeeded"))
    finally:
        _current.reset(token)
        observation.end()


@contextmanager
def model_role(role: str) -> Iterator[None]:
    """Associate each retry/repair HTTP attempt with its logical model role."""
    token = _role.set(role)
    try:
        yield
    finally:
        _role.reset(token)


def current_role() -> str | None:
    """Return only the typed application role name."""
    return _role.get()


def record_evidence(data: BaseModel) -> None:
    """Attach a sanitized snapshot to the active persistence node."""
    observation = _parent()
    if observation is not None:
        observation.evidence(data)


class GraphTraceCallback(BaseCallbackHandler):
    """Per-invocation node spans; raw callback arguments never enter the SDK."""

    run_inline = True

    def __init__(self) -> None:
        self._runs: dict[UUID, Observation] = {}
        self._root = _current.get()

    def on_chain_start(
        self,
        serialized: dict[str, object] | None,
        inputs: object,
        *,
        run_id: UUID,
        **kwargs: object,
    ) -> None:
        """LangChain's untyped callback boundary is projected to fixed node names."""
        name = kwargs.get("name")
        parent = self._root
        if not isinstance(name, str) or name not in _NODES or parent is None:
            return
        observation = parent.owner.begin(name, TraceMetadata())
        parent.nodes[name] = observation
        self._runs[run_id] = observation

    def on_chain_end(self, outputs: object, *, run_id: UUID, **kwargs: object) -> None:
        """Close the node without serializing arbitrary graph state."""
        failed = (
            isinstance(outputs, Command)
            and isinstance(outputs.update, dict)
            and (
                outputs.update.get("status") == "failed"
                or outputs.update.get("correction_status") is CorrectionStatus.TERMINAL
                or bool(outputs.update.get("failures"))
                or outputs.update.get("failure") is not None
                or outputs.update.get("operation_failure") is not None
            )
        )
        abstained = (
            isinstance(outputs, Command)
            and isinstance(outputs.update, dict)
            and (
                outputs.update.get("abstained") is True
                or outputs.update.get("knowledge_abstention_reason") is not None
            )
        )
        self._finish(run_id, failed=failed, abstained=abstained)

    def close(self) -> None:
        """Close any unfinished node spans after graph cancellation or callback failure."""
        for run_id in tuple(self._runs):
            self._finish(run_id, failed=True)
        if self._root is not None:
            self._root.nodes.clear()

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: object) -> None:
        """Exception prose and traceback never reach the tracing SDK."""
        self._finish(run_id, failed=True)

    def _finish(self, run_id: UUID, *, failed: bool = False, abstained: bool = False) -> None:
        entry = self._runs.pop(run_id, None)
        if entry is None:
            return
        status = "failed" if failed else "abstained" if abstained else "succeeded"
        entry.update(TraceMetadata(status=status))
        entry.end()


def _parent() -> Observation | None:
    parent = _current.get()
    if parent is None or not parent.nodes:
        return parent
    try:
        name = get_config().get("metadata", {}).get("langgraph_node")
    except RuntimeError:
        return parent
    return parent.nodes.get(name, parent) if isinstance(name, str) else parent


def mark_degraded(component: str) -> None:
    """Keep a turn-level record of recovered model fallback."""
    root = _current.get()
    if root is not None and component not in root.metadata.degraded_components:
        root.update(
            TraceMetadata(degraded_components=[*root.metadata.degraded_components, component])
        )


def record_turn_status(status: str) -> None:
    """Reuse the cleanup transaction's result, without another observability DB read."""
    root = _current.get()
    if root is not None:
        root.update(TraceMetadata(status=status))


def update_current_observation(metadata: TraceMetadata) -> None:
    """Update the active node with reviewed metadata only."""
    observation = _parent()
    if observation is not None:
        observation.update(metadata)
