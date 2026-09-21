"""Final OTLP boundary also strips scope credentials, events and resource baggage."""

from collections.abc import Sequence

import structlog
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Status

from app.core.masking import safe_attributes

logger = structlog.get_logger(__name__)
_NAMES = frozenset(
    {
        "turn",
        "prepare",
        "rewrite_question",
        "router",
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
        "retrieval_encode",
        "retrieval_admission",
        "retrieval_arm",
        "retrieval_fusion",
        "retrieval_rerank",
        "retrieval_filter",
        "answer_data",
        "answer_data_routed",
        "answer_knowledge",
        "specialist_projection",
        "persist_evidence",
        "format_answer",
        "llm_completion",
        "mcp_execute",
    }
)


class SafeSpanExporter(SpanExporter):
    """Sanitize copies after SDK project routing, without changing the live spans."""

    def __init__(self, exporter: SpanExporter) -> None:
        self.exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Drop the batch if projection fails; credentials remain only in HTTP auth."""
        try:
            safe = [
                ReadableSpan(
                    name=span.name if span.name in _NAMES else "observation",
                    context=span.context,
                    parent=span.parent,
                    kind=span.kind,
                    start_time=span.start_time,
                    end_time=span.end_time,
                    attributes=safe_attributes(span.attributes or {}),
                    resource=Resource({"service.name": "insightpilot"}),
                    instrumentation_scope=InstrumentationScope(
                        name="langfuse-sdk", version="4.15.1"
                    ),
                    status=Status(span.status.status_code),
                    events=(),
                    links=(),
                )
                for span in spans
            ]
            return self.exporter.export(safe)
        except Exception:
            logger.exception("trace_export_failed")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Delegate the application-owned bounded shutdown."""
        self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Only lifecycle/acceptance code flushes; requests never wait for export."""
        return self.exporter.force_flush(timeout_millis)
