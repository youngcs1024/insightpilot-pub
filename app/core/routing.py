"""Closed routing strategies shared by production and its evaluation."""

from enum import StrEnum


class RoutingStrategy(StrEnum):
    """Rules-only is an evaluation control, never a production default."""

    PREFILTER_ONLY = "prefilter_only"
    LLM_ONLY = "llm_only"
    HYBRID = "hybrid"
