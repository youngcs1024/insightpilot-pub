"""Compatibility imports for the data specialist's result summarization API."""

from app.agents.data.summarize import (
    COLUMN_CAP,
    SAMPLE_CAP,
    column_stats,
    package_result,
    render_block,
    summarize_result,
)

__all__ = [
    "COLUMN_CAP",
    "SAMPLE_CAP",
    "column_stats",
    "package_result",
    "render_block",
    "summarize_result",
]
