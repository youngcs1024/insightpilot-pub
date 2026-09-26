"""Deterministic type-specific comparison; stored payloads are never normalized in place."""

import unicodedata
from enum import StrEnum

from app.schemas.memory import (
    FormatPreferenceContent,
    MemoryPayload,
    MetricOverrideContent,
    RegionFocusContent,
    TerminologyContent,
)


class MemoryMatch(StrEnum):
    """Only matching logical keys can duplicate or replace one another."""

    DISTINCT = "distinct"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


def normalize(text: str) -> str:
    """Fold width, case and whitespace without changing the persisted original."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokens(text: str) -> set[str]:
    """Keep individual Han characters and continuous alphanumeric words, including negation."""
    result: set[str] = set()
    word = ""
    for char in normalize(text):
        han = unicodedata.name(char, "").startswith("CJK UNIFIED IDEOGRAPH")
        if char.isalnum() and not han:
            word += char
            continue
        if word:
            result.add(word)
            word = ""
        if han:
            result.add(char)
    if word:
        result.add(word)
    return result


def similar(left: str, right: str) -> bool:
    """Use inclusive Jaccard 0.8; tokenless content only matches normalized exact text."""
    a, b = tokens(left), tokens(right)
    if not a or not b:
        return normalize(left) == normalize(right)
    return 5 * len(a & b) >= 4 * len(a | b)


def compare(candidate: MemoryPayload, existing: MemoryPayload) -> MemoryMatch:
    """Compare content only, without interpreting SQL or extraction confidence."""
    if candidate.memory_type is not existing.memory_type:
        return MemoryMatch.DISTINCT
    left, right = candidate.content, existing.content
    if isinstance(left, MetricOverrideContent) and isinstance(right, MetricOverrideContent):
        if left.metric_key != right.metric_key:
            return MemoryMatch.DISTINCT
        duplicate = left.patch == right.patch
    elif isinstance(left, RegionFocusContent) and isinstance(right, RegionFocusContent):
        duplicate = set(left.region_ids) == set(right.region_ids)
    elif isinstance(left, TerminologyContent) and isinstance(right, TerminologyContent):
        if normalize(left.term) != normalize(right.term):
            return MemoryMatch.DISTINCT
        duplicate = similar(left.means, right.means)
    elif isinstance(left, FormatPreferenceContent) and isinstance(right, FormatPreferenceContent):
        duplicate = left == right
    else:
        return MemoryMatch.DISTINCT
    return MemoryMatch.DUPLICATE if duplicate else MemoryMatch.CONFLICT
