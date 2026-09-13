"""Each rule is isolated from unrelated filters; exact boundary scores are intentional."""
# ruff: noqa: PLR2004 -- fixed relevance scores and counts are acceptance expectations.

import pytest
from pydantic import ValidationError

from app.core.errors import IngestionRegistryError, RetrievalConfigurationError
from app.retrieval.config import FilterConfig
from app.retrieval.filtering import canonical_source, fallback, filter_candidates, filter_ranked
from tests.rerank_support import scored


def test_empty_input_returns_empty() -> None:
    assert filter_candidates([], FilterConfig()) == []


def test_all_below_absolute_floor_returns_empty() -> None:
    result = filter_ranked(scored([0.1, 0.2]), FilterConfig())
    assert result.candidates == []
    assert result.top_rerank_score == 0.2
    assert result.meets_floor is False
    assert result.reranked


def test_relative_threshold_applies_when_above_floor() -> None:
    output = filter_candidates(scored([0.9, 0.5, 0.4]), FilterConfig(gap_threshold=1))
    assert [item.scores.rerank for item in output] == [0.9, 0.5]


def test_absolute_floor_applies_when_relative_is_lower() -> None:
    output = filter_candidates(scored([0.4, 0.3, 0.29]), FilterConfig(gap_threshold=1))
    assert [item.scores.rerank for item in output] == [0.4, 0.3]


def test_per_document_cap_enforced() -> None:
    values = scored([0.9] * 6, ["a.md"] * 6)
    assert len(filter_candidates(values, FilterConfig(gap_threshold=1))) == 3


def test_score_cliff_truncates_before_k() -> None:
    output = filter_candidates(scored([0.9, 0.88, 0.85, 0.6, 0.55]), FilterConfig())
    assert len(output) == 3


def test_smooth_scores_take_full_k() -> None:
    values = scored([0.9 - index * 0.02 for index in range(10)])
    assert filter_candidates(values, FilterConfig()) == values[:8]


def test_diversity_keeps_two_documents_when_both_relevant() -> None:
    values = scored([0.9] * 5 + [0.88], ["a.md"] * 5 + ["b.md"])
    result = filter_candidates(values, FilterConfig())
    assert [item.source_path for item in result] == ["a.md"] * 3 + ["b.md"]


def test_high_count_and_minimum_determine_cap() -> None:
    values = scored([0.9, 0.5, 0.49], ["a.md"] * 3)
    config = FilterConfig(gap_threshold=1, min_per_doc=2)
    assert filter_candidates(values, config) == values[:2]


def test_equal_gap_and_relative_threshold_survive() -> None:
    values = scored([1, 0.75, 0.5])
    assert filter_candidates(values, FilterConfig(gap_threshold=0.25)) == values


def test_equal_high_ratio_counts_toward_source_cap() -> None:
    values = scored([1, 0.5, 0.5], ["a.md"] * 3)
    assert filter_candidates(values, FilterConfig(high_ratio=0.5, gap_threshold=1)) == values


def test_stable_ties_do_not_regroup_sources_or_mutate_inputs() -> None:
    values = scored([0.8] * 4, ["a.md", "b.md", "a.md", "b.md"])
    before = [item.model_dump() for item in values]
    assert filter_candidates(values, FilterConfig()) == values
    assert [item.model_dump() for item in values] == before


def test_unsorted_input_uses_global_top_and_final_k_one() -> None:
    values = scored([0.4, 0.9, 0.5])
    assert filter_candidates(values, FilterConfig(final_k=1)) == [values[1]]


@pytest.mark.parametrize("suffix", ["_副本", " (副本)", "_copy", " (copy)", " (2)", "_COPY (3)"])
def test_copy_suffixes_share_cap(suffix: str) -> None:
    values = scored([0.8] * 4, ["policies/Rule.md"] * 3 + [f"policies/Rule{suffix}.md"])
    assert len(filter_candidates(values, FilterConfig())) == 3


@pytest.mark.parametrize("path", ["other/rule.md", "policies/rule_v2.md", "policies/rule2026.md"])
def test_directories_and_formal_versions_remain_distinct(path: str) -> None:
    assert canonical_source(path) != canonical_source("policies/rule.md")


def test_unscored_fallback_preserves_order_and_unknown_floor() -> None:
    values = scored([0.9] * 10, ["a.md"] * 5 + ["b.md"] * 5)
    for item in values:
        item.scores.rerank = None
    output = fallback(values, FilterConfig(final_k=4))
    assert output.candidates == [*values[:3], values[5]]
    assert output.meets_floor is None
    assert output.top_rerank_score is None
    assert not output.reranked


def test_missing_source_fails_instead_of_collapsing_unknowns() -> None:
    values = scored([0.8])
    values[0].source_path = None
    with pytest.raises(IngestionRegistryError):
        filter_candidates(values, FilterConfig())


def test_missing_rerank_score_is_not_fusion_relevance() -> None:
    values = scored([0.8])
    values[0].scores.rerank = None
    with pytest.raises(RetrievalConfigurationError):
        filter_candidates(values, FilterConfig())


@pytest.mark.parametrize("config", [
    {"dynamic_ratio": -0.1}, {"absolute_floor": 1.1}, {"high_ratio": float("nan")},
    {"gap_threshold": 1.1}, {"min_per_doc": 0}, {"max_per_doc": 101},
    {"min_per_doc": 4, "max_per_doc": 3}, {"final_k": 0},
])
def test_invalid_filter_config_rejected(config: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        FilterConfig.model_validate(config)
