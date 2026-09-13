"""Step 3.3 corpus acceptance, including the real migrated PostgreSQL metric catalog."""

# ruff: noqa: PLR2004 -- fixed corpus counts, dates and acceptance bounds.

from collections import Counter
from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.metric import MetricRepository
from app.schemas.corpus import CorpusDocument, CorpusFormat, DocumentType
from data.corpus_io import corpus_statistics, load_corpus
from data.corpus_metrics import validate_metric_memos

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def corpus() -> list[CorpusDocument]:
    return load_corpus()


def test_manifest_matches_files(corpus: list[CorpusDocument]) -> None:
    assert len(corpus) == 60
    assert Counter(document.metadata.doc_type for document in corpus) == {
        DocumentType.POLICY: 12,
        DocumentType.PROMO_RULE: 15,
        DocumentType.METRIC_MEMO: 8,
        DocumentType.SOP: 10,
        DocumentType.REGION_RULE: 6,
        DocumentType.ANALYSIS_NOTE: 9,
    }
    assert Counter(document.entry.format for document in corpus) == {
        CorpusFormat.MARKDOWN: 54,
        CorpusFormat.EXCEL: 4,
        CorpusFormat.PDF: 2,
    }


def test_every_doc_has_front_matter(corpus: list[CorpusDocument]) -> None:
    # Binary metadata is the explicitly approved same-name YAML sidecar equivalent.
    for document in corpus:
        assert document.metadata.title.strip()
        assert document.text.strip()
        assert document.metadata.effective_from >= date(2025, 3, 1)
        expected = (
            None
            if document.entry.format is CorpusFormat.MARKDOWN
            else document.entry.path + ".meta.yaml"
        )
        assert document.entry.metadata_path == expected


async def test_metric_memos_agree_with_catalog(
    corpus: list[CorpusDocument], db_session: AsyncSession
) -> None:
    # db_session owns a transaction against actual app migrations, not a YAML substitute.
    definitions = await MetricRepository(db_session).list_active()
    assert len(definitions) == 6
    validate_metric_memos(corpus, definitions)


def test_promo_doc_covers_trap_t7(corpus: list[CorpusDocument]) -> None:
    document = next(item for item in corpus if item.entry.path == "promo_2026_summer.md")
    assert document.metadata.doc_type is DocumentType.PROMO_RULE
    assert document.metadata.effective_from == date(2026, 8, 5)
    assert document.metadata.effective_to == date(2026, 8, 21)
    for fact in (
        "promo_id = 17",
        "Asia/Shanghai",
        "[2026-08-05, 2026-08-21)",
        "先用后付",
        "扩展至服饰品类 apparel",
        "退货率较高",
        "支付同期群退款率",
        "2026-12-15",
        "相关性不等于因果",
        "数值必须来自独立查询证据",
    ):
        assert fact in document.text


def test_at_least_three_refund_policy_versions(corpus: list[CorpusDocument]) -> None:
    policies = [item for item in corpus if item.entry.path.startswith("refund_policy_v")]
    assert len(policies) == 3
    first, second, third = sorted(policies, key=lambda item: item.metadata.effective_from)
    assert first.metadata.supersedes is None
    assert first.metadata.effective_from == date(2025, 3, 1)
    assert first.metadata.effective_to == second.metadata.effective_from == date(2026, 1, 1)
    assert second.metadata.effective_to == third.metadata.effective_from == date(2026, 8, 1)
    assert second.metadata.supersedes == first.entry.path
    assert third.metadata.supersedes == second.entry.path
    assert third.metadata.effective_to is None
    for instant, expected in ((date(2026, 7, 31), second), (date(2026, 8, 1), third)):
        applicable = [
            item
            for item in policies
            if item.metadata.effective_from <= instant
            and (item.metadata.effective_to is None or instant < item.metadata.effective_to)
        ]
        assert applicable == [expected]
    assert "人工复核" in first.text
    assert "材料分组" in second.text
    assert "先用后付专用清单" in third.text


def test_binary_sources_have_real_chinese_text(corpus: list[CorpusDocument]) -> None:
    for document in corpus:
        if document.entry.format is CorpusFormat.EXCEL:
            assert document.text.count("## ") >= 2
            assert document.text.count("| --- |") >= 2
        if document.entry.format is CorpusFormat.PDF:
            assert "商品" in document.text
            assert "退款" in document.text
            assert "\ufffd" not in document.text


def test_corpus_variety_and_negative_controls(corpus: list[CorpusDocument]) -> None:
    summary = corpus_statistics(corpus)
    assert 200 <= summary.min_chars <= 500
    assert 3000 <= summary.max_chars <= 4500
    controls = {item.entry.path for item in corpus if item.entry.negative_control}
    assert controls == {
        "analysis/office_equipment_return.md",
        "analysis/employee_travel.md",
    }
    by_path = {item.entry.path: item.text for item in corpus}
    assert "暂不形成归因结论" in by_path["analysis/august_refunds.md"]
    assert "证据不足" in by_path["analysis/august_refunds.md"]
    assert "region_id = 3" in by_path["regions/east_rename.md"]
    assert "2026-05-01" in by_path["regions/east_rename.md"]
    assert "SKU-A1023" in by_path["sop/category_codes.xlsx"]
    assert "真实种子编码" in by_path["sop/category_codes.xlsx"]
    assert "培训供应商样例" in by_path["sop/category_codes.xlsx"]
    assert "不支持 category 粒度" in by_path["metrics/category_rollup.md"]
    assert "可以超过 100%" in by_path["metrics/refund_cross_month.md"]
