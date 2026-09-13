"""Pure source, identity, splitting and CLI failure contracts for ingestion."""

# ruff: noqa: PLR2004 -- fixed splitter limits and synthetic acceptance values.

import os
from pathlib import Path

import openpyxl
import pytest
from pydantic import ValidationError

from app.core.errors import CorpusValidationError, IngestionAlreadyRunning
from app.core.settings_base import ProcessSettings
from app.schemas.ingestion import PreparedDocument, canonical, document_id
from app.services.chunking import prepare, table_fragments
from app.services.ingestion_config import IngestionSettings
from app.services.ingestion_plan import stage
from app.services.loaders import discover, load_source, snapshot
from data.corpus_io import read_document
from scripts import ingest
from tests.corpus_support import entry, metadata_source, write_inventory
from tests.ingestion_support import write_sources

ROOT = Path(__file__).resolve().parents[2]


def prepared(root: Path, name: str = "rule.md", config: IngestionSettings | None = None) -> PreparedDocument:
    source = entry(name)
    snapshot_value = snapshot(root, source, 20_000_000)
    return prepare(load_source(source, snapshot_value.raw, snapshot_value.sidecar, snapshot_value.fingerprint), config or IngestionSettings())


def test_markdown_parent_includes_heading_trail(tmp_path: Path) -> None:
    write_sources(tmp_path, ["rule.md"], body="# 一级\n\n## 二级\n\n### 三级\n\n" + "业务内容。" * 200)
    result = prepared(tmp_path)
    assert len(result.chunks) > 1
    assert all(item.heading_path == "一级 / 二级 / 三级" for item in result.chunks)
    assert all(item.parent_content.startswith("# 一级\n## 二级\n### 三级") for item in result.chunks)
    assert all(len(item.content) <= 600 for item in result.chunks)
    assert result.chunks[-1].parent_content != result.chunks[-1].content


def test_xlsx_split_repeats_header_row(tmp_path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "费用"
    sheet.append(["SKU", "费用"])
    for index in range(70):
        sheet.append([f"SKU-{index}", "费用标准" * 5])
    workbook.create_sheet("说明").append(["说明", "备注"])
    workbook.save(tmp_path / "fees.xlsx")
    workbook.close()
    (tmp_path / "fees.xlsx.meta.yaml").write_text(metadata_source())
    result = prepared(tmp_path, "fees.xlsx")
    chunks = [item for item in result.chunks if item.heading_path == "费用"]
    assert len(chunks) > 1
    assert all(item.content.startswith("| SKU | 费用 |\n| --- | --- |") for item in chunks)
    assert sum(len(item.content.splitlines()) - 2 for item in chunks) == 70
    assert any(item.heading_path == "说明" for item in result.chunks)


def test_oversized_table_row_stays_intact() -> None:
    row = "| " + "甲" * 700 + " |"
    fragments = table_fragments("| 标题 |\n| --- |\n" + row + "\n| 乙 |", 600)
    assert fragments[0].splitlines()[-1] == row
    assert len(fragments) == 2


def test_fresh_roots_reproduce_identities(tmp_path: Path) -> None:
    other = tmp_path / "second"
    other.mkdir()
    write_sources(tmp_path, ["rule.md"])
    write_sources(other, ["rule.md"])
    assert prepared(tmp_path) == prepared(other)


def test_line_endings_and_unicode_normalize(tmp_path: Path) -> None:
    write_sources(tmp_path, ["rule.md"], body="# Café\n\n规则。")
    before = prepared(tmp_path)
    path = tmp_path / "rule.md"
    path.write_bytes(path.read_bytes().replace("é".encode(), "e\u0301".encode()).replace(b"\n", b"\r\n"))
    after = prepared(tmp_path)
    assert before.document.document_version == after.document.document_version
    assert [item.chunk_uuid for item in before.chunks] == [item.chunk_uuid for item in after.chunks]
    assert before.document.source_fingerprint != after.document.source_fingerprint
    assert document_id("Café.md") == document_id("Cafe\u0301.md")


def test_metadata_and_splitter_settings_version_identity(tmp_path: Path) -> None:
    write_sources(tmp_path, ["rule.md"])
    before = prepared(tmp_path)
    changed = prepared(tmp_path, config=IngestionSettings(child_size=300))
    assert before.document.document_version == changed.document.document_version
    assert before.document.chunking_version != changed.document.chunking_version
    path = tmp_path / "rule.md"
    path.write_text(path.read_text().replace("title: 测试规则", "title: 新标题"))
    assert prepared(tmp_path).document.document_version != before.document.document_version


def test_snapshot_uses_same_bytes_after_source_changes(tmp_path: Path) -> None:
    write_sources(tmp_path, ["rule.md"])
    snapshot_value = snapshot(tmp_path, entry(), 20000)
    before = prepared(tmp_path)
    (tmp_path / "rule.md").write_text("broken")
    assert prepare(load_source(entry(), snapshot_value.raw, snapshot_value.sidecar, snapshot_value.fingerprint), IngestionSettings()) == before


def test_unchanged_stage_does_not_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_sources(tmp_path, ["rule.md"])
    old = prepared(tmp_path).document
    def forbidden(*args: object) -> None:
        pytest.fail("unchanged snapshot was parsed")
    monkeypatch.setattr("app.services.ingestion_plan.load_source", forbidden)
    plan = stage(tmp_path, [old], IngestionSettings())
    assert not plan.changed and not plan.failed and not plan.deleted


def test_nonsemantic_source_changes_only_refresh_fingerprint(tmp_path: Path) -> None:
    write_sources(tmp_path, ["rule.md"])
    old = prepared(tmp_path).document
    path = tmp_path / "rule.md"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    plan = stage(tmp_path, [old], IngestionSettings())
    assert not plan.changed and len(plan.refreshed) == 1
    assert plan.refreshed[0].document_version == old.document_version


def test_empty_manifest_retires_last_document(tmp_path: Path) -> None:
    write_sources(tmp_path, ["rule.md"])
    old = prepared(tmp_path).document
    write_inventory(tmp_path, [])
    assert len(stage(tmp_path, [old], IngestionSettings()).deleted) == 1


@pytest.mark.parametrize("content", ["", "{}", "documents: [", "documents: []\ndocuments: []"])
def test_bad_manifest_aborts_discovery(tmp_path: Path, content: str) -> None:
    (tmp_path / "MANIFEST.yaml").write_text(content)
    with pytest.raises(CorpusValidationError):
        discover(tmp_path)


def test_escaping_symlink_and_size_limit_are_file_failures(tmp_path: Path) -> None:
    write_sources(tmp_path, ["rule.md"])
    old = prepared(tmp_path).document
    plan = stage(tmp_path, [old], IngestionSettings(max_source_bytes=1))
    assert plan.failed and not plan.deleted
    (tmp_path / "link.md").symlink_to(tmp_path.parent / "outside.md")
    write_inventory(tmp_path, [entry("link.md")])
    assert stage(tmp_path, [], IngestionSettings()).failed


def test_corpus_binary_loaders_agree_with_authoring_and_keep_pdf_pages() -> None:
    root = ROOT / "data/corpus"
    sources = discover(root)
    binaries = [source for source in sources.documents if source.path.endswith((".pdf", ".xlsx"))]
    assert len(binaries) == 6
    for source in binaries:
        snapshot_value = snapshot(root, source, 20_000_000)
        loaded = load_source(source, snapshot_value.raw, snapshot_value.sidecar, snapshot_value.fingerprint)
        authored = read_document(root, source)
        result = prepare(loaded, IngestionSettings())
        assert result.document.metadata == authored.metadata and result.chunks
        if source.path.endswith(".pdf"):
            assert all(item.page is not None and item.page >= 1 for item in result.chunks)
            assert all(item.text.strip() in authored.text for item in loaded.parts)


@pytest.mark.parametrize("changes", [{"child_overlap": 600}, {"parent_overlap": 2000}, {"timeout_s": 0}])
def test_invalid_splitter_settings(changes: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        IngestionSettings(**changes)


def test_ingestion_process_is_credential_scoped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in os.environ:
        if key.startswith("IP_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(ProcessSettings, "project_root", tmp_path)
    monkeypatch.setenv("IP_DATABASE__APP_PASSWORD", "synthetic-only")
    monkeypatch.setenv("IP_MODEL_RUNTIME__AUTH_TOKEN", "synthetic-only")
    assert ingest.IngestionProcessSettings.load().ingestion.child_size == 600
    monkeypatch.setenv("IP_MIGRATION__PASSWORD", "must-not-leak")
    with pytest.raises(ValidationError) as error:
        ingest.IngestionProcessSettings.load()
    assert "must-not-leak" not in str(error.value)


def test_cli_lock_failure_has_nonzero_safe_exit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    settings = object()
    monkeypatch.setattr(ingest.IngestionProcessSettings, "load", lambda: settings)
    async def run(*args: object) -> int:
        raise IngestionAlreadyRunning("secret backend prose")
    monkeypatch.setattr(ingest, "run", run)
    assert ingest.main(["--corpus", "data/corpus"]) == 1
    output = capsys.readouterr().err
    assert "INGESTION_ALREADY_RUNNING" in output and "secret backend prose" not in output


def test_canonical_json_is_stable() -> None:
    assert canonical({"b": "中文", "a": 1}) == '{"a":1,"b":"中文"}'
