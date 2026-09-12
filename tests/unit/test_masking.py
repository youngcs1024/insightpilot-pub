"""Business data never survives telemetry projection, including indirect copies."""

import json

import pytest
from pydantic import SecretStr

from app.agents.summarize import package_result
from app.core import masking
from tests.agents.support import result


def test_result_rows_redacted() -> None:
    data = package_result(result([[739159]]), ["private-assumption"])
    before = data.model_dump_json()
    projected = masking.mask(data)
    assert projected["rows"] == "[<redacted: 1 rows × 1 cols>]"  # noqa: RUF001
    assert "739159" not in json.dumps(projected)
    assert "private-assumption" not in json.dumps(projected)
    assert data.model_dump_json() == before
    assert masking.mask(projected) == projected


def test_column_names_preserved() -> None:
    projected = masking.mask(package_result(result(), []))
    assert projected["columns"] == [{"name": "n", "type": "int4"}]


def test_pii_column_values_redacted() -> None:
    data = {
        "columns": [{"name": "email", "type": "text", "pii": True}],
        "rows": [["person@example.invalid"]],
        "result_summary": {
            "sample_rows": [["person@example.invalid"]],
            "columns": [{"name": "email", "minimum": "person@example.invalid"}],
        },
        "generation_block": "person@example.invalid",
        "messages": [{"content": "person@example.invalid"}],
        "answer": "person@example.invalid",
        "sql": "SELECT 'person@example.invalid'",
    }
    assert "person@example.invalid" not in json.dumps(masking.mask(data))


@pytest.mark.parametrize(
    "credential", ["Bearer secret-value", "sk-secret-value", "password=secret-value"]
)
def test_credential_pattern_redacted(credential: str) -> None:
    assert "secret-value" not in str(masking.mask({"model": credential}))
    assert masking.mask({"api_key": "secret-value", "role": SecretStr("secret-value")}) == {
        "api_key": "***",
        "role": "***",
    }


def test_masking_failure_redacts_rather_than_exposes(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(data: object, depth: int) -> object:
        raise RuntimeError("private-value")

    monkeypatch.setattr(masking, "_walk", broken)
    assert masking.mask({"rows": [["private-value"]]}) == masking.REDACTED


def test_empty_rows_keep_column_count_and_unknown_objects_are_not_stringified() -> None:
    class Dangerous:
        def __str__(self) -> str:  # noqa: PLE0307 -- deliberately fails if rendered.
            pytest.fail("Unknown object must not be rendered")

    assert masking.mask(Dangerous()) == masking.REDACTED
    assert masking.mask({"rows": [], "columns": [{"name": "n", "type": "int4"}]})["rows"] == (
        "[<redacted: 0 rows × 1 cols>]"  # noqa: RUF001
    )


def test_token_counts_preserved_but_business_total_removed() -> None:
    assert masking.mask({"prompt_tokens": 10, "completion_tokens": 2, "total": 739159}) == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total": masking.REDACTED,
    }


@pytest.mark.parametrize(
    "value", ['"private-text"', '["private-text"]', '[["private-text"]]', "private-text"]
)
def test_unstructured_serialized_output_is_never_exported(value: str) -> None:
    safe = masking.safe_attributes({"langfuse.observation.output": value})
    assert "private-text" not in json.dumps(safe)
