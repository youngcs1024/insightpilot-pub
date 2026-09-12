"""The public SQL fixture must retain complete, valid acceptance evidence."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from app.core.errors import ValidationError
from tests.seed_cases import CASE_FILE, TRAP_IDS, load_cases


def test_frozen_cases_cover_every_trap_and_original_intervals() -> None:
    cases = load_cases().traps
    assert {case.id for case in cases} == TRAP_IDS
    expected = (
        (".05", ".20"),
        (".05", ".50"),
        (".05", ".15"),
        ("1", "1"),
        (".05", ".20"),
        (".20", ".40"),
        (".20", ".40"),
        (".05", ".15"),
    )
    assert [(case.lower_delta, case.upper_delta) for case in cases] == [
        (Decimal(low), Decimal(high)) for low, high in expected
    ]


@pytest.mark.parametrize(
    "damage",
    ["missing", "duplicate", "same_sql", "multiple_sql", "write_sql", "bounds", "eval_ids"],
)
def test_damaged_resources_fail(tmp_path: Path, damage: str) -> None:
    body = json.loads(CASE_FILE.read_text())
    if damage == "missing":
        body["traps"].pop()
    elif damage == "duplicate":
        body["traps"][-1] = body["traps"][0]
    elif damage == "same_sql":
        body["traps"][0]["correct_sql"] = body["traps"][0]["naive_sql"]
    elif damage == "multiple_sql":
        body["traps"][0]["naive_sql"] += " SELECT 1;"
    elif damage == "write_sql":
        body["traps"][0]["naive_sql"] = "DELETE FROM biz.orders;"
    elif damage == "bounds":
        body["traps"][0]["upper_delta"] = "0.01"
    else:
        body["traps"][1]["eval_cases"] = body["traps"][0]["eval_cases"]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValidationError):
        load_cases(path)


@pytest.mark.parametrize("body", ["", "{}", '{"schema_version":2}', "not json"])
def test_invalid_resource_fails(tmp_path: Path, body: str) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(body)
    with pytest.raises(ValidationError):
        load_cases(path)


def test_missing_resource_fails(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_cases(tmp_path / "missing.json")
