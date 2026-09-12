"""Exercise the locked linter against product punctuation and retained safeguards."""

import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter

ROOT = Path(__file__).resolve().parents[2]


class RuffDiagnostic(BaseModel):
    """Stable machine-readable rule identity, independent of terminal rendering."""

    code: str


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ('message = "日期，过滤；范围（上海）"\n', None),
        ('message = "pаssword"\n', "RUF001"),  # noqa: RUF001 -- rejection fixture.
        ("class Broken(Exception):\n    pass\n", "N818"),
        ("value = 1  # noqa: RUF001\n", "RUF100"),
    ],
)
def test_ruff_product_punctuation_policy(tmp_path: Path, source: str, rule: str | None) -> None:
    target = tmp_path / "sample.py"
    target.write_text(source)
    result = subprocess.run(  # noqa: S603 -- fixed locked local linter and fixture path.
        [
            str(ROOT / ".venv/bin/ruff"),
            "check",
            "--output-format",
            "json",
            "--config",
            str(ROOT / "pyproject.toml"),
            "--select",
            "RUF001,RUF002,RUF003,N818,RUF100",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == (0 if rule is None else 1)
    diagnostics = TypeAdapter(list[RuffDiagnostic]).validate_json(result.stdout)
    assert [item.code for item in diagnostics] == ([] if rule is None else [rule])
