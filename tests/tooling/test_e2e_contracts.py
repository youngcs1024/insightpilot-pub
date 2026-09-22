"""Deployment and partition regressions for ordinary E2E, without containers."""

from pathlib import Path

import pytest
import yaml

from scripts.check_deployment_contracts import load_compose
from scripts.e2e_contracts import isolation_issues
from scripts.public_files import allowed_path

pytest_plugins = ("pytester",)

ROOT = Path(__file__).resolve().parents[2]


def test_e2e_configuration_is_explicitly_reviewed_and_public() -> None:
    path = ROOT / "docker-compose.e2e.yml"
    load_compose(path)
    assert isolation_issues(path) == []
    assert allowed_path("docker-compose.e2e.yml")
    assert allowed_path("docker/Dockerfile.e2e.dockerignore")
    production = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert "model-tunnel" in production["services"]
    assert "inference" not in production["services"]
    assert "tests" in (ROOT / ".dockerignore").read_text().splitlines()


@pytest.mark.parametrize(
    "mutation", ["provider", "model", "business", "ssh", "volume", "port", "gpu", "secret"]
)
def test_e2e_isolation_fails_closed(tmp_path: Path, mutation: str) -> None:
    document = yaml.safe_load((ROOT / "docker-compose.e2e.yml").read_text())
    api = document["services"]["api"]
    if mutation == "provider":
        api["environment"]["IP_LLM__BASE_URL"] = "https://example.invalid/v1"
    elif mutation == "model":
        api["environment"]["IP_MODEL_RUNTIME"] = '{"base_url":"http://model-tunnel:8100"}'
    elif mutation == "business":
        api["environment"]["IP_BUSINESS__PASSWORD"] = "synthetic"  # noqa: S105 -- intentionally invalid test configuration.
    elif mutation == "ssh":
        api["volumes"] = ["./ssh:/run/ssh:ro"]
    elif mutation == "volume":
        document["volumes"]["pgdata"] = {"external": True}
    elif mutation == "port":
        api["ports"] = ["18000:8000"]
    elif mutation == "gpu":
        api["gpus"] = "all"
    else:
        api["env_file"] = ".env"
    path = tmp_path / "compose.yml"
    path.write_text(yaml.safe_dump(document))
    assert isolation_issues(path)


@pytest.mark.parametrize(
    ("markers", "valid"),
    [
        (["e2e", "integration"], True),
        (["e2e"], False),
        (["e2e", "integration", "storage"], False),
        (["e2e", "external"], True),
        (["e2e", "external", "gpu"], True),
    ],
)
def test_e2e_partition_contract_before_deselection(
    pytester: pytest.Pytester, markers: list[str], valid: bool
) -> None:
    pytester.makeini("[pytest]\nmarkers =\n e2e\n integration\n storage\n external\n gpu\n")
    pytester.makeconftest((ROOT / "tests/collection_support.py").read_text())
    pytester.makepyfile(
        "import pytest\n"
        + "".join(f"@pytest.mark.{marker}\n" for marker in markers)
        + "def test_case():\n    raise AssertionError('body executed')\n"
    )
    result = pytester.runpytest_subprocess("--collect-only", "-m", "not e2e")
    assert result.ret == (
        pytest.ExitCode.NO_TESTS_COLLECTED if valid else pytest.ExitCode.USAGE_ERROR
    )


def test_test_image_and_evidence_are_not_production_inputs() -> None:
    source = (ROOT / "docker/Dockerfile.api").read_text()
    assert "COPY tests" not in source
    assert "e2e" not in source
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["integration"]["steps"]
    upload = next(step for step in steps if step.get("id") == "test_upload")
    assert "e2e-evidence/" in upload["with"]["path"]
    assert "always()" in upload["if"]
