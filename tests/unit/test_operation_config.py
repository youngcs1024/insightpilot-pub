"""Every newly introduced call budget is configurable, finite and bounded."""

import pytest
from pydantic import BaseModel, ValidationError

from app.core.config_models import (
    LLMSettings,
    MCPSettings,
    ModelRuntimeClientSettings,
    RetrievalSettings,
    Settings,
)
from app.core.settings_base import environment_keys


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (LLMSettings, "timeout_s"),
        (MCPSettings, "timeout_s"),
        (RetrievalSettings, "search_timeout_s"),
        (ModelRuntimeClientSettings, "embed_timeout_s"),
        (ModelRuntimeClientSettings, "rerank_timeout_s"),
    ],
)
@pytest.mark.parametrize("invalid", [0, -1, 121, float("nan"), float("inf")])
def test_operation_timeout_bounds(model: type[BaseModel], field: str, invalid: float) -> None:
    with pytest.raises(ValidationError) as caught:
        model.model_validate({field: invalid})
    assert any(error["loc"] == (field,) for error in caught.value.errors())


def test_new_settings_extend_process_allowlist() -> None:
    assert {
        "IP_LLM__TIMEOUT_S",
        "IP_MCP__TIMEOUT_S",
        "IP_RETRIEVAL__SEARCH_TIMEOUT_S",
        "IP_MODEL_RUNTIME__EMBED_TIMEOUT_S",
        "IP_MODEL_RUNTIME__RERANK_TIMEOUT_S",
    } <= environment_keys(Settings)
