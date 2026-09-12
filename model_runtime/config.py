"""Model runtime settings; no API singleton, network, or model initialization."""

from typing import Literal

from pydantic import Field

from app.core.settings_base import ConfigModel, ProcessSettings, Secret


class ModelServerSettings(ConfigModel):
    """Pinned model identity and bounded single-GPU inference configuration."""

    auth_token: Secret = Field(repr=False)
    embed_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    rerank_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    device: Literal["cuda:0"] = "cuda:0"
    precision: Literal["fp16", "fp32"] = "fp16"
    workers: int = Field(default=1, ge=1, le=1)
    max_concurrency: int = Field(default=1, ge=1, le=1)
    queue_capacity: int = Field(default=8, ge=1, le=8)
    embed_batch: int = Field(default=16, ge=1, le=16)
    rerank_batch: int = Field(default=16, ge=1, le=16)
    embed_max_length: int = Field(default=512, ge=1, le=512)
    rerank_max_length: int = Field(default=320, ge=1, le=320)


class ModelRuntimeSettings(ProcessSettings):
    """Only runtime fields are accepted; deployment inputs use a different file."""

    process_name = "model-runtime"
    model_server: ModelServerSettings
