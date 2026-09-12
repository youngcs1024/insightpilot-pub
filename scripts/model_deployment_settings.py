"""Remote model deployment inputs; this module does not deploy anything."""

from pydantic import DirectoryPath, Field

from app.core.settings_base import ConfigModel, ProcessSettings


class ModelDeploymentFields(ConfigModel):
    """An immutable image, designated GPU and existing persistent cache directory."""

    image: str = Field(pattern=r"^(?:[^\s@]+@)?sha256:[0-9a-f]{64}$")
    gpu_id: str = Field(pattern=r"^GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
    cache_path: DirectoryPath


class ModelDeploymentSettings(ProcessSettings):
    """Operator-only configuration, isolated from the model's runtime secret."""

    process_name = "model-deployment"
    model_server: ModelDeploymentFields
