"""Model-only diagnostic settings carry no database, SSH or LLM credentials."""

from app.core.config_models import ModelRuntimeClientSettings
from app.core.settings_base import ProcessSettings


class ModelDiagnosticsSettings(ProcessSettings):
    """Independent process file for probes and benchmark clients."""

    process_name = "model-diagnostics"
    model_runtime: ModelRuntimeClientSettings
