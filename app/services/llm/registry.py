"""Load committed probe evidence; never discover model capabilities by guessing."""

from enum import IntEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.core.config_models import LLMSettings
from app.core.errors import LlmConfigurationError
from app.core.llm_config import ModelRole, ModelRoleSettings
from app.core.settings_base import PROJECT_ROOT


class StructuredTier(IntEnum):
    """The ordered structured-output ladder."""

    NATIVE = 1
    TOOL = 2
    PROMPTED = 3


class CapabilityReport(BaseModel):
    """Runtime projection of the version-one probe report, excluding raw prompts."""

    schema_version: Literal[1]
    evidence_source: Literal["live_api"]
    model: str = Field(min_length=1, max_length=200)
    execution_complete: Literal[True]
    recommended_tier: StructuredTier


class ModelRegistry:
    """Private capability and settings snapshots; calls never update either."""

    def __init__(self, settings: LLMSettings, reports: list[CapabilityReport]) -> None:
        self._roles = {role: settings.for_role(role) for role in ModelRole}
        self._tiers = {report.model: report.recommended_tier for report in reports}
        if len(self._tiers) != len(reports):
            raise LlmConfigurationError()
        for config in self._roles.values():
            chain = [config.model, *config.fallback_models]
            if len(set(chain)) != len(chain) or any(model not in self._tiers for model in chain):
                raise LlmConfigurationError()

    @classmethod
    def load(cls, settings: LLMSettings) -> "ModelRegistry":
        """Read local reports once before accepting requests, relative to the project."""
        try:
            reports = [
                CapabilityReport.model_validate_json(_absolute(path).read_bytes())
                for path in settings.capabilities_paths
            ]
            return cls(settings, reports)
        except (OSError, ValidationError) as exc:
            raise LlmConfigurationError() from exc

    def role(self, role: ModelRole) -> ModelRoleSettings:
        """Return an independent settings copy so callers cannot modify the registry."""
        return self._roles[role].model_copy(deep=True)

    def tier(self, model: str) -> StructuredTier:
        """Return the probed starting tier for a configured model."""
        try:
            return self._tiers[model]
        except KeyError as exc:
            raise LlmConfigurationError() from exc


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path
