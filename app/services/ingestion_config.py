"""Bounded ingestion settings without importing the API configuration singleton."""

from typing import Self

from pydantic import Field, model_validator

from app.core.settings_base import ConfigModel, require_configuration
from app.schemas.ingestion import canonical, digest


class IngestionSettings(ConfigModel):
    """A splitter change deliberately creates a new chunking version."""

    child_size: int = Field(default=600, ge=1, le=32000)
    child_overlap: int = Field(default=100, ge=0, le=31999)
    parent_size: int = Field(default=2000, ge=1, le=20000)
    parent_overlap: int = Field(default=200, ge=0, le=19999)
    table_size: int = Field(default=600, ge=1, le=32000)
    timeout_s: float = Field(default=1800, ge=1, le=7200)
    max_source_bytes: int = Field(default=20_000_000, ge=1, le=100_000_000)

    @model_validator(mode="after")
    def overlaps(self) -> Self:
        """Every overlap must be smaller than its corresponding window."""
        require_configuration(self.child_overlap < self.child_size, "Invalid child overlap")
        require_configuration(self.parent_overlap < self.parent_size, "Invalid parent overlap")
        return self

    def splitter_config(self) -> str:
        """Freeze all text-affecting choices, excluding operational budgets."""
        return canonical(
            {
                "normalizer": "nfc-lf-v1",
                "loader": "markdown-openpyxl-pypdf-v1",
                "splitter": "langchain-text-splitters-1.1.2-v1",
                "headers": ["#", "##", "###"],
                "separators": ["\n\n", "\n", " ", ""],
                **self.model_dump(exclude={"timeout_s", "max_source_bytes"}),
            }
        )

    def chunking_version(self) -> str:
        """Hash the exact splitter description persisted in the active manifest."""
        return digest(self.splitter_config())
