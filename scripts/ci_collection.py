"""Bounded collection evidence, independent of pytest and application settings."""

from typing import Annotated

from pydantic import BaseModel, Field

MAX_COLLECTORS = 50
MAX_NODE_LENGTH = 300


class CollectionReport(BaseModel):
    """Preserve raw collection status without copying tracebacks or executing tests."""

    raw_exit: int = Field(ge=0, le=5)
    collected: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    failed_nodes: list[Annotated[str, Field(max_length=MAX_NODE_LENGTH)]] = Field(
        default_factory=list, max_length=MAX_COLLECTORS
    )

    @property
    def accepted(self) -> bool:
        """A successful empty or contradictory report cannot satisfy collection."""
        return (
            self.raw_exit == 0
            and self.collected > 0
            and self.failed_count == 0
            and not self.failed_nodes
        )
