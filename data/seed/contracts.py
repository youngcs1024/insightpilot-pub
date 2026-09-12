"""Versioned generation and import contracts."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import ConflictError, ValidationError
from data.seed.rows import SeedRow

BASELINE_ORDERS = 50000
BASELINE_MONTHS = 18

TABLE_NAMES = (
    "regions",
    "promotions",
    "products",
    "customers",
    "orders",
    "order_items",
    "refunds",
    "inventory",
)
END = datetime(2026, 8, 31, 16, tzinfo=UTC)
CUTOFF = datetime(2026, 12, 14, 16, tzinfo=UTC)


class SeedError(ValidationError):
    """Invalid or inconsistent seed inputs; never retry."""

    code = "SEED_INVALID"


class SeedConflictError(ConflictError):
    """Existing output or database belongs to another dataset."""

    code = "SEED_CONFLICT"


class SeedContract(BaseModel):
    """Reject unknown fields in stored and process-boundary contracts."""

    model_config = ConfigDict(extra="forbid")


class Parameters(SeedContract):
    """Bounded deterministic workload; months end at September 2026."""

    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    orders: int = Field(default=50000, ge=1, le=1000000)
    months: int = Field(default=18, ge=1, le=120)

    @property
    def baseline(self) -> bool:
        """Full-scale datasets retain the declared quotas for every seed."""
        return self.orders == BASELINE_ORDERS and self.months == BASELINE_MONTHS


class FileManifest(SeedContract):
    """Canonical table identity."""

    table: str
    rows: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Manifest(SeedContract):
    """No wall-clock or absolute paths enter the content identity."""

    format_version: int = 1
    design_version: str = "2026-09-07-step2.1"
    parameters: Parameters
    files: list[FileManifest]


class Dataset(SeedContract):
    """Ordered typed rows passed from generation to export."""

    parameters: Parameters
    tables: list[list[SeedRow]]


class SeedResult(SeedContract):
    """Explicit outcome for operators and container acceptance."""

    outcome: str
    manifest: Manifest
    directory: Path
