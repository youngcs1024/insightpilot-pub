"""Shared ordinary-test identities used by selection, coverage and diagnostics."""

from typing import Literal

Partition = Literal["unit", "integration", "storage"]
PARTITIONS: tuple[Partition, ...] = ("unit", "integration", "storage")
SELECTIONS: dict[Partition, str] = {
    "unit": "not integration and not gpu and not external",
    "integration": "integration and not storage and not gpu and not external",
    "storage": "integration and storage and not gpu and not external",
}
