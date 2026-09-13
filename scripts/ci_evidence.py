"""Separate measured check results from evidence validity and upstream acceptance."""

from enum import StrEnum

from pydantic import BaseModel, Field

from scripts.ci_partitions import PARTITIONS, Partition
from scripts.ci_policy import Result


class CoverageState(StrEnum):
    """Retain why each partition does or does not provide usable measurements."""

    VALID = "valid"
    MISSING = "missing"
    EMPTY = "empty"
    UNREADABLE = "unreadable"
    INVALID = "invalid"
    EXPECTED_MISSING = "expected_missing"


class CheckEvidence(BaseModel):
    """A successful measurement cannot override invalid evidence or failed prerequisites."""

    checks_passed: bool | None
    evidence_valid: bool
    upstream: dict[Partition, Result] = Field(default_factory=dict)

    partitions: dict[Partition, CoverageState] = Field(default_factory=dict)

    @property
    def artifact_error(self) -> bool:
        """Only known absent output from skipped work is an expected evidence gap."""
        expected_gap = (
            set(self.partitions) == set(PARTITIONS)
            and CoverageState.EXPECTED_MISSING in self.partitions.values()
            and all(
                state is CoverageState.VALID
                or (
                    state is CoverageState.EXPECTED_MISSING
                    and self.upstream.get(name) is Result.SKIPPED
                )
                for name, state in self.partitions.items()
            )
        )
        return not self.evidence_valid and not expected_gap

    @property
    def accepted(self) -> bool:
        """Every dimension must succeed independently."""
        return (
            self.checks_passed is True
            and self.evidence_valid
            and all(state is CoverageState.VALID for state in self.partitions.values())
            and all(result is Result.SUCCESS for result in self.upstream.values())
        )

    def describe(self) -> str:
        """Keep a passed threshold visible when another stage blocks acceptance."""
        check = (
            "unavailable"
            if self.checks_passed is None
            else ("PASS" if self.checks_passed else "FAIL")
        )
        blockers = [
            f"{name}={value.value}"
            for name, value in self.upstream.items()
            if value is not Result.SUCCESS
        ]
        return (
            f"Coverage inputs: {self.partitions}; "
            f"Checks: {check}; evidence: {'valid' if self.evidence_valid else 'incomplete/invalid'}; "
            f"upstream blockers: {', '.join(blockers) if blockers else 'none'}; "
            f"acceptance: {'PASS' if self.accepted else 'FAIL'}."
        )
