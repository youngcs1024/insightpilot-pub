"""Separate measured check results from evidence validity and upstream acceptance."""

from pydantic import BaseModel, Field

from scripts.ci_partitions import Partition
from scripts.ci_policy import Result


class CheckEvidence(BaseModel):
    """A successful measurement cannot override invalid evidence or failed prerequisites."""

    checks_passed: bool | None
    evidence_valid: bool
    upstream: dict[Partition, Result] = Field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        """Every dimension must succeed independently."""
        return (
            self.checks_passed is True
            and self.evidence_valid
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
            f"Checks: {check}; evidence: {'valid' if self.evidence_valid else 'incomplete/invalid'}; "
            f"upstream blockers: {', '.join(blockers) if blockers else 'none'}; "
            f"acceptance: {'PASS' if self.accepted else 'FAIL'}."
        )
