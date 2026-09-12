"""Read kernel high-water marks on both cgroup versions without resetting them."""

from pathlib import Path

from spikes.capacity.contracts import FailureKind, ProbeError


def memory_peak(root: Path = Path("/sys/fs/cgroup")) -> int:
    """Read the current container's counter; callers use a private cgroup namespace."""
    for relative in ("memory.peak", "memory/memory.max_usage_in_bytes"):
        path = root / relative
        if not path.is_file():
            continue
        try:
            peak = int(path.read_text())
        except (OSError, ValueError) as exc:
            raise ProbeError(FailureKind.SAMPLING, "Invalid kernel memory peak.") from exc
        if peak > 0:
            return peak
    raise ProbeError(FailureKind.SAMPLING, "Kernel memory peak counter unavailable.")
