"""Compare complete, identity-matched FP16/FP32 benchmark artifacts."""

import argparse
import statistics
from pathlib import Path

from pydantic import BaseModel

from app.core.errors import ValidationError
from scripts.model_evidence import Benchmark

MIN_DISTINCT_RANKS = 2
MIN_SPEARMAN = 0.95


class Comparison(BaseModel):
    """Numerical correlation does not claim retrieval quality or abstention quality."""

    spearman: float
    max_absolute_difference: float
    passed: bool


def ranks(values: list[float]) -> list[float]:
    """Average ranks for ties, preserving original pair order."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            result[index] = (start + end - 1) / 2
        start = end
    return result


def compare(fp16: Benchmark, fp32: Benchmark) -> Comparison:
    """Refuse mismatched workloads or settings before measuring rank agreement."""
    if (
        fp16.input_sha256 != fp32.input_sha256
        or fp16.provenance != fp32.provenance
        or not fp16.stable_settings
        or not fp32.stable_settings
        or fp16.metadata.precision != "fp16"
        or fp32.metadata.precision != "fp32"
        or fp16.metadata.model_dump(exclude={"precision"})
        != fp32.metadata.model_dump(exclude={"precision"})
    ):
        raise ValidationError("Precision artifacts have incompatible identities")
    left, right = ranks(fp16.scores), ranks(fp32.scores)
    if len(set(left)) < MIN_DISTINCT_RANKS or len(set(right)) < MIN_DISTINCT_RANKS:
        raise ValidationError("Constant scores cannot establish rank correlation")
    correlation = statistics.correlation(left, right)
    return Comparison(
        spearman=correlation,
        max_absolute_difference=max(
            abs(a - b) for a, b in zip(fp16.scores, fp32.scores, strict=True)
        ),
        passed=correlation >= MIN_SPEARMAN and fp16.accepted and fp32.accepted,
    )


def main() -> None:
    """Print measured outcomes without rewriting either source artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        Benchmark.model_validate_json(args.fp16.read_text()),
        Benchmark.model_validate_json(args.fp32.read_text()),
    )
    print(result.model_dump_json())
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
