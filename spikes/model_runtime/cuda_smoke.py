"""Offline CUDA evidence probe, invoked only in an authorized GPU container."""

import argparse
import importlib
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, DirectoryPath, Field, FiniteFloat, NonNegativeInt

DENSE_DIMENSION = 1024
TEXT_COUNT = 16
PAIR_COUNT = 20
DenseRow = Annotated[
    list[FiniteFloat], Field(min_length=DENSE_DIMENSION, max_length=DENSE_DIMENSION)
]
SparseRow = Annotated[
    dict[NonNegativeInt, Annotated[FiniteFloat, Field(ge=0)]], Field(min_length=1)
]
Score = Annotated[FiniteFloat, Field(ge=0, le=1)]


class ProbeInput(BaseModel):
    """Snapshot paths must name immutable model revisions in a pre-fetched cache."""

    embed_snapshot: DirectoryPath
    rerank_snapshot: DirectoryPath


class Outputs(BaseModel):
    """Validate every value, dimension and count before emitting successful evidence."""

    dense: Annotated[list[DenseRow], Field(min_length=TEXT_COUNT, max_length=TEXT_COUNT)]
    sparse: Annotated[list[SparseRow], Field(min_length=TEXT_COUNT, max_length=TEXT_COUNT)]
    scores: Annotated[list[Score], Field(min_length=PAIR_COUNT, max_length=PAIR_COUNT)]


class Evidence(BaseModel):
    """Measured runtime identity and output summary, never a dependency-only success."""

    python: str
    packages: dict[str, str]
    cuda: str
    device: str
    embed_revision: str
    rerank_revision: str
    precision: str = "fp16"
    dense_shape: tuple[int, int] = (TEXT_COUNT, DENSE_DIMENSION)
    sparse_counts: list[int]
    scores: list[Score]
    peak_allocated_bytes: int
    elapsed_s: float


def valid_snapshot(path: Path) -> bool:
    """Reject mutable branch names and absent model configuration before loading weights."""
    return (
        len(path.name) == 40  # noqa: PLR2004 -- Hugging Face commit SHA length.
        and all(character in "0123456789abcdef" for character in path.name)
        and (path / "config.json").is_file()
    )


def run_probe(config: ProbeInput) -> Evidence:
    """Load both fixed snapshots and execute one bounded batch on the sole visible GPU."""
    # Probe process only: enforce offline loading before importing model libraries.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Probe requires exactly one authorized visible CUDA GPU; no CPU fallback.")
    models = importlib.import_module("FlagEmbedding")
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    embedder = models.BGEM3FlagModel(
        str(config.embed_snapshot),
        devices=["cuda:0"],
        use_fp16=True,
    )
    reranker = models.FlagReranker(
        str(config.rerank_snapshot),
        devices=["cuda:0"],
        use_fp16=True,
    )
    encoded = embedder.encode(
        ["华东地区订单退款政策与商品销售数据。"] * TEXT_COUNT,
        batch_size=16,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    scores = reranker.compute_score(
        [
            ["退款政策是什么?", f"第 {index} 条政策: 签收后七天内可以申请退货。"]
            for index in range(PAIR_COUNT)
        ],
        batch_size=16,
        max_length=320,
        normalize=True,
    )
    outputs = Outputs(dense=encoded["dense_vecs"], sparse=encoded["lexical_weights"], scores=scores)
    for model in (embedder.model, reranker.model):
        parameter = next(model.parameters())
        if parameter.device.type != "cuda" or parameter.dtype != torch.float16:
            raise SystemExit("Model did not execute with CUDA FP16 parameters.")
    torch.cuda.synchronize()
    return Evidence(
        python=platform.python_version(),
        packages={name: version(name) for name in ("torch", "FlagEmbedding", "transformers")},
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),
        embed_revision=config.embed_snapshot.name,
        rerank_revision=config.rerank_snapshot.name,
        sparse_counts=[len(row) for row in outputs.sparse],
        scores=outputs.scores,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        elapsed_s=time.monotonic() - start,
    )


def main() -> int:
    """Print evidence only after all inference checks pass; errors produce a nonzero exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-snapshot", type=Path, required=True)
    parser.add_argument("--rerank-snapshot", type=Path, required=True)
    args = parser.parse_args()
    config = ProbeInput(embed_snapshot=args.embed_snapshot, rerank_snapshot=args.rerank_snapshot)
    if not all(valid_snapshot(path) for path in (config.embed_snapshot, config.rerank_snapshot)):
        parser.error(
            "Both paths must be immutable 40-hex snapshot directories containing config.json"
        )
    print(run_probe(config).model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
