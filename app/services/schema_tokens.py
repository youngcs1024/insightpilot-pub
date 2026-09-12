"""Offline cl100k_base measurement using the public tiktoken Encoding constructor."""

import base64
import gzip
import hashlib
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

import tiktoken

from app.core.errors import SchemaMetadataError

VOCABULARY = Path(__file__).resolve().parents[2] / "scripts/data/cl100k_base.tiktoken.gz"
VOCABULARY_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
# Frozen public cl100k_base definition from tiktoken 0.14.0 (MIT); see data/README.md.
PATTERN = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"


@lru_cache(maxsize=1)
def encoding(vocabulary: Path = VOCABULARY) -> tiktoken.Encoding:
    """Verify a bundled vocabulary; never perform an implicit network download."""
    try:
        contents = gzip.decompress(vocabulary.read_bytes())
        if hashlib.sha256(contents).hexdigest() != VOCABULARY_SHA256:
            raise SchemaMetadataError("Tokenizer vocabulary checksum mismatch.")
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in contents.splitlines() if line)
        }
    except (OSError, EOFError, zlib.error) as exc:
        raise SchemaMetadataError("Tokenizer vocabulary is unavailable.") from exc
    return tiktoken.Encoding(
        name="cl100k_base",
        pat_str=PATTERN,
        mergeable_ranks=ranks,
        special_tokens={
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
    )


class SchemaTokenCounter:
    """Initialize offline resources before entering graph nodes."""

    name: Literal["cl100k_base"] = "cl100k_base"

    def __init__(self) -> None:
        self._encoding = encoding()

    def count(self, text: str) -> int:
        """Measure literal schema text, including special-token spellings."""
        return len(self._encoding.encode(text, disallowed_special=()))
