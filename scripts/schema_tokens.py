"""Compatibility entry point for the application-owned offline tokenizer."""

from functools import lru_cache

import tiktoken

from app.services.schema_tokens import VOCABULARY
from app.services.schema_tokens import encoding as load_encoding


@lru_cache(maxsize=1)
def encoding() -> tiktoken.Encoding:
    """Retain the CLI entry point and its explicit local vocabulary override."""
    return load_encoding(VOCABULARY)
