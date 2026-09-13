"""Print complete corpus statistics without database, GPU or network access."""

import argparse
import sys
from pathlib import Path

from app.core.errors import CorpusValidationError
from data.corpus_io import CORPUS_ROOT, corpus_statistics, load_corpus


def main(argv: list[str] | None = None) -> int:
    """Emit typed JSON, or a fixed safe diagnostic and a failing exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CORPUS_ROOT)
    args = parser.parse_args(argv)
    try:
        print(corpus_statistics(load_corpus(args.root)).model_dump_json(indent=2))
        return 0
    except CorpusValidationError as exc:
        print(exc.code + ": " + exc.user_message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
