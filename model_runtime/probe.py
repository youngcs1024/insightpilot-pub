"""Authenticated local readiness probe; secrets never enter command arguments."""

import argparse
import json
import urllib.request

from app.schemas.model_runtime import ReadyResult
from model_runtime.config import ModelRuntimeSettings


def main() -> None:
    """Return nonzero on unavailable, unauthenticated or malformed readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", action="store_true")
    args = parser.parse_args()
    headers = {}
    settings = ModelRuntimeSettings.load().model_server if args.ready else None
    if settings is not None:
        headers["Authorization"] = "Bearer " + settings.auth_token.get_secret_value()
    request = urllib.request.Request(  # noqa: S310 -- fixed HTTP loopback URLs only.
        "http://127.0.0.1:8100/ready" if args.ready else "http://127.0.0.1:8100/health",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310 -- fixed loopback URL.
            result = json.load(response)
        if settings is not None:
            ready = ReadyResult.model_validate(result)
            passed = (
                ready.metadata.embed_revision == settings.embed_revision
                and ready.metadata.rerank_revision == settings.rerank_revision
                and ready.metadata.precision == settings.precision
            )
        else:
            passed = isinstance(result, dict) and result.get("status") == "ok"
    except (OSError, ValueError):
        passed = False
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
