"""Standalone CPU-only HTTP fixture copied into the isolated SSH target container."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

TOKEN = "isolated-model-token"  # noqa: S105 -- public synthetic fixture credential.


class Handler(BaseHTTPRequestHandler):
    """Minimal authenticated wire responses; no production inference or imports."""

    metadata: dict[str, object] = {}

    def send_json(self, status: int, body: dict[str, object]) -> None:
        content = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def authenticated(self) -> bool:
        if self.headers.get("Authorization") == "Bearer " + TOKEN:
            return True
        self.send_json(401, {
            "code": "MODEL_RUNTIME_AUTH", "message": "Authentication required",
            "request_id": "fixture", "retryable": False,
        })
        return False

    def do_GET(self) -> None:  # noqa: N802 -- standard-library HTTP handler hook.
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
        elif self.authenticated():
            self.send_json(200, {
                "ready": True, "request_id": "fixture", "metadata": self.metadata,
            })

    def do_POST(self) -> None:  # noqa: N802 -- standard-library HTTP handler hook.
        if not self.authenticated():
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        texts = body["texts"]
        self.send_json(200, {
            "request_id": self.headers["X-Request-ID"],
            "ms": 1, "queue_ms": 0, "inference_ms": 1,
            "metadata": self.metadata,
            "dense": [[1.0, *([0.0] * 1023)] for _ in texts],
            "sparse": [{"42": 0.5} for _ in texts],
        })

    def log_message(self, format: str, *args: object) -> None:
        """The fixture never logs request content or credentials."""


def main() -> None:
    """Read only synthetic fixture metadata after explicit server startup."""
    Handler.metadata = json.loads(Path("/fixture/model_metadata.json").read_text())
    with HTTPServer(("127.0.0.1", 8100), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
