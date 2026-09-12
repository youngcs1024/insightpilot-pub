"""Authenticated, loopback-only echo server for Step 0.3."""

import secrets

import structlog
from mcp.server import MCPServer
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl, SecretStr

from spikes.mcp_v2.settings import ENDPOINT, Settings


class StaticTokenVerifier:
    """Verify the operator-provided spike token without an external auth service."""

    def __init__(self, token: SecretStr) -> None:
        self._token = token

    async def verify_token(self, token: str) -> AccessToken | None:
        """Reject unknown tokens and assign a stable session principal."""
        if not secrets.compare_digest(token.encode(), self._token.get_secret_value().encode()):
            return None
        return AccessToken(token=token, client_id="insightpilot-spike", scopes=["echo"])


def main() -> None:
    """Serve the sole echo tool with SDK-managed authentication and sessions."""
    settings = Settings()
    server: MCPServer[None] = MCPServer(
        "insightpilot-step-0-3",
        log_level="WARNING",
        token_verifier=StaticTokenVerifier(settings.mcp.auth_token),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl("http://127.0.0.1:9100"),
            resource_server_url=AnyHttpUrl(ENDPOINT),
            required_scopes=["echo"],
        ),
    )

    @server.tool()
    async def echo(text: str) -> str:
        """Return the supplied text unchanged."""
        return text

    structlog.get_logger().info("spike_server_starting", host="127.0.0.1", port=9100)
    server.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=9100,
        streamable_http_path="/mcp",
        stateless_http=False,
    )


if __name__ == "__main__":
    main()
