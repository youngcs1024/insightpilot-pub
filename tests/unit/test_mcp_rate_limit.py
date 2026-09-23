"""MCP quotas count tool attempts independently for each authenticated principal."""

from mcp_server.rate_limit import TOOL_CALL_LIMIT, ToolCallLimiter


def test_sixty_attempts_admitted_and_rejections_still_count() -> None:
    now = [0.0]
    limiter = ToolCallLimiter(clock=lambda: now[0])
    assert all(limiter.admit("insightpilot-api") for _ in range(TOOL_CALL_LIMIT))
    now[0] = 1.0
    assert not limiter.admit("insightpilot-api")
    now[0] = 60.0
    assert all(limiter.admit("insightpilot-api") for _ in range(TOOL_CALL_LIMIT - 1))
    assert not limiter.admit("insightpilot-api")
    now[0] = 120.0
    assert limiter.admit("insightpilot-api")


def test_principals_have_independent_windows() -> None:
    limiter = ToolCallLimiter(clock=lambda: 0.0)
    assert all(limiter.admit("first") for _ in range(TOOL_CALL_LIMIT))
    assert not limiter.admit("first")
    assert limiter.admit("second")
