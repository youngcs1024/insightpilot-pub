"""Conservative UTF-8 byte budgets, an upper bound for byte-based LLM tokenizers."""

from app.core.errors import ContextBudgetExceeded

HISTORY_TOKENS = 1500
SUMMARY_TOKENS = 500
DATA_TOKENS = 2000


def token_bound(value: str) -> int:
    """Count one token per UTF-8 byte; never underestimate mixed Chinese/SQL text."""
    return len(value.encode("utf-8"))


def require_budget(value: str, budget: int) -> str:
    """Fail explicitly when required content cannot fit the conservative bound."""
    if token_bound(value) > budget:
        raise ContextBudgetExceeded()
    return value


def bounded_text(value: str, budget: int) -> str:
    """Trim optional history at a UTF-8 boundary, retaining an explicit marker."""
    if token_bound(value) <= budget:
        return value
    marker = " [truncated]"
    return value.encode("utf-8")[: budget - len(marker)].decode("utf-8", errors="ignore") + marker
