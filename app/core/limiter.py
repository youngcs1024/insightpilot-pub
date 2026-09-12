"""App-local SlowAPI memory quotas applied after identity validation."""

from math import ceil
from pathlib import Path
from time import time
from typing import Literal
from uuid import UUID

from fastapi import Request
from limits import RateLimitItemPerSecond
from slowapi import Limiter

from app.core.config_models import RateLimitSettings
from app.core.errors import QuotaExceededError

AuthRoute = Literal["register", "login", "refresh", "me", "conversations", "messages"]


class ConfiguredLimiter(Limiter):
    """Disable SlowAPI's untyped ambient RATELIMIT_* configuration source."""

    def get_app_config[T](self, key: str, default_value: T = None) -> T:  # type: ignore[assignment]  # Match SlowAPI's generic optional default.
        """Use only explicit constructor settings and library defaults."""
        return default_value


class AuthLimiter:
    """Use SlowAPI's public strategy to avoid decorator/dependency ordering hazards."""

    def __init__(self, settings: RateLimitSettings) -> None:
        self.settings = settings
        self.backend = ConfiguredLimiter(
            key_func=lambda: "unused",
            storage_uri="memory://",
            strategy="fixed-window",
            config_filename=str(Path(__file__).with_name("limiter.env")),
            storage_options={},
        )

    def check(self, route: AuthRoute, request: Request, user_id: UUID | None = None) -> None:
        """Consume every applicable window using only a validated user identity."""
        key = (
            f"user:{user_id}"
            if user_id
            else f"ip:{request.client.host if request.client else 'unknown'}"
        )
        delays: list[int] = []
        rules = {
            "register": self.settings.register_rules,
            "login": self.settings.login_rules,
            "refresh": self.settings.refresh_rules,
            "me": self.settings.me_rules,
            "conversations": self.settings.conversations_rules,
            "messages": self.settings.messages_rules,
        }
        for rule in rules[route]:
            limit = RateLimitItemPerSecond(rule.requests, rule.seconds)
            if not self.backend.limiter.hit(limit, route, key):
                reset, _ = self.backend.limiter.get_window_stats(limit, route, key)
                delays.append(max(1, ceil(reset - time())))
        if delays:
            raise QuotaExceededError(max(delays))
