"""JWT validation, UTF-8 password boundaries and quota behavior without I/O."""

import threading
from pathlib import Path
from uuid import uuid4

import bcrypt
import jwt
import pytest
from fastapi import Request
from pydantic import SecretStr

from app.core.config_models import RateLimitSettings, RateRule, SecuritySettings
from app.core.errors import AuthenticationError, PasswordPolicyError, QuotaExceededError
from app.core.limiter import AuthLimiter
from app.core.security import (
    MAX_PASSWORD_BYTES,
    TokenCodec,
    hash_password,
    password_failures,
    verify_password,
)


@pytest.fixture
def codec() -> TokenCodec:
    return TokenCodec(
        SecuritySettings(
            jwt_secret=SecretStr("unit-test-signing-key-at-least-48-characters-long-for-tests"),
            bcrypt_rounds=4,
        )
    )


@pytest.mark.parametrize("claim", ["sub", "typ", "exp", "iat", "jti"])
def test_missing_claim_rejected(codec: TokenCodec, claim: str) -> None:
    _, claims = codec.create(uuid4(), "access")
    payload = claims.model_dump(mode="json")
    payload.pop(claim)
    token = jwt.encode(payload, codec.settings.jwt_secret.get_secret_value(), algorithm="HS256")
    with pytest.raises(AuthenticationError):
        codec.decode(token, "access")


@pytest.mark.parametrize("value", ["bad-uuid", "", 123])
def test_invalid_subject_rejected(codec: TokenCodec, value: str | int) -> None:
    _, claims = codec.create(uuid4(), "access")
    payload = claims.model_dump(mode="json")
    payload["sub"] = value
    token = jwt.encode(payload, codec.settings.jwt_secret.get_secret_value(), algorithm="HS256")
    with pytest.raises(AuthenticationError):
        codec.decode(token, "access")


def test_signature_and_algorithm_rejected(codec: TokenCodec) -> None:
    _, claims = codec.create(uuid4(), "access")
    for key, algorithm in (
        ("different-signing-key-32-characters", "HS256"),
        (codec.settings.jwt_secret.get_secret_value(), "HS384"),
    ):
        token = jwt.encode(claims.model_dump(mode="json"), key, algorithm=algorithm)
        with pytest.raises(AuthenticationError):
            codec.decode(token, "access")


def test_random_identifiers_and_ttls(codec: TokenCodec) -> None:
    user_id = uuid4()
    token, first = codec.create(user_id, "access")
    _, second = codec.create(user_id, "access")
    _, refresh = codec.create(user_id, "refresh")
    assert first.jti != second.jti
    assert first.exp - first.iat == 30 * 60
    assert refresh.exp - refresh.iat == 14 * 86400
    assert codec.decode(token, "access").sub == user_id


async def test_password_utf8_boundary() -> None:
    accepted = "Aa1!" + "中" * 22 + "ab"
    assert len(accepted.encode()) == MAX_PASSWORD_BYTES
    hashed = await hash_password(accepted, 4)
    assert await verify_password(accepted, hashed)
    assert not await verify_password(accepted + "中", hashed)
    assert not await verify_password("wrong", hashed)
    assert not await verify_password("wrong", "invalid-hash")
    with pytest.raises(PasswordPolicyError):
        await hash_password(accepted + "中", 4)
    assert set(password_failures("")) == {
        "at least 8 characters",
        "an uppercase letter",
        "a lowercase letter",
        "a digit",
        "a symbol",
    }


def test_every_quota_applies_and_apps_are_isolated() -> None:
    settings = RateLimitSettings(
        me_rules=[RateRule(requests=10, seconds=60), RateRule(requests=1, seconds=3600)]
    )
    limiter = AuthLimiter(settings)
    request = Request({"type": "http", "client": ("10.0.0.1", 123)})
    user = uuid4()
    limiter.check("me", request, user)
    with pytest.raises(QuotaExceededError) as error:
        limiter.check("me", request, user)
    assert error.value.retry_after > 0
    AuthLimiter(settings).check("me", request, user)


@pytest.mark.parametrize("claim", ["iat", "exp"])
@pytest.mark.parametrize("value", [[], {}, True, "123"])
def test_invalid_numeric_claim_rejected(codec: TokenCodec, claim: str, value: object) -> None:
    _, claims = codec.create(uuid4(), "access")
    payload = claims.model_dump(mode="json")
    payload[claim] = value
    token = jwt.encode(payload, codec.settings.jwt_secret.get_secret_value(), algorithm="HS256")
    with pytest.raises(AuthenticationError):
        codec.decode(token, "access")


def test_ambient_limiter_config_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("RATELIMIT_STORAGE_URL=redis://unreachable.invalid\n")
    monkeypatch.setenv("RATELIMIT_STORAGE_OPTIONS", "not-a-dictionary")
    monkeypatch.setenv("RATELIMIT_ENABLED", "false")
    limiter = AuthLimiter(RateLimitSettings(me_rules=[RateRule(requests=1, seconds=60)]))
    request = Request({"type": "http", "client": ("10.0.0.1", 123)})
    limiter.check("me", request)
    with pytest.raises(QuotaExceededError):
        limiter.check("me", request)


async def test_password_work_runs_outside_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    caller = threading.get_ident()
    worker_threads: list[int] = []
    original_hash, original_check = bcrypt.hashpw, bcrypt.checkpw

    def hashing(password: bytes, salt: bytes) -> bytes:
        worker_threads.append(threading.get_ident())
        return original_hash(password, salt)

    def checking(password: bytes, hashed: bytes) -> bool:
        worker_threads.append(threading.get_ident())
        return original_check(password, hashed)

    monkeypatch.setattr(bcrypt, "hashpw", hashing)
    monkeypatch.setattr(bcrypt, "checkpw", checking)
    hashed = await hash_password("Synthetic1!", 4)
    assert await verify_password("Synthetic1!", hashed)
    assert worker_threads
    assert all(worker != caller for worker in worker_threads)
