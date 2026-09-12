"""Authentication acceptance against migrated PostgreSQL and production routes."""

import asyncio
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

import httpx
import jwt
import pytest
from sqlalchemy import func, select, update

from app.application import create_app
from app.core.config_models import RateRule, Settings
from app.core.errors import AuthenticationError, DatabaseError
from app.core.logging import setup_logging
from app.core.security import TokenCodec
from app.db.models import RefreshToken, User
from app.db.session import Database
from app.repositories.refresh_token import RefreshTokenRepository
from app.schemas.auth import TokenClaims

pytestmark = pytest.mark.integration
PASSWORD = "Auth-test-password1!"  # noqa: S105 -- synthetic acceptance input
OK, CREATED, BAD, UNAUTHORIZED, CONFLICT, LIMITED = 200, 201, 400, 401, 409, 429


async def register(client: httpx.AsyncClient, email: str | None = None) -> httpx.Response:
    return await client.post(
        "/api/v1/auth/register",
        json={
            "email": email or f"{uuid4().hex}@example.com",
            "password": PASSWORD,
            "display_name": "Tester",
        },
    )


async def login(client: httpx.AsyncClient, email: str) -> httpx.Response:
    return await client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})


async def tokens(client: httpx.AsyncClient) -> httpx.Response:
    profile = await register(client)
    assert profile.status_code == CREATED, profile.text
    return await login(client, profile.json()["email"])


async def test_register_then_login(client: httpx.AsyncClient) -> None:
    profile = await register(client)
    assert profile.status_code == CREATED, profile.text
    assert "password" not in profile.text
    response = await login(client, profile.json()["email"])
    assert response.status_code == OK, response.text
    me = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {response.json()['access_token']}"}
    )
    assert me.status_code == OK
    assert me.json()["id"] == profile.json()["id"]
    assert me.json()["request_id"] == me.headers["X-Request-ID"]


async def test_login_wrong_password_401(client: httpx.AsyncClient) -> None:
    profile = await register(client)
    response = await client.post(
        "/api/v1/auth/login", json={"email": profile.json()["email"], "password": "wrong"}
    )
    unknown = await client.post(
        "/api/v1/auth/login", json={"email": "unknown@example.com", "password": "wrong"}
    )
    assert response.status_code == unknown.status_code == UNAUTHORIZED
    assert response.json()["message"] == unknown.json()["message"]


async def test_duplicate_email_409(unauthenticated_client: httpx.AsyncClient) -> None:
    email = f"{uuid4().hex}@example.com"
    responses = await asyncio.gather(
        register(unauthenticated_client, email), register(unauthenticated_client, email.upper())
    )
    assert sorted(r.status_code for r in responses) == [CREATED, CONFLICT]


async def test_expired_token_401(client: httpx.AsyncClient, settings: Settings) -> None:
    token, claims = TokenCodec(settings.security).create(uuid4(), "access")
    payload = claims.model_dump(mode="json")
    payload["exp"] = int((datetime.now(UTC) - timedelta(seconds=1)).timestamp())
    token = jwt.encode(payload, settings.security.jwt_secret.get_secret_value(), algorithm="HS256")
    response = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == UNAUTHORIZED
    assert response.json()["code"] == "token_expired"
    assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_wrong_token_type_rejected(client: httpx.AsyncClient) -> None:
    pair = (await tokens(client)).json()
    me = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {pair['refresh_token']}"}
    )
    refresh = await client.post(
        "/api/v1/auth/refresh", json={"refresh_token": pair["access_token"]}
    )
    assert me.status_code == refresh.status_code == UNAUTHORIZED


async def test_refresh_rotates_token(
    unauthenticated_client: httpx.AsyncClient, auth_database: Database, settings: Settings
) -> None:
    original = (await tokens(unauthenticated_client)).json()
    rotated = await unauthenticated_client.post(
        "/api/v1/auth/refresh", json={"refresh_token": original["refresh_token"]}
    )
    assert rotated.status_code == OK
    assert rotated.json()["refresh_token"] != original["refresh_token"]
    restarted = create_app(settings, database=auth_database)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=restarted), base_url="http://test"
    ) as client:
        replay = await client.post(
            "/api/v1/auth/refresh", json={"refresh_token": original["refresh_token"]}
        )
        valid = await client.post(
            "/api/v1/auth/refresh", json={"refresh_token": rotated.json()["refresh_token"]}
        )
    assert replay.status_code == UNAUTHORIZED
    assert valid.status_code == OK


async def test_concurrent_refresh_only_one_succeeds(
    unauthenticated_client: httpx.AsyncClient,
) -> None:
    pair = (await tokens(unauthenticated_client)).json()
    results = await asyncio.gather(
        *(
            unauthenticated_client.post(
                "/api/v1/auth/refresh", json={"refresh_token": pair["refresh_token"]}
            )
            for _ in range(2)
        )
    )
    assert sorted(r.status_code for r in results) == [OK, UNAUTHORIZED]


async def test_password_policy_enforced(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "weak@example.com", "password": "a", "display_name": "Weak"},
    )
    assert response.status_code == BAD
    for rule in ("8 characters", "uppercase", "digit", "symbol"):
        assert rule in response.json()["message"]
    assert "Password requires" in response.json()["message"]


async def test_rate_limit_returns_429_with_retry_after(
    client: httpx.AsyncClient,
) -> None:
    responses = [
        await client.post(
            "/api/v1/auth/register",
            json={"email": "weak@example.com", "password": "a", "display_name": "Weak"},
        )
        for _ in range(6)
    ]
    assert responses[-1].status_code == LIMITED
    assert int(responses[-1].headers["Retry-After"]) > 0
    assert responses[-1].json()["code"] == "RATE_LIMIT_EXCEEDED"


async def test_disabled_user_rejected(
    unauthenticated_client: httpx.AsyncClient, auth_database: Database, settings: Settings
) -> None:
    pair = (await tokens(unauthenticated_client)).json()
    claims = TokenCodec(settings.security).decode(pair["access_token"], "access")
    async with auth_database.session() as session, session.begin():
        await session.execute(update(User).where(User.id == claims.sub).values(is_active=False))
    me = await unauthenticated_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {pair['access_token']}"}
    )
    refresh = await unauthenticated_client.post(
        "/api/v1/auth/refresh", json={"refresh_token": pair["refresh_token"]}
    )
    assert me.status_code == refresh.status_code == UNAUTHORIZED


async def test_refresh_transaction_rolls_back(
    unauthenticated_client: httpx.AsyncClient,
    auth_database: Database,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = (await tokens(unauthenticated_client)).json()
    claims = TokenCodec(settings.security).decode(pair["refresh_token"], "refresh")

    async def fail(self: RefreshTokenRepository, claims: TokenClaims) -> None:
        raise DatabaseError()

    with monkeypatch.context() as patch:
        patch.setattr(RefreshTokenRepository, "add", fail)
        response = await unauthenticated_client.post(
            "/api/v1/auth/refresh", json={"refresh_token": pair["refresh_token"]}
        )
    assert response.status_code == httpx.codes.INTERNAL_SERVER_ERROR
    async with auth_database.session() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(RefreshToken)
            .where(RefreshToken.user_id == claims.sub, RefreshToken.used_at.is_not(None))
        )
    assert count == 0
    retry = await unauthenticated_client.post(
        "/api/v1/auth/refresh", json={"refresh_token": pair["refresh_token"]}
    )
    assert retry.status_code == OK


@pytest.mark.parametrize("credential", [None, "Bearer malformed", "Basic abc"])
async def test_missing_or_malformed_credentials(
    unauthenticated_client: httpx.AsyncClient, credential: str | None
) -> None:
    response = await unauthenticated_client.get(
        "/api/v1/auth/me", headers={"Authorization": credential} if credential else {}
    )
    assert response.status_code == UNAUTHORIZED
    assert set(response.json()) == {"code", "message", "request_id"}


async def test_user_quota_is_independent_of_ip(
    unauthenticated_client: httpx.AsyncClient, auth_database: Database, settings: Settings
) -> None:
    first, second = (
        (await tokens(unauthenticated_client)).json(),
        (await tokens(unauthenticated_client)).json(),
    )
    config = settings.model_copy(deep=True)
    config.rate_limits.me_rules = [RateRule(requests=1, seconds=60)]
    application = create_app(config, database=auth_database)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application, client=("10.0.0.1", 123)),
        base_url="http://test",
    ) as client:
        for pair in (first, second):
            result = await client.get(
                "/api/v1/auth/me", headers={"Authorization": f"Bearer {pair['access_token']}"}
            )
            assert result.status_code == OK
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application, client=("10.0.0.2", 123)),
        base_url="http://test",
    ) as client:
        result = await client.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {first['access_token']}"}
        )
        assert result.status_code == LIMITED


async def test_no_password_or_tokens_in_logs(
    unauthenticated_client: httpx.AsyncClient,
    settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging(settings)
    pair = (await tokens(unauthenticated_client)).json()
    await unauthenticated_client.post(
        "/api/v1/auth/refresh", json={"refresh_token": pair["refresh_token"]}
    )
    invalid = await unauthenticated_client.post(
        "/api/v1/auth/login", json={"email": "not-an-email", "password": PASSWORD}
    )
    logs = capsys.readouterr().out
    for secret in (PASSWORD, pair["access_token"], pair["refresh_token"]):
        assert secret not in logs
        assert secret not in invalid.text
    assert "user_registered" in logs
    assert "refresh_rotated" in logs


async def test_unissued_refresh_is_rejected(
    unauthenticated_client: httpx.AsyncClient, settings: Settings
) -> None:
    pair = (await tokens(unauthenticated_client)).json()
    codec = TokenCodec(settings.security)
    user = codec.decode(pair["access_token"], "access").sub
    unissued, _ = codec.create(user, "refresh")
    response = await unauthenticated_client.post(
        "/api/v1/auth/refresh", json={"refresh_token": unissued}
    )
    assert response.status_code == UNAUTHORIZED


async def test_missing_user_rejected(client: httpx.AsyncClient, settings: Settings) -> None:
    token, _ = TokenCodec(settings.security).create(uuid4(), "access")
    response = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == UNAUTHORIZED


async def test_refresh_identifiers_are_hashed_and_user_scoped(
    unauthenticated_client: httpx.AsyncClient, auth_database: Database, settings: Settings
) -> None:
    first, second = (
        (await tokens(unauthenticated_client)).json(),
        (await tokens(unauthenticated_client)).json(),
    )
    codec = TokenCodec(settings.security)
    claims = codec.decode(first["refresh_token"], "refresh")
    other = codec.decode(second["refresh_token"], "refresh").sub
    async with auth_database.session() as session, session.begin():
        assert not await RefreshTokenRepository(session, other).consume(claims)
        stored = await session.scalar(
            select(RefreshToken).where(RefreshToken.user_id == claims.sub)
        )
        assert stored is not None
        assert stored.jti_digest == sha256(claims.jti.encode()).hexdigest()
        assert stored.used_at is None
        with pytest.raises(AuthenticationError):
            await RefreshTokenRepository(session, other).add(claims)


async def test_invalid_me_credentials_are_rate_limited(
    auth_database: Database, settings: Settings
) -> None:
    config = settings.model_copy(deep=True)
    config.rate_limits.me_rules = [RateRule(requests=1, seconds=60)]
    application = create_app(config, database=auth_database)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        first = await client.get("/api/v1/auth/me")
        second = await client.get("/api/v1/auth/me", headers={"X-Forwarded-For": "10.0.0.99"})
    assert first.status_code == UNAUTHORIZED
    assert second.status_code == LIMITED
