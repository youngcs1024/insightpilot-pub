"""Real PostgreSQL tests of transaction and role guarantees of shared fixtures."""

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User
from app.db.session import Database
from tests import factories
from tests.shared_database import TestPostgres, rollback_connection

pytestmark = pytest.mark.integration


async def test_shared_session_uses_application_role(db_session: AsyncSession) -> None:
    assert await db_session.scalar(text("SELECT current_user")) == "app_rw"
    assert not await db_session.scalar(
        text("SELECT has_database_privilege(current_user, 'insightpilot_business', 'CONNECT')")
    )
    user = factories.user()
    db_session.add(user)
    await db_session.commit()
    assert await db_session.get(User, user.id) is user


@pytest.mark.parametrize("fail", [False, True])
async def test_outer_rollback_survives_commit_and_failed_flush(
    migrated_db: TestPostgres, fail: bool
) -> None:
    user = factories.user()
    email = user.email
    async with (
        rollback_connection(migrated_db.app) as connection,
        AsyncSession(bind=connection, join_transaction_mode="create_savepoint") as session,
    ):
        session.add(user)
        await session.commit()
        if fail:
            session.add(factories.user(email=email))
            with pytest.raises(IntegrityError):
                await session.flush()
            await session.rollback()
        assert await session.scalar(select(User.id).where(User.email == email)) is not None
    database = Database(migrated_db.app)
    database.start()
    try:
        async with database.session() as session:
            assert await session.scalar(select(User.id).where(User.email == email)) is None
    finally:
        await database.aclose()


async def test_bootstrap_provides_all_roles_and_databases(db_session: AsyncSession) -> None:
    roles = set(
        (
            await db_session.scalars(
                text(
                    "SELECT rolname FROM pg_roles WHERE rolname IN ('app_owner', 'biz_owner', 'app_rw', 'etl_rw', 'mcp_ro')"
                )
            )
        ).all()
    )
    assert roles == {"app_owner", "biz_owner", "app_rw", "etl_rw", "mcp_ro"}
    databases = set(
        (
            await db_session.scalars(
                text(
                    "SELECT datname FROM pg_database WHERE datname IN ('insightpilot_app', 'insightpilot_business')"
                )
            )
        ).all()
    )
    assert databases == {"insightpilot_app", "insightpilot_business"}
    assert (
        await db_session.scalar(text("SELECT version_num FROM alembic_version_app"))
        == "0010_knowledge_evidence"
    )
