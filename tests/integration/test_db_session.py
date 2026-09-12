"""Real asyncpg transactions, ownership, constraints and restart recovery."""

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import CheckConstraint, Connection, ForeignKey, MetaData, String, inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config_models import DatabaseSettings
from app.core.errors import (
    ConflictError,
    DatabaseError,
    DatabaseTimeoutError,
    UpstreamUnavailableError,
)
from app.db.base import NAMING_CONVENTION, Base, TimestampMixin, UserOwnedMixin
from app.db.session import Database
from app.repositories.base import UserScopedRepository
from scripts.deployment import compose_prefix, process_environment, run_command
from tests.database_support import DatabaseStack

pytestmark = pytest.mark.integration
CONFLICT = 409
UNAVAILABLE = 503


class FixtureBase(Base):
    """Separate test metadata: importing tests cannot populate application migrations."""

    __abstract__ = True
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Owner(FixtureBase):
    """Test-only parent for ownership foreign key reflection."""

    __tablename__ = "step12_owners"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)


class OwnedRow(UserOwnedMixin, TimestampMixin, FixtureBase):
    """A test-only user-owned row with named integrity constraints."""

    __tablename__ = "step12_owned_rows"
    __table_args__ = (CheckConstraint("length(value) > 0", name="value_nonempty"),)
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("step12_owners.id"), index=True)
    value: Mapped[str] = mapped_column(String(100), unique=True)


class OwnedRepository(UserScopedRepository[OwnedRow]):
    """Exercise the protected query entrypoint with actual PostgreSQL rows."""

    model = OwnedRow

    async def list_owned(self) -> list[OwnedRow]:
        """Return only the constructor's user rows."""
        return list((await self._session.scalars(self._scoped())).all())

    async def get_owned(self, row_id: UUID) -> OwnedRow | None:
        """Additional predicates must retain ownership even for a known foreign ID."""
        return await self._session.scalar(self._scoped().where(self.model.id == row_id))


@pytest.fixture
def database_settings(database_stack: DatabaseStack) -> DatabaseSettings:
    """Only application credentials enter the code under test."""
    return DatabaseSettings(
        port=database_stack.settings.db_host_port,
        app_password=database_stack.settings.bootstrap.app_password,
        pool_size=1,
        max_overflow=0,
        pool_timeout_s=0.05,
    )


@pytest.fixture
async def database(
    database_stack: DatabaseStack, database_settings: DatabaseSettings
) -> AsyncIterator[Database]:
    """Owner creates test tables; the runtime remains unable to perform DDL."""
    owner_settings = database_settings.model_copy(
        update={
            "app_user": "postgres",
            "app_password": database_stack.settings.postgres_superuser_password,
        }
    )
    owner_engine = create_async_engine(
        owner_settings.app_url,
        hide_parameters=True,
        connect_args={"timeout": 5, "command_timeout": 30},
    )
    try:
        async with owner_engine.begin() as connection:
            await connection.execute(text("SET LOCAL ROLE app_owner"))
            await connection.run_sync(FixtureBase.metadata.create_all)
    finally:
        await owner_engine.dispose()
    resource = Database(database_settings)
    resource.start()
    try:
        yield resource
    finally:
        await resource.aclose()


async def insert_row(session: AsyncSession) -> OwnedRow:
    """Create an isolated test user's row without owning the transaction."""
    owner = Owner()
    session.add(owner)
    await session.flush()
    row = OwnedRow(user_id=owner.id, value=uuid4().hex)
    session.add(row)
    await session.flush()
    return row


async def test_session_rollback_on_exception(database: Database) -> None:
    with pytest.raises(ConflictError):  # noqa: PT012 -- exercise an entire transaction failure and cleanup.
        async with database.session() as session:
            row = await insert_row(session)
            row_id = row.id
            raise ConflictError("synthetic service failure")
    async with database.session() as session:
        assert await session.get(OwnedRow, row_id) is None


async def test_uncommitted_success_rolls_back(database: Database) -> None:
    async with database.session() as session:
        row_id = (await insert_row(session)).id
    async with database.session() as session:
        assert await session.get(OwnedRow, row_id) is None


async def test_explicit_commit_and_timestamps(database: Database) -> None:
    async with database.session() as session:
        async with session.begin():
            row = await insert_row(session)
        original = row.created_at
        assert original.tzinfo is not None
        assert row.updated_at == original
        row_id = row.id  # expire_on_commit=False: no implicit I/O here.
    async with database.session() as session, session.begin():
        saved = await session.get(OwnedRow, row_id)
        assert saved is not None
        saved.value = uuid4().hex
        await session.flush()
        await session.refresh(saved)
        assert saved.created_at == original
        assert saved.updated_at > original
        assert saved.updated_at.tzinfo is not None


async def test_cancelled_session_rolls_back_and_returns_connection(database: Database) -> None:
    row_id = uuid4()
    with pytest.raises(TimeoutError):  # noqa: PT012 -- exercise an entire transaction failure and cleanup.
        async with asyncio.timeout(0.1), database.session() as session:
            row_id = (await insert_row(session)).id
            await asyncio.Event().wait()
    async with database.session() as session:
        assert await session.get(OwnedRow, row_id) is None
        assert await session.scalar(text("SELECT 1")) == 1


async def test_user_scoped_repository_filters_by_user(database: Database) -> None:
    async with database.session() as session, session.begin():
        first, second = await insert_row(session), await insert_row(session)
        repo = OwnedRepository(session, first.user_id)
        assert [row.id for row in await repo.list_owned()] == [first.id]
        assert await repo.get_owned(second.id) is None
        assert await repo.get_owned(first.id) is first
        assert await OwnedRepository(session, uuid4()).list_owned() == []
        assert [row.id for row in await OwnedRepository(session, second.user_id).list_owned()] == [
            second.id
        ]


async def test_naming_convention_applied(database: Database) -> None:
    async with database.engine.connect() as connection:

        def reflect(sync_connection: object) -> None:
            inspector = inspect(sync_connection)
            assert inspector is not None
            assert (
                inspector.get_pk_constraint("step12_owned_rows")["name"] == "pk_step12_owned_rows"
            )
            assert (
                inspector.get_foreign_keys("step12_owned_rows")[0]["name"]
                == "fk_step12_owned_rows_user_id_step12_owners"
            )
            assert (
                inspector.get_unique_constraints("step12_owned_rows")[0]["name"]
                == "uq_step12_owned_rows_value"
            )
            assert (
                inspector.get_check_constraints("step12_owned_rows")[0]["name"]
                == "ck_step12_owned_rows_value_nonempty"
            )
            assert "ix_step12_owned_rows_user_id" in {
                item["name"] for item in inspector.get_indexes("step12_owned_rows")
            }

        await connection.run_sync(reflect)
    assert not set(FixtureBase.metadata.tables) & set(Base.metadata.tables)


async def test_pool_exhaustion_maps_to_503_and_recovers(database: Database) -> None:
    async with database.session() as occupied:
        await occupied.execute(text("SELECT 1"))
        with pytest.raises(DatabaseTimeoutError) as error:
            async with database.session() as waiting:
                await waiting.execute(text("SELECT 1"))
        assert error.value.http_status == UNAVAILABLE
    async with database.session() as recovered:
        assert await recovered.scalar(text("SELECT 1")) == 1


async def test_unique_conflict_maps_to_409_and_rolls_back(database: Database) -> None:
    async with database.session() as session, session.begin():
        existing = await insert_row(session)
    with pytest.raises(ConflictError) as error:  # noqa: PT012 -- exercise an entire transaction failure and cleanup.
        async with database.session() as session, session.begin():
            new = await insert_row(session)
            new_id = new.id
            new.value = existing.value
            await session.flush()
    assert error.value.http_status == CONFLICT
    async with database.session() as session:
        assert await session.get(OwnedRow, new_id) is None
        assert await session.get(OwnedRow, existing.id) is not None


async def test_non_unique_integrity_error_is_not_conflict(database: Database) -> None:
    with pytest.raises(DatabaseError) as error:  # noqa: PT012 -- exercise an entire transaction failure and cleanup.
        async with database.session() as session:
            row = await insert_row(session)
            row.value = ""
            await session.flush()
    assert type(error.value) is DatabaseError


async def restart_postgres(stack: DatabaseStack) -> None:
    """Restart only the fixture's named PostgreSQL service, preserving its volume."""
    prefix = compose_prefix(stack.docker, stack.settings, stack.call)
    env = process_environment(stack.settings)
    await asyncio.to_thread(run_command, [*prefix, "restart", "postgres"], env, timeout=60)
    await asyncio.to_thread(
        run_command, [*prefix, "up", "-d", "--wait", "postgres"], env, timeout=60
    )


async def test_pool_pre_ping_survives_restart(
    database: Database, database_stack: DatabaseStack
) -> None:
    async with database.session() as session:
        assert await session.scalar(text("SELECT 1")) == 1
    await restart_postgres(database_stack)
    async with database.session() as session:
        assert await session.scalar(text("SELECT 1")) == 1


async def test_restart_during_transaction_fails_without_replay(
    database: Database, database_stack: DatabaseStack
) -> None:
    with pytest.raises(UpstreamUnavailableError):  # noqa: PT012 -- exercise an entire transaction failure and cleanup.
        async with database.session() as session, session.begin():
            row_id = (await insert_row(session)).id
            await restart_postgres(database_stack)
            await session.execute(text("SELECT 1"))
    async with database.session() as session:
        assert await session.get(OwnedRow, row_id) is None


@pytest.mark.parametrize("slow_startup", [False, True])
async def test_command_timeout_is_bounded(
    database: Database,
    database_settings: DatabaseSettings,
    monkeypatch: pytest.MonkeyPatch,
    slow_startup: bool,
) -> None:
    # 50 ms can expire during dialect initialization on an instrumented runner,
    # accidentally satisfying raises() before the intended query ever executes.
    short = Database(database_settings.model_copy(update={"command_timeout_s": 1}))
    short.start()
    if slow_startup:
        dialect = short.engine.sync_engine.dialect
        initialize = dialect.initialize

        def delayed_initialize(connection: Connection) -> None:
            connection.exec_driver_sql("SELECT pg_sleep(0.1)")
            initialize(connection)

        monkeypatch.setattr(dialect, "initialize", delayed_initialize)
    try:
        async with short.session() as session:
            assert await session.scalar(text("SELECT 1")) == 1
        with pytest.raises(DatabaseTimeoutError):
            async with short.session() as session:
                await session.execute(text("SELECT pg_sleep(5)"))
        async with short.session() as session:
            assert await session.scalar(text("SELECT 1")) == 1
    finally:
        await short.aclose()


async def test_runtime_has_no_ddl_privilege(database: Database) -> None:
    async with database.session() as session:
        assert await session.scalar(text("SELECT current_user")) == "app_rw"
        assert not await session.scalar(
            text("SELECT has_schema_privilege(current_user, 'public', 'CREATE')")
        )


async def test_separate_sessions_do_not_share_pending_state(database: Database) -> None:
    async with database.session() as first, database.session() as second:
        assert first is not second
        first.add(Owner())
        assert first.new
        assert not second.new


async def test_failed_authentication_is_typed_unavailable(
    database: Database, database_settings: DatabaseSettings
) -> None:
    invalid = Database(
        database_settings.model_copy(update={"app_password": SecretStr("wrong-test-password")})
    )
    invalid.start()
    try:
        with pytest.raises(UpstreamUnavailableError):
            async with invalid.session() as session:
                await session.execute(text("SELECT 1"))
    finally:
        await invalid.aclose()


async def test_commit_failure_is_translated(database: Database) -> None:
    async with database.session() as session, session.begin():
        existing = await insert_row(session)
    with pytest.raises(ConflictError):  # noqa: PT012 -- commit flushes pending duplicate data.
        async with database.session() as session:
            session.add(OwnedRow(user_id=existing.user_id, value=existing.value))
            await session.commit()
