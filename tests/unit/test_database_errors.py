"""Error translation uses driver types and SQLSTATE rather than diagnostic strings."""

import pytest
from asyncpg import InvalidPasswordError, PostgresError
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from app.core.errors import (
    ConflictError,
    DatabaseError,
    DatabaseTimeoutError,
    UpstreamUnavailableError,
)
from app.db.session import translate_database_error


class DriverError(Exception):
    """Minimal structured driver failure."""

    def __init__(self, sqlstate: str) -> None:
        super().__init__("timeout unique password failed after")
        self.sqlstate = sqlstate


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (IntegrityError("", None, DriverError("23505")), ConflictError),
        (IntegrityError("", None, DriverError("23503")), DatabaseError),
        (IntegrityError("", None, DriverError("23514")), DatabaseError),
        (DBAPIError("", None, DriverError("57014")), DatabaseTimeoutError),
        (
            DBAPIError("", None, DriverError("08006"), connection_invalidated=True),
            UpstreamUnavailableError,
        ),
        (OperationalError("", None, DriverError("08001")), UpstreamUnavailableError),
        (DBAPIError("", None, DriverError("28P01")), UpstreamUnavailableError),
        (InvalidPasswordError("private driver failure"), UpstreamUnavailableError),
        (DBAPIError("", None, DriverError("57P03")), UpstreamUnavailableError),
        (PoolTimeoutError("unrelated prose"), DatabaseTimeoutError),
        (TimeoutError("unrelated prose"), DatabaseTimeoutError),
        (ConnectionRefusedError("unrelated prose"), UpstreamUnavailableError),
        (SQLAlchemyError("unique constraint timeout"), DatabaseError),
    ],
)
def test_database_error_classification(
    error: SQLAlchemyError | PostgresError | OSError | TimeoutError, expected: type[DatabaseError]
) -> None:
    assert type(translate_database_error(error)) is expected
