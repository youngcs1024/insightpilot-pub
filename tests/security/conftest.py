"""Security tests use the shared isolated PostgreSQL fixture from tests/conftest.py."""

from tests.api.conftest import isolated_api_environment

__all__ = ["isolated_api_environment"]
