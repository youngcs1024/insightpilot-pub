"""Uvicorn entrypoint: required process settings validate before serving requests."""

from app.application import create_app
from app.core.config import settings

app = create_app(settings)
