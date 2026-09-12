"""Lightweight process configuration support; importing this never creates API settings."""

import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, ClassVar, Self, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import DotEnvSettingsSource

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# A non-file sentinel lets _env_file=None explicitly disable automatic dotenv discovery.
AUTO_ENV_FILE = Path("__insightpilot_process_env_auto__")
Secret = Annotated[SecretStr, Field(min_length=1)]


class Environment(StrEnum):
    """Supported deployment environments; misspellings are configuration errors."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class ConfigModel(BaseModel):
    """Reject unknown nested fields and hide raw input in validation messages."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


def environment_keys(model: type[BaseModel], prefix: str = "IP_") -> set[str]:
    """Derive accepted environment paths, including optional nested models."""
    keys: set[str] = set()
    for name, field in model.model_fields.items():
        key = f"{prefix}{name.upper()}"
        keys.add(key)
        # Collections are supplied as JSON; their item fields are not environment paths.
        annotations = (
            () if get_origin(field.annotation) in (list, dict) else get_args(field.annotation)
        )
        for annotation in (field.annotation, *annotations):
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                keys.update(environment_keys(annotation, f"{key}__"))
    return keys


def check_process_keys(model: type[BaseModel]) -> None:
    """Reject foreign/unknown IP settings without copying any values into errors."""
    allowed = environment_keys(model)
    errors: list[InitErrorDetails] = [
        {"type": "extra_forbidden", "loc": (key,), "input": None}
        for key in sorted(os.environ)
        if key.upper().startswith("IP_") and key.upper() not in allowed
    ]
    if errors:
        raise ValidationError.from_exception_data(model.__name__, errors, hide_input=True)


def resolve_process_env_file(root: Path, process: str, environment: Environment) -> Path | None:
    """Choose the first existing process file; never search the working directory."""
    names = (
        f".env.{process}.{environment.value}.local",
        f".env.{process}.{environment.value}",
        f".env.{process}.local",
        f".env.{process}",
    )
    return next((root / name for name in names if (root / name).is_file()), None)


class ProcessSettings(BaseSettings):
    """Common source policy for independently loaded, credential-scoped processes."""

    model_config = SettingsConfigDict(
        env_prefix="IP_",
        env_nested_delimiter="__",
        env_file=AUTO_ENV_FILE,
        env_file_encoding="utf-8",
        extra="forbid",
        hide_input_in_errors=True,
    )
    process_name: ClassVar[str]
    project_root: ClassVar[Path] = PROJECT_ROOT
    environment: Environment = Environment.DEVELOPMENT

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Keep explicit constructor/environment values ahead of one dotenv source."""
        check_process_keys(cls)
        # Pydantic's source hook is an untyped third-party adapter, not an app boundary.
        initial = init_settings()
        environment_input = initial.get("environment", Environment.DEVELOPMENT)
        if "environment" not in initial:
            environment_input = next(
                (v for k, v in os.environ.items() if k.upper() == "IP_ENVIRONMENT"),
                Environment.DEVELOPMENT,
            )
        environment = TypeAdapter(
            Environment, config=ConfigDict(hide_input_in_errors=True)
        ).validate_python(environment_input)
        if (
            isinstance(dotenv_settings, DotEnvSettingsSource)
            and dotenv_settings.env_file == AUTO_ENV_FILE
        ):
            dotenv_settings = DotEnvSettingsSource(
                settings_cls,
                env_file=resolve_process_env_file(cls.project_root, cls.process_name, environment),
            )
        return init_settings, env_settings, dotenv_settings

    @classmethod
    def load(cls) -> Self:
        """Construct a process configuration using its declared sources."""
        return cls()


def require_configuration(condition: bool, message: str) -> None:
    """Report a cross-field validation error without echoing supplied values."""
    if not condition:
        raise PydanticCustomError("configuration_invariant", message)
