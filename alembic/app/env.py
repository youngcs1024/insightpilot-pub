"""Application migration entrypoint."""

from scripts.migration_environment import run_environment
from scripts.migration_settings import MigrationTarget

run_environment(MigrationTarget.APP)
