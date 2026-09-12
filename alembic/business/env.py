"""Business migration entrypoint; data tables arrive in Step 2.1."""

from scripts.migration_environment import run_environment
from scripts.migration_settings import MigrationTarget

run_environment(MigrationTarget.BUSINESS)
