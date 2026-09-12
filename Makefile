.DEFAULT_GOAL := help

# Resolve from this file, including when called with make -f from another directory.
override IP_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
override UV_PROJECT_ENVIRONMENT := $(IP_ROOT)/.venv
override UV_CACHE_DIR := $(IP_ROOT)/.uv-cache
export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR
unexport VIRTUAL_ENV PYTHONPATH PYTHONHOME UV_PROJECT UV_WORKING_DIRECTORY

ENV ?= development
CI_TYPES_OUTPUT ?= type-evidence
VALID_ENVS := development staging production test

# Explicit paths override inherited uv project/directory settings. Never --active.
override IP_UV := bash "$(IP_ROOT)/scripts/uv_project.sh" api
override IP_RUN := $(IP_UV) run --locked --no-env-file
override IP_RUFF_TARGETS := app model_tunnel model_runtime tests scripts spikes alembic data mcp_server evals pyproject.toml model_runtime/pyproject.toml

.PHONY: install install-mcp install-model-runtime compat dev lint lint-check format-check format-patch ruff-patch format typecheck typecheck-report test test-unit test-coverage migrate migration seed up down eval clean help check-env

check-env:
	@case "$(ENV)" in development|staging|production|test) ;; \
	*) printf '%s\n' 'Invalid ENV; expected: $(VALID_ENVS)' >&2; exit 2 ;; esac

install: check-env
	$(IP_UV) sync --locked

install-mcp: check-env
	bash "$(IP_ROOT)/scripts/uv_project.sh" mcp sync --locked

install-model-runtime: check-env
	bash "$(IP_ROOT)/scripts/uv_project.sh" model-runtime sync --locked

compat: check-env
	bash "$(IP_ROOT)/scripts/check_compat.sh"

lint: lint-check format-check

lint-check: check-env
	$(IP_RUN) ruff check $(IP_RUFF_TARGETS)

format-check: check-env
	$(IP_RUN) ruff format --check $(IP_RUFF_TARGETS)

format-patch: check-env
	$(IP_RUN) python -m scripts.ci_format_patch --tested-sha "$(CI_TESTED_SHA)" --output-dir "$(CI_FORMAT_OUTPUT)" --summary "$(CI_FORMAT_SUMMARY)" --run-url "$(CI_RUN_URL)" -- .venv/bin/ruff format --diff $(IP_RUFF_TARGETS)

ruff-patch: check-env
	$(IP_RUN) python -m scripts.ci_ruff_repair --tested-sha "$(CI_TESTED_SHA)" --output-dir "$(CI_FORMAT_OUTPUT)" --summary "$(CI_FORMAT_SUMMARY)" --run-url "$(CI_RUN_URL)" -- .venv/bin/ruff format --diff $(IP_RUFF_TARGETS)

format: check-env
	$(IP_RUN) ruff check --fix $(IP_RUFF_TARGETS)
	$(IP_RUN) ruff format $(IP_RUFF_TARGETS)

typecheck: check-env
	$(IP_RUN) mypy

typecheck-report: check-env
	$(IP_RUN) python -m scripts.ci_types --output-dir "$(CI_TYPES_OUTPUT)"

test: check-env
	$(IP_RUN) pytest -m "not gpu and not external"

test-coverage: check-env
	$(IP_RUN) pytest -m "not gpu and not external" --cov=app --cov-report=term-missing --cov-report="json:$(IP_ROOT)/.coverage.json"
	$(IP_RUN) python -m scripts.check_test_coverage "$(IP_ROOT)/.coverage.json"

test-unit: check-env
	$(IP_RUN) pytest tests/unit -m "not gpu and not external"

up: check-env
	bash "$(IP_ROOT)/scripts/compose.sh" --profile core up -d --wait

down: check-env
	bash "$(IP_ROOT)/scripts/compose.sh" --profile core down

dev: check-env
	IP_ENVIRONMENT="$(ENV)" $(IP_RUN) uvicorn app.main:app --host 127.0.0.1 --port 18000 --reload --timeout-graceful-shutdown 10

# Quote values as shell literals; messages may contain apostrophes, dollars and backticks.
ip_shell_quote = '$(subst ','"'"',$(1))'
db ?= app

migrate: check-env
	IP_ENVIRONMENT="$(ENV)" $(IP_RUN) python -m scripts.migrate_all upgrade

migration: check-env
	IP_ENVIRONMENT="$(ENV)" $(IP_RUN) python -m scripts.migrate_all revision --database $(call ip_shell_quote,$(db)) --message $(call ip_shell_quote,$(m))

seed: check-env
	$(IP_RUN) python -m data.seed.generate
	IP_ENVIRONMENT="$(ENV)" $(IP_RUN) python -m scripts.seed

s ?= nl2sql
eval: check-env
	IP_ENVIRONMENT="$(ENV)" $(IP_RUN) python -m evals.cli run --suite $(call ip_shell_quote,$(s))

clean: check-env
	@printf '%s\n' 'No files deleted. Cleanup requires user approval for each explicit file path; bulk cleanup is manual.'

help: check-env
	@printf '%s\n' \
	  'InsightPilot — Step 1.3' \
	  'install     Install locked API, development and test dependencies in root .venv' \
	  'install-mcp Install locked MCP dependencies in mcp_server/.venv' \
	  'install-model-runtime Install candidate model dependencies in model_runtime/.venv (CUDA acceptance separate)' \
	  'compat      Collect six independent dependency resolutions; no GPU validation' \
	  'lint        Manual diagnostic/CI: check lint and formatting (read-only)' \
	  'format      Apply lint fixes and formatting to project-owned code' \
	  'typecheck   Manual diagnostic/CI: strictly type-check app and project tools' \
	  'test        Manual acceptance: all project tests (GPU excluded by default)' \
	  'test-unit   Manual diagnostic: tests/unit only, not all container-free tests' \
	  'test-coverage Manual acceptance: all tests and independent 75% coverage gates' \
	  'clean       Explain manual cleanup; never delete' \
	  'help        Show this help' \
	  'Validation: no default local checklist; see docs/TESTING.md for blocker-only diagnostics' \
	  'up / down   Start isolated PostgreSQL / stop it while preserving its volume' \
	  'dev         Run API on loopback 18000 with reload and process-scoped configuration' \
	  'migrate     Upgrade both independent database histories' \
	  'migration m=… [db=app|business]  Generate one migration' \
	  'seed        Generate and atomically import business data with .env.seed' \
	  'eval s=nl2sql: explicit live evaluation, separate from ordinary CI' \
	  'ENV: development (default), staging, production, test; dev loads only API process config'
