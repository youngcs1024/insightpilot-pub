#!/usr/bin/env bash
# Never install dblink/postgres_fdw: cross-database access defeats the MCP boundary.
# Reapply grants in place. Never repair bootstrap by deleting a volume.
set -euo pipefail
IP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec bash "$IP_ROOT/scripts/compose.sh" "$@" bootstrap
