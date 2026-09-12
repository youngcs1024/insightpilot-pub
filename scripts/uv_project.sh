#!/usr/bin/env bash
# Select one independent environment; never inherit another project's selectors.
set -euo pipefail
IP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
case "${1:-}" in
    api) IP_PROJECT="$IP_ROOT" ;;
    mcp) IP_PROJECT="$IP_ROOT/mcp_server" ;;
    model-runtime) IP_PROJECT="$IP_ROOT/model_runtime" ;;
    *) printf '%s\n' 'Expected service: api, mcp, or model-runtime' >&2; exit 2 ;;
esac
shift
for IP_NAME in $(compgen -v); do
    case "$IP_NAME" in UV_*|PYTHON*|PIP_*|VIRTUAL_ENV) unset "$IP_NAME" ;; esac
done
export UV_CACHE_DIR="$IP_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$IP_ROOT/.uv-cache/python"
export UV_PROJECT_ENVIRONMENT="$IP_PROJECT/.venv"
export UV_HTTP_TIMEOUT=30 UV_HTTP_RETRIES=2
exec uv --directory "$IP_PROJECT" --project "$IP_PROJECT" "$@"
