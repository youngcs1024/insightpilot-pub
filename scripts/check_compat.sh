#!/usr/bin/env bash
# Evidence collection only: never sync environments or start containers here.
set -euo pipefail
IP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
for IP_NAME in $(compgen -v); do
    case "$IP_NAME" in UV_*|PYTHON*|VIRTUAL_ENV) unset "$IP_NAME" ;; esac
done
export UV_CACHE_DIR="$IP_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$IP_ROOT/.uv-cache/python"
export UV_PROJECT_ENVIRONMENT="$IP_ROOT/.venv"
if [[ ! -x "$IP_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' 'Run make install first to install the locked compatibility tool dependencies.' >&2
    exit 2
fi
exec "$IP_ROOT/.venv/bin/python" -I "$IP_ROOT/scripts/check_compat.py" "$@"
