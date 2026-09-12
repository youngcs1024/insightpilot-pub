#!/usr/bin/env bash
set -euo pipefail
IP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec bash "$IP_ROOT/scripts/uv_project.sh" api run --locked --no-env-file python -m scripts.model_deployment "$@"
