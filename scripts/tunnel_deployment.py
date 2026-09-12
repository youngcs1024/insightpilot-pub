"""Prepare selected read-only SSH inputs for the local model tunnel."""

import json
import os
from pathlib import Path

from model_tunnel.config import TunnelProcessSettings
from model_tunnel.render import render
from scripts.deployment import DeploymentError
from scripts.model_diagnostics_settings import ModelDiagnosticsSettings


def prepare(root: Path, *, write_config: bool) -> dict[str, str]:
    """Only explicit retrieval startup reads SSH files or creates a generated config."""
    tunnel = TunnelProcessSettings(_env_file=root / ".env.tunnel").tunnel
    client = ModelDiagnosticsSettings(_env_file=root / ".env.model-diagnostics").model_runtime
    owners = {path.stat().st_uid for path in (tunnel.jump_key_path, tunnel.target_key_path)}
    if len(owners) != 1 or 0 in owners:
        raise DeploymentError("Tunnel keys must share an explicit non-root owner")
    if any(path.stat().st_mode & 0o077 for path in (tunnel.jump_key_path, tunnel.target_key_path)):
        raise DeploymentError("Tunnel private keys must not grant group/other access")
    if tunnel.config_path.resolve().is_relative_to(root.resolve()) is False:
        raise DeploymentError("Generated tunnel configuration must remain project-local")
    if write_config:
        tunnel.config_path.parent.mkdir(parents=True, exist_ok=True)
        # This generated configuration contains hostnames only, never private key bytes.
        tunnel.config_path.write_text(render(tunnel))
        tunnel.config_path.chmod(0o644)
    values = client.model_dump(mode="json")
    values["auth_token"] = client.auth_token.get_secret_value()
    env = {
        "IP_MODEL_RUNTIME": json.dumps(values),
        "IP_TUNNEL_UID": str(next(iter(owners))),
        "IP_TUNNEL_GID": str(tunnel.target_key_path.stat().st_gid),
    }
    for key in ("config_path", "jump_key_path", "target_key_path", "known_hosts_path", "host_port"):
        env["IP_TUNNEL__" + key.upper()] = str(getattr(tunnel, key))
    # The process uid is deliberately not changed; only this container uses the key owner.
    if os.getuid() == 0:
        raise DeploymentError("Prepare tunnel configuration as a non-root project operator")
    return env
