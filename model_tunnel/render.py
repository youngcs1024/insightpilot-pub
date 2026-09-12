"""Render a selected pair of hosts; never copy or discover trust records."""

from model_tunnel.config import TunnelSettings


def render(config: TunnelSettings) -> str:
    """Inputs are validated as single SSH tokens before interpolation."""
    return f"""Host ip-jump
    HostName {config.jump_host}
    User {config.jump_user}
    Port {config.jump_port}
    IdentityFile /run/ip-ssh/jump_key

Host ip-model
    HostName {config.target_host}
    User {config.target_user}
    Port {config.target_port}
    IdentityFile /run/ip-ssh/target_key
    ProxyJump ip-jump

Host *
    BatchMode yes
    IdentitiesOnly yes
    StrictHostKeyChecking yes
    UserKnownHostsFile /run/ip-ssh/known_hosts
    GlobalKnownHostsFile /dev/null
    ConnectTimeout 10
    ConnectionAttempts 1
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ExitOnForwardFailure yes
    ForwardAgent no
"""
