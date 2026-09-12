"""Provide OpenSSH a passwd entry for the selected read-only key owner's UID."""

import os
import sys
from tempfile import NamedTemporaryFile


def main() -> None:
    """No root/chown/key copying: synthesize NSS records inside this container only."""
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise SystemExit("The model tunnel must run as a non-root user")
    with NamedTemporaryFile(mode="w", prefix="ip-passwd-", delete=False) as passwd:
        passwd.write(f"appuser:x:{uid}:{gid}:InsightPilot:/tmp:/bin/sh\n")
    with NamedTemporaryFile(mode="w", prefix="ip-group-", delete=False) as group:
        group.write(f"appuser:x:{gid}:\n")
    env = dict(os.environ)
    env.update(
        {
            "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libnss_wrapper.so",
            "NSS_WRAPPER_PASSWD": passwd.name,
            "NSS_WRAPPER_GROUP": group.name,
        }
    )
    if not sys.argv[1:]:
        raise SystemExit("A container command is required")
    os.execvpe(sys.argv[1], sys.argv[1:], env)  # noqa: S606 -- explicit Docker container command.


if __name__ == "__main__":
    main()
