"""Foreground SSH supervision with bounded reconnect and forwarded HTTP health."""

import argparse
import asyncio
import json
import signal
import urllib.request
from contextlib import suppress

SSH_COMMAND = (
    "ssh",
    "-F",
    "/run/ip-ssh/config",
    "-N",
    "-T",
    "-g",
    "-L",
    "0.0.0.0:8100:127.0.0.1:8100",
    "ip-model",
)
BACKOFF = (1, 2, 4, 8, 16, 30)
PROBE_INTERVAL = 10
FAILURE_LIMIT = 3


def healthy() -> bool:
    """Forwarded unauthenticated liveness detects a stuck SSH forwarding path."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=2) as response:
            return bool(json.load(response).get("status") == "ok")
    except (OSError, ValueError):
        return False


async def terminate(process: asyncio.subprocess.Process) -> None:
    """Forward termination, then kill only this owned SSH child if it does not exit."""
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()


async def monitor(process: asyncio.subprocess.Process, stop: asyncio.Event) -> bool:
    """Detect an unhealthy forwarded path; report whether the connection was healthy."""
    failures = 0
    observed_health = False
    while process.returncode is None and not stop.is_set():
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=PROBE_INTERVAL)
        if stop.is_set():
            break
        if await asyncio.to_thread(healthy):
            failures = 0
            observed_health = True
            continue
        failures += 1
        if failures >= FAILURE_LIMIT:
            break
    return observed_health


async def supervise(stop: asyncio.Event) -> None:
    """No shell, key mutation, trust-on-first-use or busy reconnect loop."""
    attempt = 0
    while not stop.is_set():
        process = await asyncio.create_subprocess_exec(
            *SSH_COMMAND,
            stdout=asyncio.subprocess.DEVNULL,
            # SSH diagnostics are visible; never pipe model requests or tokens to SSH.
            stderr=None,
        )
        try:
            if await monitor(process, stop):
                attempt = 0
        finally:
            await terminate(process)
        if stop.is_set():
            break
        delay = BACKOFF[min(attempt, len(BACKOFF) - 1)]
        print(
            json.dumps(
                {
                    "event": "model_tunnel_reconnect",
                    "delay_s": delay,
                    "exit_code": process.returncode,
                }
            ),
            flush=True,
        )
        attempt += 1
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=delay)


async def run() -> None:
    """Supervisor is PID-managed by Compose init and owns only one SSH child."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    await supervise(stop)


def main() -> None:
    """Start a bounded supervisor or run the image's forwarded healthcheck."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        raise SystemExit(0 if healthy() else 1)
    asyncio.run(run())


if __name__ == "__main__":
    main()
