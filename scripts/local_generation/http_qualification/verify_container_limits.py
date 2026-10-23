#!/usr/bin/env python3
"""Read-only Docker inspect checks for this run's resource caps."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess

EXPECTED = {
    "local-8b": (10 * 1024**3, 10 * 1024**3, 4_000_000_000),
    "api": (896 * 1024**2, 896 * 1024**2, None),
    "worker": (384 * 1024**2, 384 * 1024**2, None),
    "postgres": (512 * 1024**2, 512 * 1024**2, None),
    "redis": (128 * 1024**2, 128 * 1024**2, None),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--compose-project", required=True)
    p.add_argument("--compose-files", nargs="+", required=True)
    args = p.parse_args()
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("FAIL step=limits code=docker_missing")
    prefix = (
        f"evidencedesk-httpq-{os.environ.get('GITHUB_RUN_ID', 'local')}"
        f"-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    )
    if args.compose_project != prefix:
        raise SystemExit("FAIL step=limits code=project_mismatch")
    cmd = [docker, "compose", "--project-name", args.compose_project]
    for path in args.compose_files:
        cmd.extend(["--file", path])
    cmd.extend(["ps", "--all", "--format", "json"])
    raw = subprocess.check_output(  # noqa: S603 - fixed docker CLI; project is validated; no shell.
        cmd, text=True, stderr=subprocess.DEVNULL
    )
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    by_service = {str(r.get("Service")): r for r in rows}
    for service, (memory, swap, nano_cpus) in EXPECTED.items():
        row = by_service.get(service)
        if not row or not row.get("ID"):
            raise SystemExit(f"FAIL step=limits code=missing_{service}")
        container_id = str(row["ID"])
        if not re.fullmatch(r"[a-fA-F0-9]{12,64}", container_id):
            raise SystemExit(f"FAIL step=limits code=invalid_id_{service}")
        inspect = json.loads(
            subprocess.check_output(  # noqa: S603 - fixed Docker executable; container ID is validated; no shell.
                [docker, "inspect", container_id],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        )[0]
        hc = inspect.get("HostConfig", {})
        if hc.get("Memory") != memory or hc.get("MemorySwap") != swap:
            raise SystemExit(f"FAIL step=limits code=memory_{service}")
        if nano_cpus is not None and hc.get("NanoCpus") != nano_cpus:
            raise SystemExit("FAIL step=limits code=cpu_local_8b")
    print("PASS step=limits service_caps_verified=true host_guard=12GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
