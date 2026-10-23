#!/usr/bin/env python3
"""Monitor runner host memory and stop only the uniquely named smoke Compose project."""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_RE = re.compile(r"^evidencedesk-httpq-[0-9]+-[0-9]+$")
SERVICES = ("api", "local-8b", "worker", "postgres", "redis")
_STOP = False


def memory_bytes() -> tuple[int, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        raise RuntimeError("meminfo_unavailable") from None
    if set(values) != {"MemTotal", "MemAvailable"}:
        raise RuntimeError("meminfo_incomplete")
    return values["MemTotal"], values["MemAvailable"]


def stop_project(project: str, compose_files: list[Path]) -> bool:
    if not PROJECT_RE.fullmatch(project):
        raise RuntimeError("unsafe_project_name")
    argv = ["docker", "compose", "--project-name", project]
    for file in compose_files:
        argv.extend(("--file", str(file.resolve(strict=True))))
    argv.extend(("stop", *SERVICES))
    try:
        result = subprocess.run(  # noqa: S603 - fixed docker command; project is validated; no shell.
            argv, capture_output=True, text=True, timeout=45, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-project", required=True)
    parser.add_argument("--compose-file", action="append", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--limit-gib", type=float, default=12.0)
    parser.add_argument("--interval-seconds", type=float, default=0.25)
    args = parser.parse_args()
    if not PROJECT_RE.fullmatch(args.compose_project):
        print("FAIL step=resource_guard code=unsafe_project_name", file=sys.stderr)
        return 2
    if args.limit_gib <= 0 or args.interval_seconds < 0.1:
        print("FAIL step=resource_guard code=invalid_limit", file=sys.stderr)
        return 2
    try:
        files = [p.resolve(strict=True) for p in args.compose_file]
        for file in files:
            if not file.is_file():
                raise RuntimeError("compose_file_invalid")
    except (OSError, RuntimeError):
        print("FAIL step=resource_guard code=compose_file_invalid", file=sys.stderr)
        return 2

    limit = int(args.limit_gib * 1024**3)
    high_water = 0
    polls = 0
    outcome = "stopped_cleanly"
    stop_error = False

    def request_stop(_signum: int, _frame: object) -> None:
        global _STOP
        _STOP = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while not _STOP:
            total, available = memory_bytes()
            used = total - available
            high_water = max(high_water, used)
            polls += 1
            if used >= limit:
                outcome = "host_memory_limit_reached"
                stop_error = not stop_project(args.compose_project, files)
                break
            time.sleep(args.interval_seconds)
    except RuntimeError as exc:
        outcome = exc.args[0] if exc.args else "resource_read_error"
        stop_error = not stop_project(args.compose_project, files)

    receipt = {
        "schema_version": 1,
        "project": args.compose_project,
        "limit_bytes": limit,
        "host_high_water_used_bytes": high_water,
        "polls": polls,
        "outcome": outcome,
        "compose_stop_failed": stop_error,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.receipt)
    if outcome != "stopped_cleanly" or stop_error:
        print(f"FAIL step=resource_guard code={outcome}", file=sys.stderr)
        return 1
    print(f"PASS check=host_memory_guard high_water_bytes={high_water}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
