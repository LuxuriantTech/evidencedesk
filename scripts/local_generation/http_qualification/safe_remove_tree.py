#!/usr/bin/env python3
"""Remove only one run-owned directory directly under RUNNER_TEMP."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    args = p.parse_args()
    temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    target = args.path.absolute()
    if target.is_symlink() or not target.is_dir():
        raise SystemExit("FAIL step=cleanup code=not_plain_directory")
    resolved = target.resolve(strict=True)
    if resolved.parent != temp or not re.fullmatch(
        r"evidencedesk-httpq-[A-Za-z0-9]+-[A-Za-z0-9]+", resolved.name
    ):
        raise SystemExit("FAIL step=cleanup code=path_scope_rejected")
    shutil.rmtree(resolved)
    print("PASS step=cleanup scoped_run_directory_removed=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
