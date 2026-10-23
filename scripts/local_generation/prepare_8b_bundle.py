#!/usr/bin/env python3
"""Explicitly prepare a local-only, hash-pinned Qwen3 8B asset bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bundle import BundleError, prepare_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new, absent output directory")
    args = parser.parse_args()
    try:
        prepare_bundle(args.output)
    except (BundleError, OSError, ValueError) as exc:
        print(f"preparation failed: {exc}", file=sys.stderr)
        return 2
    print(f"validated 8B bundle prepared at {args.output.expanduser().absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
