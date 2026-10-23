"""Fetch only or validate only the pinned llama.cpp runtime (no model)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bundle import (
    BundleError,
    download_runtime_archive,
    prepare_runtime_directory,
    verify_runtime_archive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--archive", type=Path, help="existing archive to validate without writing it"
    )
    group.add_argument("--download-to", type=Path, help="new file path; fetches runtime only")
    group.add_argument(
        "--prepare-dir", type=Path, help="new directory; downloads and extracts runtime only"
    )
    args = parser.parse_args()
    try:
        count = (
            verify_runtime_archive(args.archive)
            if args.archive is not None
            else (
                download_runtime_archive(args.download_to)
                if args.download_to is not None
                else prepare_runtime_directory(args.prepare_dir)
            )
        )
    except (BundleError, OSError, ValueError) as exc:
        print(f"runtime verification failed: {exc}", file=sys.stderr)
        return 2
    print(f"pinned runtime validated; {count} regular runtime paths checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
