#!/usr/bin/env python3
"""Verify read-only mounted model/runtime assets, then exec the pinned server."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .bundle import MANIFEST, BundleError, verify_bundle


def main() -> int:
    root = Path(os.environ.get("EVIDENCEDESK_8B_BUNDLE", "/bundle"))
    try:
        verify_bundle(root)
    except (BundleError, OSError, ValueError) as exc:
        print(f"asset verification failed: {exc}", file=sys.stderr)
        return 78
    model = MANIFEST["model"]
    server = MANIFEST["server"]
    runtime = MANIFEST["runtime"]
    root = root.resolve(strict=True)
    binary = root / "runtime" / runtime["server_binary"]
    model_path = root / "models" / model["id"]
    command = [
        "/usr/bin/nice", "-n", str(MANIFEST["resources"]["nice"]), str(binary),
        "--model", str(model_path), "--host", server["host"], "--port", str(server["port"]),
        "--alias", server["alias"], "--ctx-size", str(MANIFEST["generation"]["context_tokens"]),
        "--threads", str(server["threads"]), "--threads-batch", str(server["threads_batch"]),
        "--batch-size", str(server["batch_size"]), "--ubatch-size", str(server["ubatch_size"]),
        "--parallel", str(server["parallel"]), "--n-gpu-layers", str(server["n_gpu_layers"]),
        "--reasoning", server["reasoning"], "--n-predict",
        str(MANIFEST["generation"]["max_tokens"]),
        "--seed", str(MANIFEST["generation"]["seed"]), "--jinja", "--no-context-shift",
    ]
    runtime_dir = str(binary.parent)
    # Use only the verified runtime tree for bundled libraries; system libraries
    # continue to resolve through the loader's normal system paths.
    os.environ["LD_LIBRARY_PATH"] = runtime_dir
    os.execv(command[0], command)  # noqa: S606 - fixed argv, no shell or user input
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
