#!/usr/bin/env python3
"""Create unique, fake-only run metadata and a private temporary Compose override."""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import stat
import uuid
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-root", required=True, type=Path)
    p.add_argument("--runner-temp", required=True, type=Path)
    p.add_argument("--compose-template", required=True, type=Path)
    args = p.parse_args()

    temp = args.runner_temp.resolve(strict=True)
    candidate = args.candidate_root.resolve(strict=True)
    template = (candidate / args.compose_template).resolve(strict=True)
    if candidate not in template.parents or not template.is_file():
        raise SystemExit("FAIL step=prepare code=template_outside_candidate")
    seed_source = template.parent / "seed_http_smoke.py"
    if not seed_source.is_file():
        raise SystemExit("FAIL step=prepare code=seed_helper_missing")
    env_file = os.environ.get("GITHUB_ENV")
    if not env_file or not Path(env_file).is_file():
        raise SystemExit("FAIL step=prepare code=github_env_missing")
    run_id, attempt = (
        os.environ.get("GITHUB_RUN_ID", "local"),
        os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
    )
    if not run_id.isalnum() or not attempt.isalnum():
        raise SystemExit("FAIL step=prepare code=invalid_run_identity")
    project = f"evidencedesk-httpq-{run_id}-{attempt}".lower()
    run_dir = temp / project
    if run_dir.exists() or run_dir.is_symlink():
        raise SystemExit("FAIL step=prepare code=run_directory_already_exists")
    bundle = run_dir / "bundle"
    override = run_dir / "compose.override.yaml"
    seed_script = run_dir / "http_qualification_seed.py"

    nonce = secrets.token_hex(5)
    values = {
        "COMPOSE_PROJECT_NAME": project,
        "HTTP_RUN_DIR": str(run_dir),
        "HTTP_BUNDLE_DIR": str(bundle),
        "HTTP_COMPOSE_OVERRIDE": str(override),
        "HTTP_SEED_SCRIPT": str(seed_script),
        "HTTP_API_IMAGE": f"evidencedesk-httpq-api:{run_id}-{attempt}",
        "HTTP_SIDECAR_IMAGE": f"evidencedesk-httpq-sidecar:{run_id}-{attempt}",
        "TEST_DB_PASSWORD": secrets.token_urlsafe(32),
        "TEST_JWT_SECRET": secrets.token_urlsafe(48),
        "SMOKE_ADMIN_PASSWORD": secrets.token_urlsafe(24),
        "SMOKE_ANALYST_PASSWORD": secrets.token_urlsafe(24),
        "SMOKE_READER_PASSWORD": secrets.token_urlsafe(24),
        "SMOKE_ANALYST_USERNAME": f"qa.analyst.{nonce}",
        "SMOKE_READER_USERNAME": f"qa.reader.{nonce}",
        "SMOKE_NONCE": nonce,
        "SMOKE_DOSSIER_A": str(uuid.uuid4()),
        "SMOKE_DOSSIER_B": str(uuid.uuid4()),
        "SMOKE_DOSSIER_EMPTY": str(uuid.uuid4()),
    }
    lines = []
    for key, value in values.items():
        if "\n" in value or "\r" in value:
            raise SystemExit("FAIL step=prepare code=invalid_env_value")
        lines.append(f"{key}={value}\n")
    run_dir.mkdir(mode=0o700)
    try:
        shutil.copyfile(template, override)
        override.chmod(stat.S_IRUSR | stat.S_IWUSR)
        shutil.copyfile(seed_source, seed_script)
        seed_script.chmod(stat.S_IRUSR | stat.S_IWUSR)
        with open(env_file, "r+b") as f:
            previous_size = f.seek(0, os.SEEK_END)
            try:
                f.write("".join(lines).encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            except BaseException:
                f.truncate(previous_size)
                f.flush()
                os.fsync(f.fileno())
                raise
    except BaseException as exc:
        if run_dir.parent == temp and run_dir.is_dir() and not run_dir.is_symlink():
            shutil.rmtree(run_dir)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise SystemExit("FAIL step=prepare code=setup_failed") from None
    print(
        "PASS step=prepare unique_project=true fake_credentials=true temporary_override=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
