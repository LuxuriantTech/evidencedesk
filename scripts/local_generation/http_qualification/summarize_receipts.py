#!/usr/bin/env python3
"""Reduce smoke receipts to statuses and resource ceilings, omitting IDs/content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_CHECKS = {
    "available": {
        "api_health_ready",
        "unauthenticated_401",
        "role_upload_denied_403",
        "authenticated_upload_202",
        "worker_ingestion_completed",
        "citation_bounded_to_requested_dossier",
        "empty_dossier_abstained_without_citation",
    },
    "unavailable": {"backend_stopped_returned_sanitized_503"},
}


def phase_status(path: Path, expected_phase: str) -> str:
    if not path.exists():
        return "not_recorded"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("phase") != expected_phase:
        return "invalid"
    checks = data.get("checks")
    if (
        not isinstance(checks, list)
        or any(not isinstance(check, str) for check in checks)
        or len(checks) != len(set(checks))
        or set(checks) != EXPECTED_CHECKS[expected_phase]
    ):
        return "invalid"
    return "pass"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--http-receipt", required=True, type=Path)
    p.add_argument("--http-unavailable", required=True, type=Path)
    p.add_argument("--memory-receipt", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    memory = json.loads(args.memory_receipt.read_text(encoding="utf-8"))
    result = {
        "available_http": phase_status(args.http_receipt, "available"),
        "unavailable_http": phase_status(args.http_unavailable, "unavailable"),
        "host_memory_guard": memory.get("outcome", "invalid"),
        "host_memory_limit_bytes": memory.get("limit_bytes"),
        "peak_host_used_bytes": memory.get("host_high_water_used_bytes"),
        "raw_prompts_answers_tokens_or_document_ids_included": False,
    }
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    if (
        result["available_http"] != "pass"
        or result["unavailable_http"] != "pass"
        or result["host_memory_guard"] != "stopped_cleanly"
    ):
        print("FAIL step=summary code=qualification_incomplete")
        return 1
    print("PASS step=summary safe_fields_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
