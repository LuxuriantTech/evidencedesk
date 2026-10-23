#!/usr/bin/env python3
"""Synthetic-only HTTP smoke for an already-started EvidenceDesk 8B pilot stack."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

MAX_RESPONSE_BYTES = 1_000_000
MAX_WAIT_SECONDS = 180
DEFAULT_REQUEST_TIMEOUT_SECONDS = 20.0
ASK_REQUEST_TIMEOUT_SECONDS = 240.0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> urllib.request.Request | None:
        return None


class SmokeFailure(Exception):
    def __init__(self, step: str, code: str) -> None:
        self.step = step
        self.code = code
        super().__init__(f"{step}:{code}")


def loopback_base(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.port is None
    ):
        raise SmokeFailure("base_url", "not_loopback_http")
    return f"http://{parsed.hostname}:{parsed.port}"


def _read_json(response: Any, step: str) -> tuple[int, Any]:
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise SmokeFailure(step, "response_too_large")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SmokeFailure(step, "invalid_json") from None
    return int(response.status), data


def request_json(
    base: str,
    path: str,
    *,
    method: str = "GET",
    token: str | None = None,
    json_body: dict[str, Any] | None = None,
    form_body: dict[str, str] | None = None,
    multipart: tuple[str, bytes, str] | None = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    step: str,
) -> tuple[int, Any]:
    if not 0 < timeout <= 300:
        raise SmokeFailure(step, "invalid_timeout")
    headers = {"Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body: bytes | None = None
    if json_body is not None:
        body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif form_body is not None:
        body = urllib.parse.urlencode(form_body).encode("ascii")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif multipart is not None:
        filename, payload, mime = multipart
        boundary = f"ed-http-{uuid.uuid4().hex}"
        parts = [
            (
                f"--{boundary}\r\nContent-Disposition: form-data; "
                'name="is_synthetic"\r\n\r\ntrue\r\n'
            ).encode(),
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'
            ).encode(),
            payload,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        body = b"".join(parts)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    request = urllib.request.Request(  # noqa: S310 - base is validated loopback and path is caller-controlled only inside this script.
        base + path, data=body, headers=headers, method=method
    )
    try:
        # The URL is already constrained to loopback. Disable environment proxies and
        # redirects as a second guard so credentials can never leave that origin.
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )
        with opener.open(request, timeout=timeout) as response:
            return _read_json(response, step)
    except urllib.error.HTTPError as error:
        try:
            return _read_json(error, step)
        except SmokeFailure:
            return int(error.code), {}
    except (urllib.error.URLError, TimeoutError, OSError):
        raise SmokeFailure(step, "transport_error") from None


def require_status(
    result: tuple[int, Any], expected: int, step: str, code: str | None = None
) -> Any:
    status, payload = result
    if status != expected:
        raise SmokeFailure(step, f"http_{status}")
    if code is not None:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if not isinstance(detail, dict) or detail.get("code") != code:
            raise SmokeFailure(step, "unexpected_error_code")
    return payload


def validate_answered(
    payload: dict[str, Any], *, doc_id: str, date: str, source: str, step: str
) -> None:
    citations = payload.get("citations")
    if payload.get("status") != "answered" or payload.get("answerable") is not True:
        raise SmokeFailure(step, "not_answered")
    if not isinstance(citations, list) or not citations:
        raise SmokeFailure(step, "citation_missing")
    if payload.get("supporting_document") != doc_id:
        raise SmokeFailure(step, "supporting_document_mismatch")
    if date not in str(payload.get("answer", "")):
        raise SmokeFailure(step, "answer_span_mismatch")
    if date not in str(payload.get("supporting_excerpt", "")):
        raise SmokeFailure(step, "supporting_excerpt_mismatch")
    for citation in citations:
        if not isinstance(citation, dict) or citation.get("document_id") != doc_id:
            raise SmokeFailure(step, "cross_dossier_citation")
        excerpt = citation.get("excerpt")
        if (
            citation.get("page") != 1
            or not isinstance(excerpt, str)
            or excerpt not in source
        ):
            raise SmokeFailure(step, "citation_source_mismatch")


def validate_abstention(payload: dict[str, Any]) -> None:
    if (
        payload.get("status") != "abstained"
        or payload.get("answerable") is not False
        or payload.get("citations") != []
        or payload.get("supporting_document") is not None
        or payload.get("extracted_fields") != {}
    ):
        raise SmokeFailure("empty_dossier_refusal", "abstention_contract_failed")


def _uuid_env(key: str) -> str:
    try:
        return str(uuid.UUID(os.environ[key]))
    except (KeyError, ValueError):
        raise SmokeFailure("fixture_config", f"invalid_{key.lower()}") from None


def _write_receipt(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _login(base: str, username: str, password: str, role: str) -> str:
    payload = require_status(
        request_json(
            base,
            "/api/v1/auth/token",
            method="POST",
            form_body={"username": username, "password": password},
            step=f"login_{role}",
        ),
        200,
        f"login_{role}",
    )
    token = payload.get("access_token")
    if not isinstance(token, str) or len(token) < 16:
        raise SmokeFailure(f"login_{role}", "token_missing")
    return token


def _upload(
    base: str, token: str, dossier_id: str, filename: str, text: str, step: str
) -> str:
    payload = require_status(
        request_json(
            base,
            f"/api/v1/dossiers/{dossier_id}/documents",
            method="POST",
            token=token,
            multipart=(filename, text.encode("utf-8"), "text/plain"),
            step=step,
        ),
        202,
        step,
    )
    try:
        return str(uuid.UUID(str(payload["id"])))
    except (KeyError, ValueError):
        raise SmokeFailure(step, "document_id_missing") from None


def _wait_ingested(
    base: str, token: str, dossier_id: str, document_id: str, step: str
) -> None:
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        payload = require_status(
            request_json(
                base,
                f"/api/v1/dossiers/{dossier_id}/documents",
                token=token,
                step=step,
            ),
            200,
            step,
        )
        if not isinstance(payload, list):
            # This endpoint returns a JSON array; the transport validates dicts for
            # API contracts, so use its raw array helper below if reached.
            raise SmokeFailure(step, "invalid_document_list")
        for item in payload:
            if isinstance(item, dict) and item.get("id") == document_id:
                if item.get("status") == "completed":
                    break
                if item.get("status") == "failed":
                    raise SmokeFailure(step, "ingestion_failed")
        else:
            time.sleep(2)
            continue
        if any(
            isinstance(item, dict)
            and item.get("id") == document_id
            and item.get("status") == "completed"
            for item in payload
        ):
            break
        time.sleep(2)
    else:
        raise SmokeFailure(step, "ingestion_timeout")


def _phase_available(base: str, receipt_path: Path) -> dict[str, Any]:
    dossier_a = _uuid_env("SMOKE_DOSSIER_A")
    dossier_b = _uuid_env("SMOKE_DOSSIER_B")
    dossier_empty = _uuid_env("SMOKE_DOSSIER_EMPTY")
    nonce = os.environ.get("SMOKE_NONCE", "")
    if not nonce or len(nonce) > 24 or not nonce.isalnum():
        raise SmokeFailure("fixture_config", "invalid_nonce")
    analyst_name = os.environ.get("SMOKE_ANALYST_USERNAME", "demo.analyst")
    analyst_password = os.environ.get("SMOKE_ANALYST_PASSWORD", "")
    reader_name = os.environ.get("SMOKE_READER_USERNAME", "demo.reader")
    reader_password = os.environ.get("SMOKE_READER_PASSWORD", "")
    if not analyst_password or not reader_password:
        raise SmokeFailure("fixture_config", "test_password_missing")

    checks: list[str] = []
    require_status(request_json(base, "/health", step="api_health"), 200, "api_health")
    require_status(request_json(base, "/ready", step="api_ready"), 200, "api_ready")
    checks.append("api_health_ready")
    require_status(
        request_json(base, "/api/v1/dossiers", step="unauthenticated_list"),
        401,
        "unauthenticated_list",
        "authentication_required",
    )
    require_status(
        request_json(
            base,
            f"/api/v1/dossiers/{dossier_a}/ask",
            method="POST",
            json_body={"question": "unauthenticated synthetic question"},
            step="unauthenticated_ask",
        ),
        401,
        "unauthenticated_ask",
        "authentication_required",
    )
    checks.append("unauthenticated_401")

    analyst = _login(base, analyst_name, analyst_password, "analyst")
    me = require_status(
        request_json(base, "/api/v1/auth/me", token=analyst, step="analyst_me"),
        200,
        "analyst_me",
    )
    if me.get("username") != analyst_name or me.get("role") != "analyst":
        raise SmokeFailure("analyst_me", "wrong_principal")
    reader = _login(base, reader_name, reader_password, "reader")
    require_status(
        request_json(
            base,
            f"/api/v1/dossiers/{dossier_a}/documents",
            method="POST",
            token=reader,
            multipart=(
                "reader-denied.txt",
                b"Synthetic denied upload fixture.",
                "text/plain",
            ),
            step="reader_upload_denied",
        ),
        403,
        "reader_upload_denied",
    )
    checks.append("role_upload_denied_403")

    asset = f"FIXTURE-{nonce}"
    source_a = f"Test-only synthetic record. Asset {asset} inspection date 2032-03-17."
    source_b = f"Test-only synthetic record. Asset {asset} inspection date 2032-11-29."
    question = f"What is the inspection date for asset {asset}?"
    doc_a = _upload(
        base, analyst, dossier_a, f"fixture-a-{nonce}.txt", source_a, "upload_a"
    )
    doc_b = _upload(
        base, analyst, dossier_b, f"fixture-b-{nonce}.txt", source_b, "upload_b"
    )
    checks.append("authenticated_upload_202")
    _wait_ingested(base, analyst, dossier_a, doc_a, "ingest_a")
    _wait_ingested(base, analyst, dossier_b, doc_b, "ingest_b")
    checks.append("worker_ingestion_completed")

    content_a = require_status(
        request_json(
            base, f"/api/v1/documents/{doc_a}/content", token=analyst, step="content_a"
        ),
        200,
        "content_a",
    )
    if not isinstance(content_a, list) or not content_a:
        raise SmokeFailure("content_a", "chunks_missing")
    if not all(
        isinstance(chunk, dict) and chunk.get("page") == 1 for chunk in content_a
    ):
        raise SmokeFailure("content_a", "chunk_shape_invalid")

    answer_a = require_status(
        request_json(
            base,
            f"/api/v1/dossiers/{dossier_a}/ask",
            method="POST",
            token=analyst,
            json_body={"question": question},
            timeout=ASK_REQUEST_TIMEOUT_SECONDS,
            step="ask_a",
        ),
        200,
        "ask_a",
    )
    validate_answered(
        answer_a, doc_id=doc_a, date="2032-03-17", source=source_a, step="ask_a"
    )
    answer_b = require_status(
        request_json(
            base,
            f"/api/v1/dossiers/{dossier_b}/ask",
            method="POST",
            token=analyst,
            json_body={"question": question},
            timeout=ASK_REQUEST_TIMEOUT_SECONDS,
            step="ask_b",
        ),
        200,
        "ask_b",
    )
    validate_answered(
        answer_b, doc_id=doc_b, date="2032-11-29", source=source_b, step="ask_b"
    )
    checks.append("citation_bounded_to_requested_dossier")

    empty_answer = require_status(
        request_json(
            base,
            f"/api/v1/dossiers/{dossier_empty}/ask",
            method="POST",
            token=analyst,
            json_body={"question": f"What date is recorded for absent asset {asset}?"},
            timeout=ASK_REQUEST_TIMEOUT_SECONDS,
            step="empty_dossier_refusal",
        ),
        200,
        "empty_dossier_refusal",
    )
    validate_abstention(empty_answer)
    checks.append("empty_dossier_abstained_without_citation")

    result = {
        "schema_version": 1,
        "phase": "available",
        "checks": checks,
        "dossiers": {"a": dossier_a, "b": dossier_b, "empty": dossier_empty},
        "documents": {"a": doc_a, "b": doc_b},
    }
    _write_receipt(receipt_path, result)
    return result


def _phase_unavailable(base: str, receipt_path: Path) -> dict[str, Any]:
    try:
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous.get("phase") != "available":
            raise ValueError("wrong phase")
        dossier_a = str(uuid.UUID(previous["dossiers"]["a"]))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise SmokeFailure("unavailable_receipt", "invalid_receipt") from None
    analyst_name = os.environ.get("SMOKE_ANALYST_USERNAME", "demo.analyst")
    analyst_password = os.environ.get("SMOKE_ANALYST_PASSWORD", "")
    token = _login(base, analyst_name, analyst_password, "analyst")
    nonce = os.environ.get("SMOKE_NONCE", "fallback")
    status, payload = request_json(
        base,
        f"/api/v1/dossiers/{dossier_a}/ask",
        method="POST",
        token=token,
        json_body={"question": f"Repeat synthetic fixture question {nonce}."},
        timeout=ASK_REQUEST_TIMEOUT_SECONDS,
        step="backend_unavailable",
    )
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if (
        status != 503
        or not isinstance(detail, dict)
        or detail.get("code") != "local_model_unavailable"
    ):
        raise SmokeFailure("backend_unavailable", "expected_sanitized_503")
    result = {
        "schema_version": 1,
        "phase": "unavailable",
        "checks": ["backend_stopped_returned_sanitized_503"],
    }
    _write_receipt(receipt_path.with_name("http-unavailable-receipt.json"), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("available", "unavailable"), required=True)
    parser.add_argument("--base-url", default=os.environ.get("SMOKE_API_BASE_URL", ""))
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        base = loopback_base(args.base_url)
        result = (
            _phase_available(base, args.receipt)
            if args.phase == "available"
            else _phase_unavailable(base, args.receipt)
        )
    except SmokeFailure as exc:
        print(f"FAIL step={exc.step} code={exc.code}", file=sys.stderr)
        return 1
    except Exception:
        # Do not echo exception details; those can include request bodies or tokens.
        print("FAIL step=unexpected code=internal_error", file=sys.stderr)
        return 1
    for check in result["checks"]:
        print(f"PASS check={check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
