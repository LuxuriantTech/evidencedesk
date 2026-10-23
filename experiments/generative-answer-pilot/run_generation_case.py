"""One candidate-model evaluation in an isolated Python process; exposed dev only."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from ci_common import (GENERATION_MANIFEST, GENERATION_MANIFEST_SHA256,
                       environment_record as common_environment_record,
                       same_server_identity, server_identity_sha256,
                       sha256_file as common_sha256_file, tree_hash, verify_dataset)

HERE = Path(__file__).resolve().parent
DATASETS = ("E4", "E5", "E6")
CANDIDATE_COMMIT = "120e233bb8ff09cabc1bd3c5004a8545039459aa"
EXPECTED_DATA_MANIFEST_SHA256 = GENERATION_MANIFEST_SHA256
PROVIDER_SHA256 = "d03addcbd0074b06e543791fae932587866e159781c0f9c2a5bf2c2ce41578cc"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_hash(root: Path) -> dict[str, str]:
    if root.is_file():
        return {root.name: sha256_file(root)}
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}


def environment_record(venv: Path, lock_path: Path) -> dict[str, Any]:
    return common_environment_record(venv, lock_path)


def verify_inputs(dataset: str, root: Path) -> tuple[Path, Path, dict[str, str]]:
    observed = verify_dataset(dataset, root)
    corpus = json.loads((root / dataset / "corpus_manifest.json").read_text(encoding="utf-8"))
    return root / dataset / "evaluation_cases.json", root / dataset / "corpus_manifest.json", observed


def _write_exclusive_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class DurableAnswerProvider:
    """Persist each completed model call before evaluation advances to its next case."""

    def __init__(self, provider: Any, output: Path) -> None:
        self.provider = provider
        self.output = output
        self.index = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    def answer(self, question: str, ranked: Any) -> Any:
        self.index += 1
        index = self.index
        result: Any = None
        failure: dict[str, str] | None = None
        diagnostics_before = len(self.provider.diagnostics)
        try:
            result = self.provider.answer(question, ranked)
            return result
        except BaseException as exc:
            failure = {"type": type(exc).__name__, "message": str(exc),
                       "traceback": traceback.format_exc()}
            raise
        finally:
            latest = (self.provider.diagnostics[-1]
                      if len(self.provider.diagnostics) > diagnostics_before else None)
            saved_result = asdict(result) if is_dataclass(result) else result
            _write_exclusive_json(self.output / f"answer-{index:03d}.json", {
                "index": index, "question": question, "response": saved_result,
                "provider_diagnostic": latest, "provider_diagnostic_present": latest is not None,
                "exception": failure,
            })


def git_identity(repo: Path) -> str:
    if not (repo / ".git").exists():
        raise RuntimeError("source must be a Git worktree")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != CANDIDATE_COMMIT:
        raise RuntimeError(f"candidate source commit mismatch: {commit}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise RuntimeError("candidate source must be a clean detached worktree")
    return commit


def server_info(base_url: str) -> dict[str, Any]:
    if base_url != "http://127.0.0.1:50193":
        raise RuntimeError("model endpoint must be the fixed loopback-only CI endpoint")
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = build_opener(ProxyHandler({}), NoRedirect())
    records: dict[str, Any] = {}
    for suffix in ("/health", "/v1/models", "/props"):
        req = Request(base_url + suffix, headers={"Accept": "application/json"})
        try:
            with opener.open(req, timeout=4) as response:
                raw = response.read(1024 * 1024)
                body = json.loads(raw.decode("utf-8"))
                records[suffix] = {"status": response.status, "sha256": hashlib.sha256(raw).hexdigest(),
                                   "raw_body_base64": base64.b64encode(raw).decode("ascii"),
                                   "body": body}
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"model server identity check failed at {suffix}: {type(exc).__name__}: {exc}") from exc
    return records


def validate_server_info(records: dict[str, Any], expected_model: str) -> None:
    if records["/health"]["status"] != 200 or records["/health"]["body"].get("status") != "ok":
        raise RuntimeError("model server health contract failed")
    ids = [x.get("id") for x in records["/v1/models"]["body"].get("data", []) if isinstance(x, dict)]
    if expected_model not in ids:
        raise RuntimeError("model server loaded the wrong model identity")


def snapshots_unchanged(pre: dict[str, Any], post: dict[str, Any]) -> bool:
    keys = ("source", "inputs", "model", "runtime_archive", "server_binary", "minilm", "environment", "config")
    return (all(pre.get(key) == post.get(key) for key in keys)
            and same_server_identity(pre["server"], post["server"]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--datasets-root", type=Path, default=HERE / "datasets")
    parser.add_argument("--minilm-model-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:50193")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--go", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    if not args.go:
        raise RuntimeError("candidate generation did not run: explicit CI --go required")
    repo = args.repo.resolve(strict=True)
    commit = git_identity(repo)
    if Path(sys.prefix).resolve() != args.venv.resolve(strict=True) or sys.version_info[:3] != (3, 12, 14):
        raise RuntimeError("candidate runner requires its separately recorded Python 3.12.14 CI venv")
    cases, corpus, input_hashes = verify_inputs(args.dataset, args.datasets_root.resolve(strict=True))
    expected_case_count = len(json.loads(cases.read_text(encoding="utf-8")).get("cases", []))
    if expected_case_count != 28:
        raise RuntimeError("exposed CI dataset must contain exactly 28 cases")
    for path in (args.minilm_model_dir, args.model_file, args.runtime_archive, args.server_binary):
        if not path.exists():
            raise FileNotFoundError(path)
    artifact = json.loads((HERE / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    if sha256_file(HERE / "ARTIFACT_MANIFEST.json") != "94a6bec40382c308f0b445b120ee1ab108b4c640c50cc9c17921756a313f5a90":
        raise RuntimeError("reviewed artifact manifest digest mismatch")
    if sha256_file(HERE / "generative_provider.py") != PROVIDER_SHA256:
        raise RuntimeError("reviewed generative provider digest mismatch")
    if (sha256_file(args.model_file) != artifact["model"]["sha256"]
            or args.model_file.stat().st_size != artifact["model"]["bytes"]):
        raise RuntimeError("model artifact SHA-256 mismatch")
    if (sha256_file(args.runtime_archive) != artifact["runtime"]["archive_sha256"]
            or args.runtime_archive.stat().st_size != artifact["runtime"]["archive_bytes"]):
        raise RuntimeError("runtime archive SHA-256 mismatch")
    if sha256_file(args.server_binary) != artifact["runtime"]["server_binary_sha256"]:
        raise RuntimeError("server binary SHA-256 mismatch")
    lock = repo / "uv.lock"
    model_manifest = repo / "infra/models/paraphrase-multilingual-minilm-l12-v2-text-onnx.json"
    required = (lock, model_manifest, repo / "evals/runner.py", repo / "evals/configs/answer-v3-frozen-v7.json")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("candidate source is missing one or more frozen runner/config blobs")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt_path = args.output / "run_receipt.json"
    per_answer_dir = args.output / "answer_calls"
    per_answer_dir.mkdir(exist_ok=False)
    tracked = [repo / "apps/api/evidencedesk_api", repo / "evals/runner.py",
               repo / "evals/answer_v3_runtime.py", *required, model_manifest,
               HERE / "run_generation_case.py", HERE / "generative_provider.py",
               HERE / "ARTIFACT_MANIFEST.json", HERE / "GENERATION_DATASET_MANIFEST.json",
               HERE / "ci_common.py"]

    # The snapshot imports the provider to fingerprint its prompt/schema. Make
    # all source and runner module paths available before that first import.
    sys.path[:0] = [str(repo / "apps/api"), str(repo), str(HERE)]

    def snapshot() -> dict[str, Any]:
        server = server_info(args.base_url)
        return {
            "source": {str(path): tree_hash(path) if path.is_dir() else sha256_file(path) for path in tracked},
            "inputs": {"evaluation_cases.json": sha256_file(cases), "corpus_manifest.json": sha256_file(corpus),
                       "input_manifest": sha256_file(GENERATION_MANIFEST)},
            "model": tree_hash(args.model_file), "runtime_archive": tree_hash(args.runtime_archive),
            "server_binary": tree_hash(args.server_binary), "minilm": tree_hash(args.minilm_model_dir),
            "environment": environment_record(args.venv, lock), "server": server,
            "server_identity_sha256": server_identity_sha256(server),
            "config": {"artifact_manifest_sha256": sha256_file(HERE / "ARTIFACT_MANIFEST.json"),
                       "provider_sha256": sha256_file(HERE / "generative_provider.py"),
                       "provider_prompt": __import__("generative_provider").PROMPT_FINGERPRINT,
                       "provider_schema": __import__("generative_provider").SCHEMA_FINGERPRINT,
                       "provider_config": __import__("generative_provider").CONFIG_FINGERPRINT},
        }

    started = {"status": "started", "role": "generation", "cohort": "new Ubuntu 24.04 public GitHub Actions CI",
               "classification": "exposed development only; not fresh/holdout", "dataset": args.dataset,
               "source_commit": commit, "expected_case_count": expected_case_count,
               "native_verdict": None, "fatal_error": None,
               "input_hashes_manifest": input_hashes, "pre": None, "post": None}
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(started, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    raw_path = args.output / "raw_results.json"
    diagnostics_path = args.output / "provider_diagnostics.json"
    provider = None
    failure: BaseException | None = None
    try:
        started["pre"] = snapshot()
        receipt_path.write_text(json.dumps(started, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        validate_server_info(started["pre"]["server"], artifact["model"]["id"])
        from evidencedesk_api.extraction import extract_supplier_fields_v3
        from evidencedesk_api.providers import LocalSemanticEmbeddingProvider
        from evidencedesk_api.retrieval import RetrievalMethod
        from evals.runner import evaluate_manifest, _engine_fingerprint
        from generative_provider import GenerativeAnswerProvider
        embeddings = LocalSemanticEmbeddingProvider(manifest_path=model_manifest, model_path=args.minilm_model_dir)
        provider = GenerativeAnswerProvider(base_url=args.base_url, model_id=artifact["model"]["id"],
                                            seed=artifact["generation"]["seed"],
                                            temperature=artifact["generation"]["temperature"],
                                            enable_thinking=artifact["generation"]["enable_thinking"],
                                            max_tokens=artifact["generation"]["max_tokens"], timeout=180.0)
        durable_provider = DurableAnswerProvider(provider, per_answer_dir)
        result = evaluate_manifest(
            cases, corpus, split="development", provider=embeddings, method=RetrievalMethod.HYBRID,
            runtime_metadata={"execution": "public CI exposed development only", "dataset": args.dataset,
                              "source_commit": commit, "artifact_manifest_sha256": sha256_file(HERE / "ARTIFACT_MANIFEST.json"),
                              "environment": environment_record(args.venv, lock), "engine_fingerprint": _engine_fingerprint()},
            answer_provider=durable_provider,
            extraction_provider=lambda chunks: extract_supplier_fields_v3(chunks, embeddings=embeddings),
        )
        raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        started["native_verdict"] = result.get("verdict")
    except BaseException as exc:
        failure = exc
        started["status"] = "execution_failed"
        started["fatal_error"] = {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()}
        (args.output / "execution_error.json").write_text(json.dumps(started["fatal_error"],
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        if provider is not None:
            try:
                diagnostics_path.write_text(json.dumps(provider.diagnostics, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
                started["provider_diagnostics_sha256"] = sha256_file(diagnostics_path)
                started["durable_answer_call_count"] = len(list(per_answer_dir.glob("answer-*.json")))
                durable_records = [json.loads(path.read_text(encoding="utf-8"))
                                   for path in sorted(per_answer_dir.glob("answer-*.json"))]
                started["durable_answer_diagnostic_missing_count"] = sum(
                    item.get("provider_diagnostic_present") is not True for item in durable_records)
                started["model_validation_error_count"] = sum(
                    item.get("model_validation_error") is not None for item in provider.diagnostics)
                started["backend_error_count"] = sum(
                    item.get("backend_error_type") is not None or item.get("preflight_error_type") is not None
                    for item in provider.diagnostics)
                if (started["durable_answer_call_count"] != expected_case_count
                        or started["durable_answer_diagnostic_missing_count"]
                        or started["model_validation_error_count"] or started["backend_error_count"]) and failure is None:
                    failure = RuntimeError("generation calls are incomplete or provider diagnostics contain errors")
                    started["status"] = "technical_failure"
            except BaseException as exc:
                started["diagnostics_finalize_error"] = {"type": type(exc).__name__, "message": str(exc),
                                                          "traceback": traceback.format_exc()}
                failure = failure or exc
        try:
            started["post"] = snapshot()
            pre, post = started.get("pre") or {}, started["post"] or {}
            validate_server_info(post["server"], artifact["model"]["id"])
            started["unchanged"] = snapshots_unchanged(pre, post)
            if not started["unchanged"]:
                raise RuntimeError("generation source/model/environment/input snapshot changed during run")
        except BaseException as exc:
            started["post_check_error"] = {"type": type(exc).__name__, "message": str(exc),
                                            "traceback": traceback.format_exc()}
            started["status"] = "post_check_failed"
            failure = failure or exc
        if failure is None:
            started["status"] = "completed_exposed_development"
        elif started.get("status") != "post_check_failed":
            started["status"] = "technical_failure" if (
                started.get("model_validation_error_count", 0) or started.get("backend_error_count", 0)
            ) else "execution_failed"
        started["finished_epoch"] = time.time()
        receipt_path.write_text(json.dumps(started, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failure:
        raise failure


if __name__ == "__main__":
    run(parse_args())
