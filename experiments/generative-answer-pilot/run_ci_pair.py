"""Pair one frozen baseline/gen run on one exposed E4/E5/E6 dataset in CI."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from ci_common import (DATASETS, GENERATION_MANIFEST, GENERATION_MANIFEST_SHA256,
                       environment_record as common_environment_record,
                       same_server_identity as same_static_server_identity,
                       server_identity_sha256 as static_server_identity_sha256,
                       sha256_file as common_sha256_file, tree_hash, verify_dataset)

HERE = Path(__file__).resolve().parent
BASELINE_COMMIT = "ffeba12c4d4d64a1e1f003f211dfa1781c64515d"
CANDIDATE_COMMIT = "120e233bb8ff09cabc1bd3c5004a8545039459aa"
COMMON_BLOBS = (
    "uv.lock", "evals/runner.py", "evals/answer_v3_runtime.py",
    "evals/configs/answer-v3-frozen-v7.json",
    "infra/models/paraphrase-multilingual-minilm-l12-v2-text-onnx.json",
)
DATA_MANIFEST_SHA256 = GENERATION_MANIFEST_SHA256
ARTIFACT_SHA256 = "94a6bec40382c308f0b445b120ee1ab108b4c640c50cc9c17921756a313f5a90"
URL = "http://127.0.0.1:50193"
SERVER_CONFIG = {
    "host": "127.0.0.1", "port": 50193, "threads": 4,
    "threads_batch": 4, "batch_size": 128, "ubatch_size": 128, "parallel": 1,
    "n_gpu_layers": 0, "reasoning": "off", "no_context_shift": True,
    "alias": "Qwen3-4B-Q4_K_M.gguf",
}
GENERATION_CONFIG = {"seed": 20261023, "temperature": 0.0, "enable_thinking": False,
                     "max_tokens": 512, "context_tokens": 6144, "max_input_tokens": 5632}
RESOURCE = {"memory_max_bytes": 6442450944, "memory_swap_max_bytes": 0,
            "cpu_quota": "400%", "nice": 10}
MIN_HOST_MEM_AVAILABLE_BYTES = 8 * 1024 * 1024 * 1024
CI_CODE_FILES = ("ci_common.py", "run_ci_pair.py", "run_generation_case.py", "remote_baseline.py",
                 "run_smoke_cases.py", "run_ci_pair.sh", "prepare_artifacts.py", "generative_provider.py",
                 "ARTIFACT_MANIFEST.json", "GENERATION_DATASET_MANIFEST.json", "README.md",
                 "test_gen_ci_preparation.py")
PROVIDER_SHA256 = "e77beaedc9f18b58a4c8287161fb516603c7bb39679668d05a1f3cc8a76447f9"
SMOKE_CASES = 4


def verify_resource_values(memory: int, swap: int, cpu_quota: int, cpu_period: int, nice: int) -> dict[str, Any]:
    actual = {"memory_max_bytes": memory, "memory_swap_max_bytes": swap,
              "cpu_quota": f"{cpu_quota * 100 / cpu_period:g}%", "nice": nice}
    if actual != RESOURCE:
        raise RuntimeError(f"CI cgroup resource caps differ from required scope: {actual}")
    return actual


def server_lifecycle_error(before_cleanup: int | None, final_returncode: int | None,
                           forced_kill: bool) -> str | None:
    if before_cleanup is not None:
        return f"llama-server exited unexpectedly before cleanup: {before_cleanup}"
    if forced_kill:
        return "llama-server required forced kill during cleanup"
    if final_returncode not in (0, -signal.SIGTERM):
        return f"llama-server cleanup returned unexpected status {final_returncode}"
    return None


def validate_generation_receipt(receipt: dict[str, Any]) -> None:
    if (receipt.get("status") != "completed_exposed_development"
            or receipt.get("model_validation_error_count", 0) != 0
            or receipt.get("backend_error_count", 0) != 0):
        raise RuntimeError("generation child reports a technical/provider validation failure")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    if root.is_file():
        return {root.name: sha256_file(root)}
    return {item.relative_to(root).as_posix(): sha256_file(item) for item in sorted(root.rglob("*"))
            if item.is_file() and ".git" not in item.parts and "__pycache__" not in item.parts}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--repo-baseline", type=Path, required=True)
    parser.add_argument("--repo-candidate", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=HERE / "datasets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--go", action="store_true", help="explicit CI GO gate")
    return parser.parse_args(argv)


def git_commit(repo: Path, expected: str) -> str:
    repo = repo.resolve(strict=True)
    if not (repo / ".git").exists():
        raise RuntimeError(f"source worktree missing: {repo}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != expected:
        raise RuntimeError(f"source commit mismatch for {repo.name}: {commit}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise RuntimeError(f"source worktree is not clean: {repo}")
    if subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=repo, check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError(f"source worktree is not detached: {repo}")
    return commit


def verify_common_blobs(baseline: Path, candidate: Path) -> dict[str, str]:
    blobs: dict[str, str] = {}
    for rel in COMMON_BLOBS:
        a, b = baseline / rel, candidate / rel
        if not a.is_file() or not b.is_file():
            raise RuntimeError(f"required common scoring/config blob missing: {rel}")
        left, right = sha256_file(a), sha256_file(b)
        if left != right:
            raise RuntimeError(f"baseline/candidate common blob differs: {rel}")
        blobs[rel] = left
    return blobs


def validate_cgroup() -> dict[str, Any]:
    try:
        group = next(line.split("::", 1)[1].strip() for line in Path("/proc/self/cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        root = Path("/sys/fs/cgroup") / group.lstrip("/")
        memory = int((root / "memory.max").read_text().strip())
        swap = int((root / "memory.swap.max").read_text().strip())
        cpu_quota, cpu_period = (int(x) for x in (root / "cpu.max").read_text().split())
        nice = os.getpriority(os.PRIO_PROCESS, 0)
    except Exception as exc:
        raise RuntimeError(f"cannot verify required cgroup v2 resource scope: {exc}") from exc
    actual = verify_resource_values(memory, swap, cpu_quota, cpu_period, nice)
    return {"group": group, "actual": actual}


def host_mem_available_bytes(meminfo: str | None = None) -> int:
    content = Path("/proc/meminfo").read_text(encoding="ascii") if meminfo is None else meminfo
    for line in content.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                break
            value = int(fields[1]) * 1024
            if value < MIN_HOST_MEM_AVAILABLE_BYTES:
                raise RuntimeError(f"host MemAvailable below 8 GiB: {value} bytes")
            return value
    raise RuntimeError("Linux /proc/meminfo does not report MemAvailable")


def verify_inputs(dataset: str, root: Path) -> dict[str, str]:
    if sha256_file(GENERATION_MANIFEST) != DATA_MANIFEST_SHA256:
        raise RuntimeError("generation dataset manifest digest mismatch")
    return verify_dataset(dataset, root)


def _safe_artifact_path(base: Path, raw: str) -> Path:
    rel = Path(raw)
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"unsafe artifact manifest path: {raw}")
    resolved = (base / rel).resolve(strict=True)
    if not resolved.is_relative_to(base.resolve(strict=True)):
        raise RuntimeError(f"artifact manifest path escapes output root: {raw}")
    return resolved


def load_artifact_manifest(path: Path, candidate_repo: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    path = path.resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "verified":
        raise RuntimeError("artifact preparation manifest status must be verified")
    if record.get("source_repo_commit") != CANDIDATE_COMMIT:
        raise RuntimeError("artifact manifest source commit does not match candidate")
    if record.get("artifact_manifest_sha256") != sha256_file(HERE / "ARTIFACT_MANIFEST.json"):
        raise RuntimeError("artifact manifest does not bind the reviewed pinned artifact manifest")
    base = path.parent
    model = record["model"]
    runtime = record["runtime"]
    minilm = record["minilm"]
    paths = {
        "model": _safe_artifact_path(base, model["path"]),
        "runtime_archive": _safe_artifact_path(base, runtime["archive_path"]),
        "runtime_root": _safe_artifact_path(base, runtime["root"]),
        "server_binary": _safe_artifact_path(base, runtime["server_binary"]),
        "minilm_dir": _safe_artifact_path(base, minilm["model_dir"]),
    }
    minilm_rel = Path(minilm["model_manifest_path"])
    if minilm_rel.is_absolute() or ".." in minilm_rel.parts:
        raise RuntimeError("unsafe MiniLM manifest path")
    paths["minilm_manifest"] = (candidate_repo / minilm_rel).resolve(strict=True)
    if not paths["minilm_manifest"].is_relative_to(candidate_repo.resolve(strict=True)):
        raise RuntimeError("MiniLM source manifest resolves outside candidate checkout")
    artifact_spec = json.loads((HERE / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    if sha256_file(HERE / "ARTIFACT_MANIFEST.json") != ARTIFACT_SHA256:
        raise RuntimeError("reviewed artifact manifest digest mismatch")
    if (model.get("id") != artifact_spec["model"]["id"]
            or model.get("revision") != artifact_spec["model"]["revision"]
            or model.get("bytes") != artifact_spec["model"]["bytes"]
            or sha256_file(paths["model"]) != model["sha256"]
            or model["sha256"] != "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"):
        raise RuntimeError("model digest mismatch")
    if (runtime.get("version") != artifact_spec["runtime"]["version"]
            or runtime.get("commit") != artifact_spec["runtime"]["commit"]
            or sha256_file(paths["runtime_archive"]) != runtime["archive_sha256"]
            or runtime["archive_sha256"] != artifact_spec["runtime"]["archive_sha256"]):
        raise RuntimeError("llama.cpp runtime archive digest mismatch")
    if sha256_file(paths["server_binary"]) != runtime["server_binary_sha256"] or runtime["server_binary_sha256"] != artifact_spec["runtime"]["server_binary_sha256"]:
        raise RuntimeError("llama-server binary digest mismatch")
    if sha256_file(paths["minilm_manifest"]) != minilm["model_manifest_sha256"]:
        raise RuntimeError("MiniLM source manifest digest mismatch")
    for file in minilm["files"]:
        item = paths["minilm_dir"] / file["path"]
        if not item.is_file() or item.stat().st_size != file["bytes"] or sha256_file(item) != file["sha256"]:
            raise RuntimeError(f"MiniLM model asset mismatch: {file['path']}")
    caps = {key: record.get("resource_scope", {}).get(key) for key in RESOURCE}
    if (record.get("server_config") != SERVER_CONFIG
            or SERVER_CONFIG != artifact_spec["server"] or GENERATION_CONFIG != artifact_spec["generation"]
            or caps != RESOURCE):
        raise RuntimeError("artifact manifest server/resource configuration mismatch")
    return record, paths


def environment_record(venv: Path, lock_path: Path) -> dict[str, Any]:
    return common_environment_record(venv, lock_path)


def _health(url: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for suffix in ("/health", "/v1/models", "/props"):
        request = urllib.request.Request(url + suffix, headers={"Accept": "application/json"})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=5) as response:
            raw = response.read(1024 * 1024)
            result[suffix] = {"status": response.status, "body_sha256": hashlib.sha256(raw).hexdigest(),
                              "raw_body_base64": base64.b64encode(raw).decode("ascii"),
                              "body": json.loads(raw.decode("utf-8"))}
    return result


def validate_server_identity(identity: dict[str, Any], expected_id: str) -> None:
    if identity["/health"]["body"].get("status") != "ok":
        raise RuntimeError("llama-server health did not return ok")
    ids = [item.get("id") for item in identity["/v1/models"]["body"].get("data", []) if isinstance(item, dict)]
    if expected_id not in ids:
        raise RuntimeError("llama-server model ID differs from frozen model")


def _write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _run_child(command: list[str], cwd: Path, output: Path, env: dict[str, str]) -> int:
    output.mkdir(parents=True, exist_ok=False)
    _write_new(output / "START.json", {"status": "started", "command": command,
                                       "started_epoch": time.time()})
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False)
    receipt = {"status": "completed" if completed.returncode == 0 else "failed",
               "returncode": completed.returncode, "stdout_sha256": sha256_file(output / "stdout.log"),
               "stderr_sha256": sha256_file(output / "stderr.log"), "finished_epoch": time.time()}
    (output / "FINISH.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return completed.returncode


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], int, Path, Path, str, str, dict[str, str], dict[str, Any], dict[str, str], dict[str, Any], dict[str, Path], dict[str, str]]:
    if shutil.disk_usage(args.output.parent.resolve()).free < 1024 * 1024 * 1024:
        raise RuntimeError("less than 1 GiB free disk before model run")
    scope = validate_cgroup()
    baseline, candidate = args.repo_baseline.resolve(strict=True), args.repo_candidate.resolve(strict=True)
    bcommit = git_commit(baseline, BASELINE_COMMIT)
    ccommit = git_commit(candidate, CANDIDATE_COMMIT)
    common = verify_common_blobs(baseline, candidate)
    env_before = environment_record(args.venv, candidate / "uv.lock")
    input_hashes = verify_inputs(args.dataset, args.data_root.resolve(strict=True))
    if sha256_file(HERE / "generative_provider.py") != PROVIDER_SHA256:
        raise RuntimeError("reviewed generative provider digest mismatch")
    memory_available_before_model = host_mem_available_bytes()
    artifact, paths = load_artifact_manifest(args.model_manifest, candidate)
    if sha256_file(paths["minilm_manifest"]) != sha256_file(candidate / COMMON_BLOBS[-1]):
        raise RuntimeError("prepared MiniLM manifest differs from both source worktrees")
    ci_code_hashes = {name: sha256_file(HERE / name) for name in CI_CODE_FILES}
    return (scope, memory_available_before_model, baseline, candidate, bcommit, ccommit, common,
            env_before, input_hashes, artifact, paths, ci_code_hashes)


def run(args: argparse.Namespace) -> None:
    if not args.go:
        raise RuntimeError("no CI evaluation without explicit --go")
    if sys.platform != "linux":
        raise RuntimeError("portable CI runner supports Ubuntu Linux only")
    if not args.output.parent.is_dir() or args.output.exists():
        raise RuntimeError("output parent must exist and output path must be new")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt_path = args.output / "run_receipt.json"
    receipt: dict[str, Any] = {
        "status": "started", "classification": "exposed development only; not fresh/holdout",
        "cohort": "new public Ubuntu 24.04 GitHub Actions; distinct from local E3",
        "dataset": args.dataset, "started_epoch": time.time(),
        "raw_result_paths": {}, "fatal_error": None, "quality_qualified": False,
    }
    _write_new(receipt_path, receipt)
    try:
        (scope, memory_available_before_model, baseline, candidate, bcommit, ccommit, common, env_before, input_hashes,
         artifact, paths, ci_code_hashes) = _preflight(args)
    except BaseException as exc:
        receipt.update({"status": "preflight_failed", "fatal_error": {
            "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "finished_epoch": time.time()})
        (args.output / "preflight_error.json").write_text(json.dumps(receipt["fatal_error"],
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    receipt.update({
        "baseline_commit": bcommit, "candidate_commit": ccommit,
        "native_common_blobs_sha256": common, "environment_before": env_before,
        "resource_scope": scope, "server_config": SERVER_CONFIG, "generation_config": GENERATION_CONFIG,
        "host_mem_available_bytes_before_model": memory_available_before_model,
        "artifact_manifest_sha256": sha256_file(args.model_manifest), "artifact_manifest_before": artifact,
        "input_hashes_before": input_hashes, "model_sha256": sha256_file(paths["model"]),
        "runtime_archive_sha256": sha256_file(paths["runtime_archive"]),
        "runtime_binary_sha256": sha256_file(paths["server_binary"]),
        "runtime_tree_hash_before": hash_tree(paths["runtime_root"]),
        "minilm_hashes_before": hash_tree(paths["minilm_dir"]),
        "ci_code_hashes_before": ci_code_hashes,
        "preparation_manifest_sha256_before": sha256_file(args.model_manifest),
    })
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    server: subprocess.Popen[bytes] | None = None
    server_logs: list[Any] = []
    failure: BaseException | None = None
    env = {"HOME": os.environ.get("HOME", "/home/runner"), "PATH": "/usr/bin:/bin",
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
           "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
           "LD_LIBRARY_PATH": str(paths["runtime_root"])}
    try:
        # Baseline process first; no generative model server is resident during this role.
        baseline_cmd = [sys.executable, str(HERE / "remote_baseline.py"), "--repo", str(baseline),
                        "--datasets-root", str(args.data_root.resolve()), "--retriever-python-prefix", str(args.venv),
                        "--minilm-model-dir", str(paths["minilm_dir"]), "--output-root", str(args.output / "baseline"),
                        "--run-id", args.output.name, "--dataset", args.dataset, "--go"]
        code = _run_child(baseline_cmd, HERE, args.output / "baseline_process", env)
        receipt["baseline_returncode"] = code
        if code != 0:
            raise RuntimeError(f"baseline child exited {code}; see baseline_process stderr and receipts")
        baseline_result = args.output / "baseline" / args.output.name / "baseline" / args.dataset / "raw_results.json"
        if not baseline_result.is_file():
            raise RuntimeError("baseline did not emit its native raw_results.json")
        receipt["raw_result_paths"]["baseline"] = str(baseline_result.relative_to(args.output))

        model_id = artifact["model"]["id"]
        command = [str(paths["server_binary"]), "--model", str(paths["model"]), "--host", "127.0.0.1",
                   "--alias", SERVER_CONFIG["alias"], "--port", "50193", "--ctx-size", "6144", "--threads", "4", "--threads-batch", "4",
                   "--batch-size", "128", "--ubatch-size", "128", "--parallel", "1",
                   "--n-gpu-layers", "0", "--reasoning", "off", "--n-predict", "512",
                   "--seed", "20261023", "--jinja", "--no-context-shift"]
        server_logs = [(args.output / "server.stdout.log").open("xb"),
                       (args.output / "server.stderr.log").open("xb")]
        server = subprocess.Popen(command, cwd=paths["runtime_root"], env=env,
                                  stdout=server_logs[0], stderr=server_logs[1])
        receipt["server_started"] = {"pid": server.pid, "command": command, "started_epoch": time.time()}
        (args.output / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        deadline = time.monotonic() + 120
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup: {server.returncode}")
            try:
                identity_before = _health(URL)
            except (OSError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("llama-server readiness timeout")
                time.sleep(0.5)
                continue
            # Persist raw endpoint bodies before validating the alias so a wrong
            # operational model ID remains inspectable in the failed receipt.
            receipt["server_identity_before"] = identity_before
            receipt["server_identity_static_sha256_before"] = static_server_identity_sha256(identity_before)
            (args.output / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            validate_server_identity(identity_before, model_id)
            break
        (args.output / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        smoke_out = args.output / "smoke" / args.dataset
        smoke_cmd = [sys.executable, str(HERE / "run_smoke_cases.py"), "--repo120e", str(candidate),
                     "--venv", str(args.venv), "--base-url", URL, "--output", str(smoke_out), "--go"]
        code = _run_child(smoke_cmd, HERE, args.output / "smoke_process", env)
        receipt["smoke_returncode"] = code
        smoke_receipt = smoke_out / "FINISH.json"
        if not smoke_receipt.is_file():
            raise RuntimeError("smoke runner did not emit FINISH.json")
        smoke_summary = json.loads(smoke_receipt.read_text(encoding="utf-8"))
        receipt["smoke_status"] = smoke_summary.get("status")
        receipt["smoke_case_count"] = len(smoke_summary.get("cases", []))
        receipt["smoke_passed_count"] = sum(item.get("passed") is True for item in smoke_summary.get("cases", []))
        receipt["raw_result_paths"]["smoke"] = str(smoke_receipt.relative_to(args.output))
        if (code != 0 or smoke_summary.get("status") != "smoke_passed"
                or receipt["smoke_case_count"] != SMOKE_CASES or receipt["smoke_passed_count"] != SMOKE_CASES):
            raise RuntimeError("four invented smoke cases did not all pass; development dataset evaluation blocked")

        gen_out = args.output / "generation" / args.dataset
        gen_cmd = [sys.executable, str(HERE / "run_generation_case.py"), "--repo", str(candidate),
                   "--venv", str(args.venv), "--dataset", args.dataset,
                   "--datasets-root", str(args.data_root.resolve()), "--minilm-model-dir", str(paths["minilm_dir"]),
                   "--model-file", str(paths["model"]), "--runtime-archive", str(paths["runtime_archive"]),
                   "--server-binary", str(paths["server_binary"]), "--base-url", URL, "--output", str(gen_out), "--go"]
        code = _run_child(gen_cmd, HERE, args.output / "generation_process", env)
        receipt["generation_returncode"] = code
        if code != 0:
            raise RuntimeError(f"generation child exited {code}; see generation receipt and diagnostics")
        gen_result = gen_out / "raw_results.json"
        if not gen_result.is_file():
            raise RuntimeError("generation child did not emit native raw_results.json")
        receipt["raw_result_paths"]["generation"] = str(gen_result.relative_to(args.output))
        generation_receipt_path = gen_out / "run_receipt.json"
        generation_receipt = json.loads(generation_receipt_path.read_text(encoding="utf-8"))
        receipt["generation_status"] = generation_receipt.get("status")
        receipt["model_validation_error_count"] = generation_receipt.get("model_validation_error_count", 0)
        receipt["backend_error_count"] = generation_receipt.get("backend_error_count", 0)
        validate_generation_receipt(generation_receipt)
        if sha256_file(HERE / "generative_provider.py") != PROVIDER_SHA256:
            raise RuntimeError("generative provider changed from the reviewed snapshot")
        identity_after = _health(URL)
        receipt["server_identity_after"] = identity_after
        receipt["server_identity_static_sha256_after"] = static_server_identity_sha256(identity_after)
        receipt["server_identity_unchanged"] = same_static_server_identity(identity_before, identity_after)
        validate_server_identity(identity_after, model_id)
        if not receipt["server_identity_unchanged"]:
            raise RuntimeError("llama-server static model identity changed during the pair")
        receipt["native_verdicts"] = {
            "baseline": json.loads(baseline_result.read_text(encoding="utf-8")).get("verdict"),
            "generation": json.loads(gen_result.read_text(encoding="utf-8")).get("verdict"),
        }
    except BaseException as exc:
        failure = exc
        receipt["status"] = "execution_failed"
        receipt["fatal_error"] = {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()}
        (args.output / "execution_error.json").write_text(json.dumps(receipt["fatal_error"],
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        if server is not None:
            server_rc_before_cleanup: int | None = None
            try:
                server_rc_before_cleanup = server.poll()
                if server_rc_before_cleanup is not None:
                    receipt["server_unexpected_exit"] = {"returncode": server_rc_before_cleanup,
                                                          "observed_epoch": time.time()}
                else:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        receipt["server_forced_kill"] = True
                        server.kill()
                        server.wait(timeout=5)
                lifecycle_error = server_lifecycle_error(server_rc_before_cleanup, server.returncode,
                                                         receipt.get("server_forced_kill", False))
                if lifecycle_error is not None:
                    failure = failure or RuntimeError(lifecycle_error)
                    receipt["server_cleanup_error"] = {"returncode": server.returncode,
                                                       "message": lifecycle_error}
            except BaseException as exc:
                failure = failure or exc
                receipt["server_cleanup_error"] = {"type": type(exc).__name__, "message": str(exc),
                                                   "traceback": traceback.format_exc()}
            finally:
                receipt["server_stopped"] = {"returncode": server.returncode, "finished_epoch": time.time()}
        for stream in server_logs:
            try:
                stream.close()
            except BaseException as exc:
                failure = failure or exc
                receipt["server_log_close_error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            receipt["input_hashes_after"] = verify_inputs(args.dataset, args.data_root.resolve(strict=True))
            receipt["environment_after"] = environment_record(args.venv, candidate / "uv.lock")
            receipt["minilm_hashes_after"] = hash_tree(paths["minilm_dir"])
            receipt["model_sha256_after"] = sha256_file(paths["model"])
            receipt["runtime_archive_sha256_after"] = sha256_file(paths["runtime_archive"])
            receipt["runtime_binary_sha256_after"] = sha256_file(paths["server_binary"])
            receipt["runtime_tree_hash_after"] = hash_tree(paths["runtime_root"])
            receipt["source_commits_after"] = {
                "baseline": git_commit(baseline, BASELINE_COMMIT),
                "candidate": git_commit(candidate, CANDIDATE_COMMIT),
            }
            receipt["integrity_unchanged"] = (
                receipt["input_hashes_after"] == receipt["input_hashes_before"]
                and receipt["environment_after"] == receipt["environment_before"]
                and receipt["minilm_hashes_after"] == receipt["minilm_hashes_before"]
                and receipt["model_sha256_after"] == receipt["model_sha256"]
                and receipt["runtime_archive_sha256_after"] == receipt["runtime_archive_sha256"]
                and receipt["runtime_binary_sha256_after"] == receipt["runtime_binary_sha256"]
                and receipt["runtime_tree_hash_after"] == receipt["runtime_tree_hash_before"]
                and sha256_file(args.model_manifest) == receipt["artifact_manifest_sha256"]
                and sha256_file(HERE / "ARTIFACT_MANIFEST.json") == ARTIFACT_SHA256
                and sha256_file(GENERATION_MANIFEST) == DATA_MANIFEST_SHA256
                and receipt["ci_code_hashes_before"] == {name: sha256_file(HERE / name) for name in CI_CODE_FILES}
                and sha256_file(args.model_manifest) == receipt["preparation_manifest_sha256_before"]
            )
            if not receipt["integrity_unchanged"]:
                raise RuntimeError("one or more model/input/environment hashes changed across the pair")
        except BaseException as exc:
            receipt["post_check_error"] = {"type": type(exc).__name__, "message": str(exc),
                                           "traceback": traceback.format_exc()}
            receipt["status"] = "post_check_failed"
            failure = failure or exc
        receipt["finished_epoch"] = time.time()
        if failure is None:
            receipt["status"] = "completed_exposed_development"
        elif receipt.get("status") != "post_check_failed":
            receipt["status"] = "execution_failed"
        (args.output / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failure:
        raise failure


if __name__ == "__main__":
    run(parse_args())
