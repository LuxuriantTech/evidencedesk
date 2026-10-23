"""Run the frozen v7 answer baseline in its own CI process on exposed E4/E5/E6."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from ci_common import DATASETS, HERE, environment_record, sha256_file, tree_hash, verify_dataset

BASELINE_COMMIT = "ffeba12c4d4d64a1e1f003f211dfa1781c64515d"
CANDIDATE_COMMIT = "120e233bb8ff09cabc1bd3c5004a8545039459aa"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="clean detached FFeba baseline worktree")
    parser.add_argument("--datasets-root", type=Path, default=HERE / "datasets")
    parser.add_argument("--retriever-python-prefix", type=Path, required=True)
    parser.add_argument("--minilm-model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--go", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    if not args.go:
        raise RuntimeError("baseline calibration did not run: explicit CI --go required")
    if not args.run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in args.run_id):
        raise ValueError("run-id must be filesystem-safe")
    repo = args.repo.resolve(strict=True)
    if not (repo / ".git").exists():
        raise RuntimeError("--repo must be detached baseline source worktree")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != BASELINE_COMMIT:
        raise RuntimeError(f"baseline commit mismatch: {commit}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise RuntimeError("baseline source worktree must be clean")
    if Path(sys.prefix).resolve() != args.retriever_python_prefix.resolve():
        raise RuntimeError("baseline runner sys.prefix differs from --retriever-python-prefix")
    if sys.version_info[:3] != (3, 12, 14):
        raise RuntimeError("CI retrieval/scoring Python must be exactly 3.12.14")
    key = args.dataset
    input_hashes = ({key: verify_dataset(key, args.datasets_root)} if key != "all" else
                    {dataset: verify_dataset(dataset, args.datasets_root) for dataset in DATASETS})
    if not args.minilm_model_dir.exists():
        raise FileNotFoundError(args.minilm_model_dir)

    sys.path[:0] = [str(repo / "apps" / "api"), str(repo)]
    from evidencedesk_api.providers import LocalSemanticEmbeddingProvider
    from evidencedesk_api.retrieval import RetrievalMethod
    from evals.answer_v3_runtime import build_frozen_answer_runtime
    from evals.runner import evaluate_manifest, _engine_fingerprint

    model_manifest = repo / "infra" / "models" / "paraphrase-multilingual-minilm-l12-v2-text-onnx.json"
    config_path = repo / "evals" / "configs" / "answer-v3-frozen-v7.json"
    lock_path = repo / "uv.lock"
    for required in (model_manifest, config_path, lock_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    provider = LocalSemanticEmbeddingProvider(manifest_path=model_manifest, model_path=args.minilm_model_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    answer_runtime = build_frozen_answer_runtime(config, provider)
    tracked = [repo / "apps" / "api" / "evidencedesk_api", repo / "evals" / "runner.py",
               repo / "evals" / "answer_v3_runtime.py", config_path, model_manifest, lock_path]
    source_before = {str(path): (tree_hash(path) if path.is_dir() else sha256_file(path)) for path in tracked}
    model_before = tree_hash(args.minilm_model_dir)
    env_before = environment_record(args.retriever_python_prefix, lock_path)
    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for key in keys:
        manifest = args.datasets_root / key / "evaluation_cases.json"
        corpus = args.datasets_root / key / "corpus_manifest.json"
        output = args.output_root / args.run_id / "baseline" / key
        output.mkdir(parents=True, exist_ok=False)
        before = {"evaluation_cases.json": sha256_file(manifest), "corpus_manifest.json": sha256_file(corpus)}
        if before != input_hashes[key]:
            raise RuntimeError(f"input digest mismatch before {key}")
        started = time.time()
        receipt: dict[str, Any] = {
            "status": "started", "role": "baseline", "dataset": key, "run_id": args.run_id,
            "classification": "exposed development only; not fresh/holdout",
            "source_hashes_before": source_before, "minilm_hashes_before": model_before,
            "environment_before": env_before, "input_hashes_before": before,
            "uv_lock_sha256": sha256_file(lock_path), "scorer_sha256": sha256_file(repo / "evals" / "runner.py"),
            "answer_config_sha256": sha256_file(config_path), "model_manifest_sha256": sha256_file(model_manifest),
            "native_verdict": None, "fatal_error": None,
        }
        receipt_path = output / "run_receipt.json"
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        runtime_meta: dict[str, Any] = {
            "execution": "remote exposed-development baseline; CI environment is a separate cohort",
            "dataset": key, "source_commit": BASELINE_COMMIT,
            "candidate_commit_for_comparison": CANDIDATE_COMMIT,
            "answer_config": "answer-v3-frozen-v7.json",
            "answer_config_sha256": sha256_file(config_path),
            "model_manifest_sha256": sha256_file(model_manifest),
            "uv_lock_sha256": sha256_file(lock_path),
            "environment_classification": "new GitHub Actions ubuntu-24.04 environment; not asserted identical to local E3 environment",
            "engine_fingerprint": _engine_fingerprint(),
        }
        failure: BaseException | None = None
        result: dict[str, Any] | None = None
        try:
            result = evaluate_manifest(
                manifest, corpus, split="development", provider=provider,
                method=RetrievalMethod.HYBRID, runtime_metadata=runtime_meta,
                answer_provider=answer_runtime.answer_provider,
                extraction_provider=answer_runtime.extraction_provider,
            )
            (output / "raw_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            receipt["native_verdict"] = result.get("verdict")
            receipt["raw_results_file_sha256"] = sha256_file(output / "raw_results.json")
        except BaseException as exc:
            failure = exc
            receipt["fatal_error"] = {"type": type(exc).__name__, "message": str(exc),
                                      "traceback": traceback.format_exc()}
            (output / "execution_error.json").write_text(json.dumps({
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finally:
            try:
                after = {"evaluation_cases.json": sha256_file(manifest), "corpus_manifest.json": sha256_file(corpus)}
                source_after = {str(path): (tree_hash(path) if path.is_dir() else sha256_file(path)) for path in tracked}
                model_after = tree_hash(args.minilm_model_dir)
                env_after = environment_record(args.retriever_python_prefix, lock_path)
                receipt.update({
                    "completed_epoch": time.time(), "source_hashes_after": source_after,
                    "input_hashes_after": after, "all_input_hashes_unchanged": before == after == input_hashes[key],
                    "minilm_hashes_after": model_after, "environment_after": env_after,
                    "environment_hashes_unchanged": env_before == env_after,
                    "environment_note": "CI environment is distinct from the locked local environment; compare only within this remote cohort.",
                })
                receipt["integrity_unchanged"] = (source_before == source_after and model_before == model_after
                                                      and env_before == env_after and before == after == input_hashes[key])
                if not receipt["integrity_unchanged"]:
                    failure = failure or RuntimeError(f"baseline source/model/environment/input changed during {key}")
            except BaseException as exc:
                receipt["post_check_error"] = {"type": type(exc).__name__, "message": str(exc),
                                               "traceback": traceback.format_exc()}
                receipt["integrity_unchanged"] = False
                failure = failure or exc
            receipt["status"] = ("integrity_failed" if not receipt.get("integrity_unchanged")
                                 else "execution_failed" if failure is not None
                                 else "completed_exposed_development")
            receipt["completed_epoch"] = time.time()
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if failure:
            raise failure


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
