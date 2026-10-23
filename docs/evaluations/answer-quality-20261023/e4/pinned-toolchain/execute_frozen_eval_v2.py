"""Run an unchanged adapter once in a new, role-isolated output root.

Orchestration only. Freeze this wrapper and its registration before E4 inference.
"""
from pathlib import Path
import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess

parser = argparse.ArgumentParser()
for field in ("inputs", "repo", "python", "model-dir", "run-root", "registration"):
    parser.add_argument("--" + field, type=Path, required=True)
parser.add_argument("--commit", required=True)
parser.add_argument("--role", choices=("baseline", "candidate"), required=True)
parser.add_argument("--registration-sha256", required=True)
args = parser.parse_args()

def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()

def write_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

def git(*commands: str) -> str:
    return subprocess.check_output(["git", "-C", str(args.repo), *commands], text=True).strip()

def exact_engine_fingerprint() -> str:
    source = (args.repo / "evals/runner.py").read_text(encoding="utf-8")
    assignments = [entry for entry in ast.parse(source).body if isinstance(entry, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "ENGINE_FINGERPRINT_PATHS"
                           for t in entry.targets)]
    sha = hashlib.sha256()
    for relative in ast.literal_eval(assignments[0].value):
        blob = subprocess.check_output(["git", "-C", str(args.repo), "show", "HEAD:" + relative])
        assert (args.repo / relative).read_bytes() == blob, "Noncanonical working-tree bytes: " + relative
        sha.update(relative.encode("utf-8") + b"\0" + blob + b"\0")
    return sha.hexdigest()

def verify_model(manifest: dict) -> list[dict]:
    verified = []
    for item in manifest["files"]:
        path = (args.model_dir / item["path"]).resolve()
        assert path.is_relative_to(args.model_dir.resolve())
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
        verified.append({**item, "verified": True})
    return verified

def probe_environment() -> dict:
    probe = (
        "import json,platform,sys,importlib.metadata as m;"
        "print(json.dumps({'platform':sys.platform,'implementation':sys.implementation.name,"
        "'version':platform.python_version(),'distributions':sorted("
        "(d.metadata.get('Name','').lower().replace('_','-'),d.version) for d in m.distributions())}))"
    )
    return json.loads(subprocess.check_output([str(args.python), "-I", "-c", probe], text=True))

def execute() -> int:
    assert digest(args.registration) == args.registration_sha256, "Registration changed."
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    assert registration["commits"][args.role] == args.commit, "Role/commit mismatch."
    assert digest(Path(__file__)) == registration["wrapper_sha256"], "Wrapper changed."
    head = git("rev-parse", "HEAD")
    assert head == args.commit and not git("status", "--porcelain"), "Source is not clean at frozen commit."
    environment = probe_environment()
    assert {k: environment[k] for k in ("platform", "implementation", "version")} == {
        "platform": "linux", "implementation": "cpython", "version": "3.12.14"}
    inventory_sha = hashlib.sha256(json.dumps(environment["distributions"], ensure_ascii=True,
                                              separators=(",", ":")).encode("utf-8")).hexdigest()
    assert inventory_sha == registration["environment_inventory_sha256"], "Installed dependencies changed."
    for relative, expected in registration["inputs_sha256"].items():
        path = (args.inputs / relative).resolve()
        assert path.is_relative_to(args.inputs.resolve()) and digest(path) == expected, "Frozen input changed."
    for relative, expected in registration["support_files_sha256"].items():
        assert digest(args.repo / relative) == expected, "Scorer/config/support file changed: " + relative
    expected_engine = exact_engine_fingerprint()
    manifest_path = args.repo / "infra/models/paraphrase-multilingual-minilm-l12-v2-text-onnx.json"
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified_model = verify_model(model_manifest)
    shutil.copytree(args.inputs / "datasets", args.run_root / "datasets")
    (args.run_root / "adapters").mkdir()
    copied_adapter = args.run_root / "adapters/run_baseline.py"
    shutil.copyfile(args.inputs / "adapters/run_baseline.py", copied_adapter)
    (args.run_root / "baseline-run").mkdir()
    copied = {p.relative_to(args.run_root).as_posix(): digest(p)
              for folder in ("datasets", "adapters")
              for p in sorted((args.run_root / folder).rglob("*")) if p.is_file()}
    assert all(digest(args.inputs / relative) == sha for relative, sha in copied.items())
    write_new(args.run_root / "START_RECEIPT.json", {
        "role": args.role, "commit": head, "tree": git("rev-parse", "HEAD^{tree}"),
        "started_at_utc": now(), "environment": environment,
        "environment_inventory_sha256": inventory_sha,
        "registration_sha256": args.registration_sha256,
        "orchestration_wrapper_sha256": digest(Path(__file__)),
        "input_files_sha256": copied, "expected_engine_fingerprint": expected_engine,
        "verified_model_files": verified_model,
        "output_name_is_fixed_by_unchanged_adapter": "baseline-run/raw_results.json"})
    env = dict(os.environ, EVIDENCEDESK_REPO=str(args.repo),
               MINILM_MODEL_DIR=str(args.model_dir), PYTHONHASHSEED="0")
    with (args.run_root / "stdout.txt").open("xb") as stdout, (args.run_root / "stderr.txt").open("xb") as stderr:
        completed = subprocess.run([str(args.python), str(copied_adapter)], cwd=args.repo,
                                   env=env, stdout=stdout, stderr=stderr, check=False)
    final = {"role": args.role, "commit": head, "finished_at_utc": now(),
             "status": "completed" if completed.returncode == 0 else "process_failed",
             "returncode": completed.returncode,
             "stdout_sha256": digest(args.run_root / "stdout.txt"),
             "stderr_sha256": digest(args.run_root / "stderr.txt")}
    raw_path = args.run_root / "baseline-run/raw_results.json"
    if raw_path.is_file():
        final["raw_results_sha256"] = digest(raw_path)
        envelope = json.loads(raw_path.read_text(encoding="utf-8"))
        assert envelope["suite_hash"] == copied["datasets/evaluation_cases.json"]
        assert envelope["corpus_hash"] == copied["datasets/corpus_manifest.json"]
        assert envelope["engine_fingerprint"] == expected_engine
        final["metric_counts"] = envelope["result"]["metric_counts"]
        final["engine_fingerprint"] = envelope["engine_fingerprint"]
    else:
        assert completed.returncode != 0, "Adapter succeeded without raw output."
    assert git("rev-parse", "HEAD") == head and not git("status", "--porcelain")
    assert all(digest(args.run_root / relative) == sha for relative, sha in copied.items())
    verify_model(model_manifest)
    assert probe_environment() == environment, "Evaluation environment changed during inference."
    write_new(args.run_root / "FINISH_RECEIPT.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return completed.returncode

# Never enter the finally/error path for a pre-existing output root.
args.run_root.mkdir(parents=False, exist_ok=False)
write_new(args.run_root / "ATTEMPT.json", {"role": args.role, "claimed_commit": args.commit,
          "started_at_utc": now(), "claimed_registration_sha256": args.registration_sha256})
try:
    exit_code = execute()
except BaseException as error:
    if not (args.run_root / "FINISH_RECEIPT.json").exists():
        write_new(args.run_root / "FINISH_RECEIPT.json", {
            "status": "failed", "role": args.role, "claimed_commit": args.commit,
            "finished_at_utc": now(), "error_type": type(error).__name__, "error": str(error),
            "preserved_files_sha256": {p.relative_to(args.run_root).as_posix(): digest(p)
                                       for p in args.run_root.rglob("*") if p.is_file()}})
    raise
raise SystemExit(exit_code)
