from __future__ import annotations
import hashlib, json, os, sys, subprocess
from pathlib import Path

base = Path(__file__).resolve().parents[1]
repo = Path(os.environ["EVIDENCEDESK_REPO"])
sys.path.insert(0, str(repo / "apps" / "api"))
sys.path.insert(0, str(repo))
from evidencedesk_api.providers import LocalSemanticEmbeddingProvider
from evidencedesk_api.retrieval import RetrievalMethod
from evals.answer_v3_runtime import build_frozen_answer_runtime
from evals.runner import evaluate_manifest, _engine_fingerprint

manifest = base / "datasets" / "evaluation_cases.json"
corpus = base / "datasets" / "corpus_manifest.json"
model_dir = Path(os.environ["MINILM_MODEL_DIR"])
model_manifest = repo / "infra" / "models" / "paraphrase-multilingual-minilm-l12-v2-text-onnx.json"
freeze_path = repo / "evals" / "configs" / "answer-v3-frozen-v7.json"
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
provider = LocalSemanticEmbeddingProvider(manifest_path=model_manifest, model_path=model_dir)
runtime = build_frozen_answer_runtime(freeze, provider)
result = evaluate_manifest(
    manifest, corpus, split="development", provider=provider,
    method=RetrievalMethod.HYBRID,
    runtime_metadata={"execution": "external custom development suite via common adapter", "engine_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(), "answer_config": "answer-v3-frozen-v7.json", "model_manifest_sha256": hashlib.sha256(model_manifest.read_bytes()).hexdigest()},
    answer_provider=runtime.answer_provider,
    extraction_provider=runtime.extraction_provider,
)
case_data = json.loads(manifest.read_text(encoding="utf-8"))
by_id = {c["id"]: c for c in case_data["cases"]}
# Add protocol-defined exact outcomes without changing runner-produced values.
for row in result["cases"]:
    gold = by_id[row["id"]]
    kind = gold["kind"]
    if kind == "answerable":
        semantic_negation = gold.get("category") == "negation" and any(row.get("citation_matches", []))
        row["protocol_outcome"] = "correct" if row.get("status") == "answered" and (row.get("answer_match") or semantic_negation) and any(row.get("citation_matches", [])) else "wrong"
    elif kind == "unanswerable":
        row["protocol_outcome"] = "correct_abstention" if row.get("status") == "abstained" and not row.get("citations") else "wrong_or_hallucinated"
    else:
        row["protocol_outcome"] = "correct_ambiguity" if row.get("status") == "ambiguous" else "wrong_or_overconfident"
    row["gold_kind"] = kind
    row["gold_answer"] = gold["expected_answer"]
    row["gold_citations"] = gold["expected_citations"]
summary = {
    "suite_hash": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "corpus_hash": hashlib.sha256(corpus.read_bytes()).hexdigest(),
    "engine_fingerprint": _engine_fingerprint(),
    "result": result,
}
out = base / "baseline-run" / "raw_results.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "engine_fingerprint": summary["engine_fingerprint"], "case_count": result["case_count"], "metric_counts": result["metric_counts"], "metrics": {k: result[k] for k in ("citation_precision", "citation_recall", "citation_case_accuracy", "retrieval_recall_at_5", "retrieval_mrr_at_5", "abstention_accuracy", "error_rate", "schema_error_rate", "latency_median_ms", "latency_p95_ms", "indexing_ms")}}, ensure_ascii=False, indent=2))




