from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import run_ci_pair as pair
import run_generation_case as generation
from run_generation_case import DurableAnswerProvider


@dataclass
class FakeAnswer:
    status: str
    answer: str


class FakeProvider:
    def __init__(self, fails: bool = False) -> None:
        self.diagnostics: list[dict[str, str]] = []
        self.fails = fails

    def answer(self, question: str, ranked: object) -> FakeAnswer:
        self.diagnostics.append({"raw_response_body": "{\"answer\":\"recorded\"}"})
        if self.fails:
            raise RuntimeError("simulated provider error")
        return FakeAnswer("answered", "recorded")


class PrepOnlyTests(unittest.TestCase):
    def test_all_declared_ci_code_files_exist(self) -> None:
        missing = [name for name in pair.CI_CODE_FILES if not (pair.HERE / name).is_file()]
        self.assertEqual(missing, [])

    def test_no_go_refuses_before_paths_or_processes(self) -> None:
        args = pair.parse_args(["--dataset", "E4", "--repo-baseline", "missing-b",
                               "--repo-candidate", "missing-c", "--venv", "missing-v",
                               "--model-manifest", "missing-m", "--output", "unused"])
        with self.assertRaisesRegex(RuntimeError, "explicit --go"):
            pair.run(args)

    def test_only_three_exposed_datasets_are_cli_choices(self) -> None:
        with self.assertRaises(SystemExit):
            pair.parse_args(["--dataset", "fresh", "--repo-baseline", "b", "--repo-candidate", "c",
                             "--venv", "v", "--model-manifest", "m", "--output", "o", "--go"])

    def test_pinned_datasets_are_six_byte_exact_cc0_inputs(self) -> None:
        for dataset in pair.DATASETS:
            before = pair.verify_inputs(dataset, pair.HERE / "datasets")
            after = pair.verify_inputs(dataset, pair.HERE / "datasets")
            self.assertEqual(set(before), {"evaluation_cases.json", "corpus_manifest.json"})
            self.assertEqual(before, after)

    def test_common_blobs_must_be_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, candidate = root / "b", root / "c"
            for repo in (baseline, candidate):
                for name in pair.COMMON_BLOBS:
                    item = repo / name
                    item.parent.mkdir(parents=True, exist_ok=True)
                    item.write_bytes(b"frozen")
            result = pair.verify_common_blobs(baseline, candidate)
            self.assertEqual(len(result), len(pair.COMMON_BLOBS))
            (candidate / pair.COMMON_BLOBS[0]).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "common blob differs"):
                pair.verify_common_blobs(baseline, candidate)

    def test_cgroup_requires_exact_caps(self) -> None:
        self.assertEqual(pair.verify_resource_values(6442450944, 0, 400000, 100000, 10), pair.RESOURCE)
        with self.assertRaisesRegex(RuntimeError, "caps differ"):
            pair.verify_resource_values(8589934592, 0, 400000, 100000, 10)

    def test_memavailable_gate_rejects_below_eight_gib(self) -> None:
        self.assertEqual(pair.host_mem_available_bytes("MemAvailable: 8388608 kB\n"), 8 * 1024**3)
        with self.assertRaisesRegex(RuntimeError, "below 8 GiB"):
            pair.host_mem_available_bytes("MemAvailable: 8388607 kB\n")

    def test_artifact_manifest_pin_matches_preparation_sha(self) -> None:
        self.assertEqual(pair.sha256_file(pair.HERE / "ARTIFACT_MANIFEST.json"), pair.ARTIFACT_SHA256)

    def test_server_identity_ignores_only_model_created_timestamp(self) -> None:
        import hashlib
        import json

        def sample(created: int, *, model: str = "Qwen3-4B-Q4_K_M.gguf", size: int = 123,
                   context: int = 6144) -> dict[str, object]:
            body = {"object": "list", "data": [{"id": model, "object": "model", "created": created,
                                                   "owned_by": "llama.cpp", "meta": {"size": size,
                                                                                       "context_length": context}}]}
            raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
            return {"/health": {"body": {"status": "ok"}},
                    "/v1/models": {"body": body, "body_sha256": hashlib.sha256(raw).hexdigest(),
                                    "raw_body_base64": __import__("base64").b64encode(raw).decode("ascii")},
                    "/props": {"body": {"model_alias": model}}}

        before, after = sample(100), sample(200)
        self.assertNotEqual(before["/v1/models"]["body_sha256"], after["/v1/models"]["body_sha256"])
        self.assertNotEqual(before["/v1/models"]["raw_body_base64"], after["/v1/models"]["raw_body_base64"])
        self.assertEqual(pair.static_server_identity_sha256(before), pair.static_server_identity_sha256(after))
        self.assertTrue(pair.same_static_server_identity(before, after))
        changed_model = sample(100, model="wrong-model")
        changed_size = sample(100, size=124)
        changed_config = sample(100, context=8192)
        self.assertFalse(pair.same_static_server_identity(before, changed_model))
        self.assertFalse(pair.same_static_server_identity(before, changed_size))
        self.assertFalse(pair.same_static_server_identity(before, changed_config))
        with self.assertRaisesRegex(RuntimeError, "model ID differs"):
            pair.validate_server_identity(changed_model, "Qwen3-4B-Q4_K_M.gguf")

        base_snapshot = {key: {"fixed": key} for key in
                         ("source", "inputs", "model", "runtime_archive", "server_binary",
                          "minilm", "environment", "config")}
        pre = {**base_snapshot, "server": before,
               "server_identity_sha256": pair.static_server_identity_sha256(before)}
        post_timestamp = {**base_snapshot, "server": after,
                          "server_identity_sha256": pair.static_server_identity_sha256(after)}
        self.assertTrue(generation.snapshots_unchanged(pre, post_timestamp))
        for changed in (changed_model, changed_size, changed_config):
            post_changed = {**base_snapshot, "server": changed,
                            "server_identity_sha256": pair.static_server_identity_sha256(changed)}
            self.assertFalse(generation.snapshots_unchanged(pre, post_changed))

    def test_provider_runtime_validation_errors_block_pair_success(self) -> None:
        good = {"status": "completed_exposed_development", "model_validation_error_count": 0,
                "backend_error_count": 0}
        pair.validate_generation_receipt(good)
        for bad in (
            {**good, "model_validation_error_count": 1},
            {**good, "backend_error_count": 1},
            {**good, "status": "technical_failure"},
        ):
            with self.assertRaisesRegex(RuntimeError, "technical/provider"):
                pair.validate_generation_receipt(bad)

    def test_unexpected_exit_or_forced_server_kill_is_failure(self) -> None:
        self.assertIsNone(pair.server_lifecycle_error(None, -15, False))
        self.assertIsNone(pair.server_lifecycle_error(None, 0, False))
        self.assertIn("unexpectedly", pair.server_lifecycle_error(1, 1, False))
        self.assertIn("forced kill", pair.server_lifecycle_error(None, -9, True))
        self.assertIn("unexpected status", pair.server_lifecycle_error(None, 3, False))

    def test_answer_call_is_durable_on_return_and_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            success_dir = root / "success"
            failure_dir = root / "failure"
            success_dir.mkdir()
            failure_dir.mkdir()
            provider = FakeProvider()
            durable = DurableAnswerProvider(provider, success_dir)
            result = durable.answer("test question", [])
            self.assertEqual(result.answer, "recorded")
            saved = __import__("json").loads((success_dir / "answer-001.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["provider_diagnostic_present"])
            self.assertEqual(saved["response"]["answer"], "recorded")

            provider = FakeProvider(fails=True)
            durable = DurableAnswerProvider(provider, failure_dir)
            with self.assertRaisesRegex(RuntimeError, "simulated provider error"):
                durable.answer("failing question", [])
            failure_file = __import__("json").loads((failure_dir / "answer-001.json").read_text(encoding="utf-8"))
            self.assertEqual(failure_file["exception"]["type"], "RuntimeError")
            self.assertTrue(failure_file["provider_diagnostic_present"])
            self.assertEqual(list(root.rglob("*.partial")), [])


if __name__ == "__main__":
    unittest.main()
