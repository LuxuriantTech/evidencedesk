from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.local_generation.http_qualification import (
    prepare_run,
    safe_remove_tree,
    summarize_receipts,
    verify_container_limits,
)


class PreparationHelperTests(unittest.TestCase):
    def test_prepare_run_creates_unique_fake_only_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "candidate"
            runner = root / "runner"
            candidate.mkdir()
            runner.mkdir()
            template = candidate / "compose.yaml"
            template.write_text("services: {}\n", encoding="utf-8")
            (candidate / "seed_http_smoke.py").write_text(
                "# synthetic identity helper\n", encoding="utf-8"
            )
            envfile = root / "github-env"
            envfile.write_text("EXISTING=preserved\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "GITHUB_RUN_ID": "12345",
                    "GITHUB_RUN_ATTEMPT": "2",
                    "GITHUB_ENV": str(envfile),
                },
            ):
                with patch(
                    "sys.argv",
                    [
                        "prepare_run",
                        "--candidate-root",
                        str(candidate),
                        "--runner-temp",
                        str(runner),
                        "--compose-template",
                        "compose.yaml",
                    ],
                ):
                    self.assertEqual(prepare_run.main(), 0)
            text = envfile.read_text(encoding="utf-8")
            self.assertIn("COMPOSE_PROJECT_NAME=evidencedesk-httpq-12345-2", text)
            self.assertIn("EXISTING=preserved\n", text)
            self.assertIn("SMOKE_DOSSIER_A=", text)
            self.assertNotIn("EvidenceDemo", text)
            self.assertFalse((runner / "evidencedesk-httpq-12345-2" / "bundle").exists())
            override = runner / "evidencedesk-httpq-12345-2" / "compose.override.yaml"
            self.assertEqual(override.read_text(encoding="utf-8"), "services: {}\n")
            self.assertTrue(
                (runner / "evidencedesk-httpq-12345-2" / "http_qualification_seed.py").exists()
            )

    def test_prepare_run_rolls_back_its_new_directory_when_copy_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, runner = root / "candidate", root / "runner"
            candidate.mkdir()
            runner.mkdir()
            (candidate / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
            (candidate / "seed_http_smoke.py").write_text("# helper\n", encoding="utf-8")
            envfile = root / "github-env"
            envfile.write_text("EXISTING=preserved\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "GITHUB_RUN_ID": "54321",
                    "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_ENV": str(envfile),
                },
            ):
                with patch(
                    "sys.argv",
                    [
                        "prepare_run",
                        "--candidate-root",
                        str(candidate),
                        "--runner-temp",
                        str(runner),
                        "--compose-template",
                        "compose.yaml",
                    ],
                ):
                    with patch(
                        "scripts.local_generation.http_qualification.prepare_run.shutil.copyfile",
                        side_effect=OSError("private path"),
                    ):
                        with self.assertRaisesRegex(SystemExit, "setup_failed"):
                            prepare_run.main()
            self.assertFalse((runner / "evidencedesk-httpq-54321-1").exists())
            self.assertEqual(envfile.read_text(encoding="utf-8"), "EXISTING=preserved\n")

    def test_prepare_run_checks_github_env_before_creating_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, runner = root / "candidate", root / "runner"
            candidate.mkdir()
            runner.mkdir()
            (candidate / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
            (candidate / "seed_http_smoke.py").write_text("# helper\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"GITHUB_RUN_ID": "98765", "GITHUB_RUN_ATTEMPT": "1"},
                clear=True,
            ):
                with patch(
                    "sys.argv",
                    [
                        "prepare_run",
                        "--candidate-root",
                        str(candidate),
                        "--runner-temp",
                        str(runner),
                        "--compose-template",
                        "compose.yaml",
                    ],
                ):
                    with self.assertRaises(SystemExit):
                        prepare_run.main()
            self.assertFalse((runner / "evidencedesk-httpq-98765-1").exists())

    def test_cleanup_rejects_directory_outside_runner_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner, outside = root / "runner", root / "outside"
            runner.mkdir()
            outside.mkdir()
            with (
                patch.dict(os.environ, {"RUNNER_TEMP": str(runner)}),
                patch("sys.argv", ["safe_remove_tree", str(outside)]),
            ):
                with self.assertRaises(SystemExit):
                    safe_remove_tree.main()
            self.assertTrue(outside.exists())

    def test_limit_check_rejects_wrong_project_before_docker(self) -> None:
        with (
            patch.dict(os.environ, {"GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1"}),
            patch(
                "sys.argv",
                [
                    "verify_container_limits",
                    "--compose-project",
                    "other",
                    "--compose-files",
                    "compose.yaml",
                ],
            ),
        ):
            with patch("subprocess.check_output") as call:
                with self.assertRaises(SystemExit):
                    verify_container_limits.main()
                call.assert_not_called()

    def test_safe_summary_uses_only_statuses_and_resource_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            available_checks = sorted(summarize_receipts.EXPECTED_CHECKS["available"])
            unavailable_checks = sorted(summarize_receipts.EXPECTED_CHECKS["unavailable"])
            (root / "available.json").write_text(
                json.dumps(
                    {
                        "phase": "available",
                        "checks": available_checks,
                        "documents": {"a": "private-id"},
                    }
                ),
                encoding="utf-8",
            )
            (root / "unavailable.json").write_text(
                json.dumps(
                    {
                        "phase": "unavailable",
                        "checks": unavailable_checks,
                        "answer": "private",
                    }
                ),
                encoding="utf-8",
            )
            (root / "memory.json").write_text(
                json.dumps(
                    {
                        "outcome": "stopped_cleanly",
                        "limit_bytes": 10,
                        "host_high_water_used_bytes": 9,
                    }
                ),
                encoding="utf-8",
            )
            out = root / "summary.json"
            argv = [
                "summary",
                "--http-receipt",
                str(root / "available.json"),
                "--http-unavailable",
                str(root / "unavailable.json"),
                "--memory-receipt",
                str(root / "memory.json"),
                "--output",
                str(out),
            ]
            with patch("sys.argv", argv):
                self.assertEqual(summarize_receipts.main(), 0)
            content = out.read_text(encoding="utf-8")
            self.assertNotIn("private", content)
            self.assertIn('"available_http": "pass"', content)

    def test_summary_rejects_incomplete_receipt_checks(self) -> None:
        self.assertEqual(
            summarize_receipts.phase_status(Path("missing"), "available"),
            "not_recorded",
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "incomplete.json"
            path.write_text(json.dumps({"phase": "available", "checks": ["ok"]}), encoding="utf-8")
            self.assertEqual(summarize_receipts.phase_status(path, "available"), "invalid")


if __name__ == "__main__":
    unittest.main()
