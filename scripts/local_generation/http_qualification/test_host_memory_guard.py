import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.local_generation.http_qualification.host_memory_guard import PROJECT_RE, stop_project


class HostGuardTests(unittest.TestCase):
    def test_project_prefix_is_exactly_scoped(self) -> None:
        self.assertTrue(PROJECT_RE.fullmatch("evidencedesk-httpq-123456-2"))
        self.assertFalse(PROJECT_RE.fullmatch("evidencedesk-httpq-123456-2-extra"))
        self.assertFalse(PROJECT_RE.fullmatch("production"))

    def test_stop_targets_only_the_named_smoke_project_services(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            compose_file = Path(directory) / "compose.yaml"
            compose_file.write_text("services: {}\n", encoding="utf-8")
            with patch(
                "scripts.local_generation.http_qualification.host_memory_guard.subprocess.run"
            ) as run:
                run.return_value.returncode = 0
                self.assertTrue(stop_project("evidencedesk-httpq-123456-2", [compose_file]))
                argv = run.call_args.args[0]
                self.assertIn("--project-name", argv)
                self.assertIn("evidencedesk-httpq-123456-2", argv)
                self.assertEqual(
                    argv[-6:],
                    ["stop", "api", "local-8b", "worker", "postgres", "redis"],
                )

    def test_stop_refuses_non_scoped_project_before_docker(self) -> None:
        with patch(
            "scripts.local_generation.http_qualification.host_memory_guard.subprocess.run"
        ) as run:
            with self.assertRaises(RuntimeError):
                stop_project("production", [])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
