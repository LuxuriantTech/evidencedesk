"""Launcher command contract tests; never execute a server binary."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.local_generation import bundle, verify_and_run


class VerifyAndRunTests(unittest.TestCase):
    def test_cache_ram_flag_comes_from_manifest_without_exec(self) -> None:
        server = bundle.MANIFEST["server"]
        self.assertEqual(server["cache_ram_mib"], 0)

        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.dict(os.environ, {"EVIDENCEDESK_8B_BUNDLE": temporary}),
                patch.object(verify_and_run, "verify_bundle") as verify_bundle,
                patch.object(os, "execv") as execv,
            ):
                self.assertEqual(verify_and_run.main(), 127)

        verify_bundle.assert_called_once_with(Path(temporary))
        command = execv.call_args.args[1]
        cache_flag_index = command.index("--cache-ram")
        self.assertEqual(command[cache_flag_index + 1], str(server["cache_ram_mib"]))
        self.assertEqual(command.count("--cache-ram"), 1)


if __name__ == "__main__":
    unittest.main()
