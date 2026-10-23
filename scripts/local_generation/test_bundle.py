"""Fast stdlib-only checks for bundle integrity and unsafe archive rejection."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.local_generation import bundle


class BundleTests(unittest.TestCase):
    def test_manifest_is_exactly_the_8b_local_pilot(self) -> None:
        model = bundle.MANIFEST["model"]
        runtime = bundle.MANIFEST["runtime"]
        server = bundle.MANIFEST["server"]
        self.assertEqual(model["id"], "Qwen3-8B-Q4_K_M.gguf")
        self.assertEqual(
            model["sha256"],
            "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
        )
        self.assertEqual(runtime["version"], "b11798")
        self.assertEqual(
            runtime["archive_sha256"],
            "9342761b01a8d82e0567c9673d3dd8f6264964e0ccb27a008193853c757cf563",
        )
        self.assertEqual(server["host"], "127.0.0.1")
        self.assertEqual(server["port"], 50193)
        self.assertEqual(bundle.MANIFEST["resources"]["memory_max_bytes"], 10 * 1024**3)
        self.assertNotIn("4B", json.dumps(bundle.MANIFEST))

    def _fixture(self, root: Path) -> tuple[Path, bytes, bytes]:
        model_bytes = b"synthetic model fixture"
        binary_bytes = b"synthetic server fixture"
        runtime = root / "runtime"
        model_dir = root / "models"
        runtime.mkdir(parents=True)
        model_dir.mkdir()
        (root / "manifest.json").write_bytes(bundle.MANIFEST_PATH.read_bytes())
        (root / "licenses").mkdir()
        (root / "licenses" / "Apache-2.0.txt").write_text("Apache License Version 2.0")
        (root / "licenses" / "MIT.txt").write_text(
            "Permission is hereby granted. THE SOFTWARE IS PROVIDED"
        )
        (model_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(model_bytes)
        archive_path = runtime / "llama-runtime.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            root_member = tarfile.TarInfo("llama-b11798")
            root_member.type = tarfile.DIRTYPE
            root_member.mode = 0o755
            archive.addfile(root_member)
            for name, content in (
                ("llama-b11798/llama-server", binary_bytes),
                ("llama-b11798/libtest.so.0.0", b"synthetic shared library"),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(content)
                member.mode = 0o755
                archive.addfile(member, io.BytesIO(content))
            for name, target in (
                ("llama-b11798/libtest.so.0", "libtest.so.0.0"),
                ("llama-b11798/libtest.so", "libtest.so.0"),
            ):
                member = tarfile.TarInfo(name)
                member.type = tarfile.SYMTYPE
                member.linkname = target
                member.mode = 0o777
                archive.addfile(member)
        bundle._safe_extract(archive_path, runtime)
        original = bundle.MANIFEST
        bundle.MANIFEST = {
            "model": {
                "id": "Qwen3-8B-Q4_K_M.gguf",
                "bytes": len(model_bytes),
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
                "license_file": "licenses/Apache-2.0.txt",
            },
            "runtime": {
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": bundle.sha256_file(archive_path)[1],
                "server_binary": "llama-b11798/llama-server",
                "server_binary_sha256": hashlib.sha256(binary_bytes).hexdigest(),
                "license_file": "licenses/MIT.txt",
            },
        }
        self.addCleanup(setattr, bundle, "MANIFEST", original)
        return root, model_bytes, binary_bytes

    def test_valid_synthetic_bundle_checks_all_files_and_materialized_link_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _ = self._fixture(Path(temporary))
            bundle.verify_bundle(root)

    def test_tampered_model_is_rejected_before_runtime_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _ = self._fixture(Path(temporary))
            (root / "models" / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"tampered")
            with self.assertRaisesRegex(bundle.BundleError, "integrity mismatch"):
                bundle.verify_bundle(root)

    def test_tampered_archive_is_rejected_even_if_extracted_binary_is_intact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _ = self._fixture(Path(temporary))
            with (root / "runtime" / "llama-runtime.tar.gz").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(bundle.BundleError, "integrity mismatch"):
                bundle.verify_bundle(root)

    def test_mutated_shared_library_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _ = self._fixture(Path(temporary))
            library = root / "runtime" / "llama-b11798" / "libtest.so"
            library.write_bytes(b"mutated library")
            with self.assertRaisesRegex(bundle.BundleError, "runtime file integrity mismatch"):
                bundle.verify_bundle(root)

    def test_unexpected_runtime_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _ = self._fixture(Path(temporary))
            (root / "runtime" / "llama-b11798" / "unexpected.so").write_bytes(b"extra")
            with self.assertRaisesRegex(bundle.BundleError, "missing or unexpected paths"):
                bundle.verify_bundle(root)

    def test_archive_traversal_and_symlink_entries_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, kind in (
                ("../escape", "file"),
                ("C:/escape", "drive"),
                (r"folder\..\escape", "backslash"),
                ("llama-b11798/link", "symlink"),
                ("llama-b11798/escape", "escape"),
            ):
                archive_path = root / f"{kind}.tar.gz"
                with tarfile.open(archive_path, "w:gz") as archive:
                    member = tarfile.TarInfo(name)
                    member.type = (
                        tarfile.SYMTYPE if kind in {"symlink", "escape"} else tarfile.REGTYPE
                    )
                    member.linkname = (
                        "../../escape" if kind == "escape" else "target"
                    ) if member.issym() else ""
                    archive.addfile(member)
                destination = root / f"out-{kind}"
                destination.mkdir()
                with self.assertRaises(bundle.BundleError):
                    bundle._safe_extract(archive_path, destination)

    def test_symlinked_model_parent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _ = self._fixture(Path(temporary) / "bundle")
            model_directory = root / "models"
            redirected = root / "redirected-models"
            model_directory.rename(redirected)
            model_directory.symlink_to(redirected, target_is_directory=True)
            with self.assertRaisesRegex(bundle.BundleError, "contains a symlink"):
                bundle.verify_bundle(root)

    def test_prepare_refuses_to_overwrite_an_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "exists"
            target.mkdir()
            with self.assertRaisesRegex(bundle.BundleError, "must not already exist"):
                bundle.prepare_bundle(target)

    def test_actual_runtime_archive_when_explicitly_configured(self) -> None:
        configured = os.environ.get("EVIDENCEDESK_RUNTIME_ARCHIVE_TEST_PATH")
        if not configured:
            self.skipTest("set EVIDENCEDESK_RUNTIME_ARCHIVE_TEST_PATH for cache validation")
        archive_path = Path(configured)
        self.assertEqual(bundle.verify_runtime_archive(archive_path), 61)

    def test_runtime_only_staging_extracts_no_model_and_checks_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_bundle, _, _ = self._fixture(root / "source")
            archive_path = source_bundle / "runtime" / "llama-runtime.tar.gz"
            output = root / "runtime-only"
            self.assertEqual(bundle.stage_runtime_archive(archive_path, output), 4)
            self.assertEqual(
                sorted(path.relative_to(output).as_posix() for path in output.rglob("*")),
                [
                    "llama-runtime.tar.gz",
                    "runtime",
                    "runtime/llama-b11798",
                    "runtime/llama-b11798/libtest.so",
                    "runtime/llama-b11798/libtest.so.0",
                    "runtime/llama-b11798/libtest.so.0.0",
                    "runtime/llama-b11798/llama-server",
                ],
            )
            bundle.verify_runtime_tree(
                output / "runtime", output / "llama-runtime.tar.gz", "llama-b11798/llama-server"
            )

    def test_runtime_staging_does_not_remove_directory_created_during_mkdir_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_bundle, _, _ = self._fixture(root / "source")
            archive_path = source_bundle / "runtime" / "llama-runtime.tar.gz"
            output = root / "raced-output"
            original_mkdir = Path.mkdir

            def create_foreign_directory_then_race(
                path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
            ) -> None:
                if path == output:
                    original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)
                    (path / "foreign-marker").write_text("owned by another process")
                    raise FileExistsError(path)
                original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

            with (
                patch.object(Path, "mkdir", create_foreign_directory_then_race),
                patch.object(shutil, "rmtree") as remove_tree,
            ):
                with self.assertRaisesRegex(bundle.BundleError, "must not already exist"):
                    bundle.stage_runtime_archive(archive_path, output)
            self.assertEqual((output / "foreign-marker").read_text(), "owned by another process")
            remove_tree.assert_not_called()


if __name__ == "__main__":
    unittest.main()
