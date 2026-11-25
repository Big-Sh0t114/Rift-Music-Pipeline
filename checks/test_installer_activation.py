from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from music_runtime import installer as installer_module
from music_runtime.settings import PipelineError


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


class InstallerActivationTests(unittest.TestCase):
    def test_initial_rename_failure_preserves_original_install(self):
        with tempfile.TemporaryDirectory(prefix="mus-library-activation-") as folder:
            runtime_dir = Path(folder) / "runtime"
            target = runtime_dir / "envs" / "audio"
            stage = runtime_dir / "envs" / ".audio.stage-test"
            (target / "Scripts").mkdir(parents=True)
            (target / "Scripts" / "python.exe").write_bytes(b"original interpreter bytes")
            (target / "install.marker").write_bytes(b"original installation record")
            stage.mkdir(parents=True)
            (stage / "new.marker").write_bytes(b"staged replacement")
            original = _snapshot(target)
            installer = installer_module.RuntimeInstaller(
                {"runtime_dir": runtime_dir, "runtime": {"device": "auto"}}
            )

            with patch.object(installer_module.os, "replace", side_effect=OSError("simulated first rename failure")) as replace:
                with self.assertRaises(PipelineError):
                    installer._activate_directory(stage, target)

            self.assertEqual(replace.call_count, 1)
            self.assertEqual(_snapshot(target), original)
            self.assertEqual((stage / "new.marker").read_bytes(), b"staged replacement")
            self.assertFalse(list(target.parent.glob(".audio.old-*")))

    def test_invalid_activated_stage_restores_original_install(self):
        with tempfile.TemporaryDirectory(prefix="mus-library-activation-rollback-") as folder:
            runtime_dir = Path(folder) / "runtime"
            target = runtime_dir / "envs" / "rift"
            stage = runtime_dir / "envs" / ".rift.stage-test"
            target.mkdir(parents=True)
            (target / "checkpoint.txt").write_bytes(b"original install bytes")
            stage.mkdir(parents=True)
            (stage / "checkpoint.txt").write_bytes(b"invalid staged bytes")
            original = _snapshot(target)
            installer = installer_module.RuntimeInstaller(
                {"runtime_dir": runtime_dir, "runtime": {"device": "auto"}}
            )

            with self.assertRaises(PipelineError):
                installer._activate_directory(stage, target, validator=lambda _directory: False)

            self.assertEqual(_snapshot(target), original)
            self.assertFalse(stage.exists())
            self.assertFalse(list(target.parent.glob(".rift.old-*")))
