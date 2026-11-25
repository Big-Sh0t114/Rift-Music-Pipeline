from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from music_runtime.installer import RuntimeInstaller


class ManagedPythonBootstrapIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_uv_python_install_disables_global_bin_shims(self):
        with tempfile.TemporaryDirectory(prefix="mus-library-python-bootstrap-") as folder:
            runtime_dir = Path(folder) / "runtime"
            candidate = runtime_dir / "tools" / "python" / "python.exe"
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(b"mock python executable")
            installer = RuntimeInstaller({"runtime_dir": runtime_dir, "runtime": {"device": "auto"}})

            async def fake_capture(command, **_kwargs):
                args = [str(value) for value in command]
                if args == ["uv-fixture", "--version"]:
                    return subprocess.CompletedProcess(args, 0, "uv 0.9.5 (fixture)\n", "")
                if args[:4] == ["uv-fixture", "python", "find", "3.11.14"]:
                    return subprocess.CompletedProcess(args, 0, f"{candidate}\n", "")
                if args == [str(candidate), "--version"]:
                    return subprocess.CompletedProcess(args, 0, "Python 3.11.14\n", "")
                self.fail(f"unexpected uv probe: {args!r}")

            run = AsyncMock(return_value=subprocess.CompletedProcess([], 0, "", ""))
            with (
                patch.object(installer, "_uv_exe", return_value="uv-fixture"),
                patch.object(installer, "_run_capture", new=AsyncMock(side_effect=fake_capture)),
                patch.object(installer, "_run", new=run),
            ):
                result = await installer._uv_python()

            self.assertEqual(result, candidate.resolve())
            install_command = run.await_args.args[0]
            self.assertIn("--no-registry", install_command)
            self.assertIn("--no-bin", install_command)
            self.assertEqual(install_command[-2:], ["--no-registry", "--no-bin"])
