from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from music_runtime import installer as installer_module
from music_runtime.installer import PLUGIN_DIR, RuntimeInstaller
from music_runtime.runtime import MusicRuntime
from music_runtime.settings import load_settings


class AutoDeviceReadinessTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self, root: Path, configured_device: str = "auto") -> dict:
        plugin_dir = root / "plugin"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        settings = load_settings(plugin_dir)
        runtime_dir = root / "runtime"
        settings["plugin_dir"] = plugin_dir
        settings["runtime_dir"] = runtime_dir
        settings["runtime"]["device"] = configured_device
        settings["ffmpeg_path"] = runtime_dir / "models" / "tools" / "ffmpeg.exe"
        return settings

    @staticmethod
    def _write_audio_assets(installer: RuntimeInstaller, resolved_device: str) -> None:
        runtime_dir = installer.runtime_dir
        audio_python = runtime_dir / "envs" / "audio" / "Scripts" / "python.exe"
        audio_python.parent.mkdir(parents=True, exist_ok=True)
        audio_python.write_bytes(b"fake isolated python")
        ffmpeg = runtime_dir / "models" / "tools" / "ffmpeg.exe"
        ffmpeg.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg.write_bytes(b"fake managed ffmpeg")

        model_dir = runtime_dir / "models" / "uvr"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_names = tuple(dict.fromkeys(str(name) for name in installer.settings["separation"].values()))
        asset_bytes: dict[str, bytes] = {"vr_model_data.json": b"{}", "mdx_model_data.json": b"{}"}
        model_assets: dict[str, list[str]] = {}
        for model_name in model_names:
            pinned_entry = installer.manifest["models"][model_name]
            names = [str(item["name"]) for item in pinned_entry["files"]]
            model_assets[model_name] = names
            for name in names:
                if name in asset_bytes:
                    continue
                if name.casefold().endswith(".yaml"):
                    asset_bytes[name] = b"audio:\n  sample_rate: 44100\n"
                else:
                    asset_bytes[name] = f"fixture:{name}".encode("utf-8")

        for name, data in asset_bytes.items():
            (model_dir / name).write_bytes(data)

        entries: dict[str, dict[str, list[dict[str, object]]]] = {}
        for model_name, names in model_assets.items():
            files = []
            for name in names:
                path = model_dir / name
                data = path.read_bytes()
                stat = path.stat()
                files.append(
                    {
                        "name": name,
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "verified": True,
                    }
                )
            entries[model_name] = {"files": files}

        audio_fingerprint = installer._component_fingerprints(resolved_device)["audio"]
        manifest = {
            "schema_version": 1,
            "installer_fingerprint": audio_fingerprint,
            "audio_separator_version": installer.manifest["audio"]["audio_separator"]["version"],
            "device": resolved_device,
            "models": entries,
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        lock_relative = installer.manifest["audio"]["locks"][resolved_device]
        lock_hash = hashlib.sha256((PLUGIN_DIR / lock_relative).read_bytes()).hexdigest()
        installer._write_install_record(
            "audio",
            {
                "ready": True,
                "lock_sha256": lock_hash,
                "device": resolved_device,
                "audio_separator_version": installer.manifest["audio"]["audio_separator"]["version"],
            },
        )

    async def _prepare_with_fake_audio(self, installer: RuntimeInstaller, resolved_device: str) -> None:
        async def resolve_device(requested: str) -> str:
            return requested if requested in {"cpu", "cuda"} else resolved_device

        async def prepare_component(component: str, device: str) -> None:
            if component == "audio":
                self._write_audio_assets(installer, device)

        installer._resolve_device = resolve_device
        installer._prepare_component = prepare_component
        fake_import_check = subprocess.CompletedProcess(
            args=["fake-python", "-c"],
            returncode=0,
            stdout=json.dumps({"audio-separator": "0.47.1", "imageio-ffmpeg": "0.6.0"}),
            stderr="",
        )
        with patch.object(installer_module.subprocess, "run", return_value=fake_import_check):
            await installer.prepare(device="auto", components=("audio",))

    async def _assert_audio_contract(self, settings: dict, installer: RuntimeInstaller, expected_device: str) -> None:
        fake_import_check = subprocess.CompletedProcess(
            args=["fake-python", "-c"],
            returncode=0,
            stdout=json.dumps({"audio-separator": "0.47.1", "imageio-ffmpeg": "0.6.0"}),
            stderr="",
        )
        with patch.object(installer_module.subprocess, "run", return_value=fake_import_check):
            check = installer.check()
            audio = check["components"]["audio"]
            self.assertTrue(audio["ready"], audio)
            self.assertEqual(audio["device"], expected_device)

            persisted_status = installer.status()
            manifest_path = installer.runtime_dir / "models" / "uvr" / "manifest.json"
            asset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(asset_manifest["device"], expected_device)
            self.assertEqual(check["fingerprints"]["audio"], asset_manifest["installer_fingerprint"])
            self.assertEqual(persisted_status["fingerprints"]["audio"], asset_manifest["installer_fingerprint"])

            runtime = MusicRuntime(settings)
            readiness = await runtime.check_ready()
            self.assertTrue(readiness["components"]["separation"]["ready"], readiness)
            await runtime.close()

    async def test_auto_resolution_and_control_prepare_preserve_audio_fingerprints(self):
        torch_was_loaded = "torch" in sys.modules
        for resolved_device in ("cpu", "cuda"):
            with self.subTest(resolved_device=resolved_device), tempfile.TemporaryDirectory() as folder:
                settings = self._settings(Path(folder))
                installer = RuntimeInstaller(settings)
                await self._prepare_with_fake_audio(installer, resolved_device)
                await self._assert_audio_contract(settings, installer, resolved_device)

                async def control_only(component: str, _device: str) -> None:
                    self.assertIn(component, {"login", "netease"})

                installer._prepare_component = control_only
                fake_import_check = subprocess.CompletedProcess(
                    args=["fake-python", "-c"],
                    returncode=0,
                    stdout=json.dumps({"audio-separator": "0.47.1", "imageio-ffmpeg": "0.6.0"}),
                    stderr="",
                )
                with patch.object(installer_module.subprocess, "run", return_value=fake_import_check):
                    await installer.prepare(device="auto", components=("control",))
                await self._assert_audio_contract(settings, installer, resolved_device)
                await installer.close()
        self.assertEqual("torch" in sys.modules, torch_was_loaded)

    async def test_explicit_cpu_and_cuda_device_checks_remain_configured(self):
        for configured_device in ("cpu", "cuda"):
            with self.subTest(configured_device=configured_device), tempfile.TemporaryDirectory() as folder:
                settings = self._settings(Path(folder), configured_device)
                installer = RuntimeInstaller(settings)
                await self._prepare_with_fake_audio(installer, configured_device)
                await self._assert_audio_contract(settings, installer, configured_device)
                await installer.close()


if __name__ == "__main__":
    unittest.main()
