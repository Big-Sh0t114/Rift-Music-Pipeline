from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from music_runtime import installer as installer_module


PLUGIN_DIR = Path(installer_module.__file__).resolve().parent
manage_spec = importlib.util.spec_from_file_location("music_runtime_manage", RUNTIME_ROOT / "scripts" / "manage.py")
manage = importlib.util.module_from_spec(manage_spec)
manage_spec.loader.exec_module(manage)
fetch_asset = importlib.import_module("music_runtime.scripts.fetch_asset")


def make_settings(runtime_dir: Path, **runtime_values) -> dict:
    return {
        "plugin_dir": PLUGIN_DIR,
        "runtime_dir": runtime_dir,
        "runtime": {
            "device": "auto",
            "rift_source": "",
            "rift_python": "",
            **runtime_values,
        },
        "separation": {
            "vocals_model": "mel_band_roformer_vocals_becruily.ckpt",
            "backing_model": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            "clean_model": "UVR-DeEcho-DeReverb.pth",
        },
        "netease": {"base_url": "http://127.0.0.1:13306", "profile": "operator"},
    }


class _FakeApiCall:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.callback(*args)


class WindowsPidLivenessTests(unittest.TestCase):
    def test_windows_query_uses_openprocess_and_never_os_kill(self):
        exit_code = {"value": 259}
        open_process = _FakeApiCall(lambda *_args: 73)

        def get_exit_code(_handle, pointer):
            pointer._obj.value = exit_code["value"]
            return 1

        get_code = _FakeApiCall(get_exit_code)
        close_handle = _FakeApiCall(lambda _handle: 1)
        kernel32 = type(
            "FakeKernel32",
            (),
            {"OpenProcess": open_process, "GetExitCodeProcess": get_code, "CloseHandle": close_handle},
        )()

        with (
            patch("ctypes.WinDLL", return_value=kernel32, create=True) as win_dll,
            patch.object(installer_module.os, "kill") as os_kill,
        ):
            self.assertTrue(installer_module._windows_pid_alive(8421))
            exit_code["value"] = 0
            self.assertFalse(installer_module._windows_pid_alive(8421))

        self.assertEqual(win_dll.call_args.args, ("kernel32",))
        self.assertTrue(win_dll.call_args.kwargs["use_last_error"])
        self.assertEqual([call[2] for call in open_process.calls], [8421, 8421])
        self.assertEqual(close_handle.calls, [(73,), (73,)])
        os_kill.assert_not_called()


class InstallerReadOnlyTests(unittest.IsolatedAsyncioTestCase):
    def test_check_on_fresh_runtime_reports_missing_without_install_or_network(self):
        with tempfile.TemporaryDirectory(prefix="mus-library-installer-check-") as folder:
            runtime_dir = Path(folder) / "fresh-runtime"
            manager = installer_module.RuntimeInstaller(make_settings(runtime_dir))
            fake_git = subprocess.CompletedProcess(["git"], 1, "", "")
            with (
                patch.object(installer_module.subprocess, "run", return_value=fake_git) as subprocess_run,
                patch.object(
                    installer_module.urllib.request,
                    "urlopen",
                    side_effect=AssertionError("check must not use the network"),
                ) as urlopen,
                patch.object(
                    installer_module.asyncio,
                    "create_subprocess_exec",
                    new_callable=AsyncMock,
                    side_effect=AssertionError("check must not start processes"),
                ) as subprocess_exec,
            ):
                report = manager.check()

            self.assertFalse(report["ready"])
            self.assertEqual(set(report["components"]), {"audio", "rift", "login", "netease"})
            self.assertTrue(all(not item["ready"] for item in report["components"].values()))
            self.assertEqual(subprocess_run.call_count, 1)
            self.assertEqual(subprocess_run.call_args.args[0][0:2], ["git", "-C"])
            urlopen.assert_not_called()
            subprocess_exec.assert_not_awaited()
            self.assertFalse(runtime_dir.exists(), "read-only check created a runtime directory")

    def test_pinned_manifest_assets_and_lockfiles_are_complete_for_offline_checks(self):
        manifest = installer_module._read_manifest()
        self.assertTrue(manifest["rift"]["lock"])
        self.assertTrue((PLUGIN_DIR / manifest["rift"]["lock"]).is_file())
        for device_lock in manifest["audio"]["locks"].values():
            self.assertTrue((PLUGIN_DIR / device_lock).is_file(), device_lock)
        for name, entry in manifest["models"].items():
            self.assertTrue(entry["files"], name)
            for asset in entry["files"]:
                self.assertTrue(asset["url"].startswith("https://"), (name, asset))
                self.assertEqual(len(asset["sha256"]), 64, (name, asset))
                self.assertGreater(int(asset["size_bytes"]), 0, (name, asset))

    async def test_valid_cached_pinned_asset_is_accepted_without_fetching(self):
        with tempfile.TemporaryDirectory(prefix="mus-library-installer-cache-") as folder:
            runtime_dir = Path(folder) / "runtime"
            target = runtime_dir / "cache" / "verified.bin"
            target.parent.mkdir(parents=True)
            payload = b"fixture pinned asset\n"
            target.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            manager = installer_module.RuntimeInstaller(make_settings(runtime_dir))
            with (
                patch.object(manager, "_run", new_callable=AsyncMock, side_effect=AssertionError("unexpected fetch")) as run,
                patch.object(
                    installer_module.urllib.request,
                    "urlopen",
                    side_effect=AssertionError("cached asset verification must not use the network"),
                ) as urlopen,
            ):
                await manager._fetch_asset("https://assets.invalid/verified.bin", target, digest, len(payload))
            run.assert_not_awaited()
            urlopen.assert_not_called()
            await manager.close()

    async def test_managed_rift_pins_source_assets_and_isolated_environment(self):
        if os.name != "nt":
            self.skipTest("the managed RIFT environment is currently Windows-only")
        with tempfile.TemporaryDirectory(prefix="mus-library-rift-prepare-") as folder:
            runtime_dir = Path(folder) / "runtime"
            manifest = copy.deepcopy(installer_module._read_manifest())
            payload = b"fixture pinned RIFT module"
            manifest["rift"]["assets"] = [{
                "path": "pretrained/fixture/config.json",
                "url": "https://assets.invalid/fixture/config.json",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }]
            settings = make_settings(runtime_dir)
            with patch.object(installer_module, "_read_manifest", return_value=manifest):
                manager = installer_module.RuntimeInstaller(settings)

            commands = []
            source_lock = PLUGIN_DIR / manifest["rift"]["lock"]
            base_python = Path(folder) / "managed-python.exe"

            async def fake_run(command, *, cwd=None, env=None, timeout=0, allow_failure=False):
                args = [str(value) for value in command]
                commands.append((args, cwd, dict(env or {})))
                if args[:2] == ["git", "clone"]:
                    stage = Path(args[-1])
                    stage.mkdir(parents=True, exist_ok=False)
                    (stage / "uv.lock").write_bytes(source_lock.read_bytes())
                    (stage / "infer.py").write_text("# pinned fixture source\n", encoding="utf-8")
                elif args[:2] == ["git", "-C"] and "checkout" in args:
                    self.assertIn(manifest["rift"]["commit"], args)
                elif args[0] == sys.executable and str(PLUGIN_DIR / "scripts" / "fetch_asset.py") in args:
                    target = Path(args[args.index("--target") + 1])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    self.assertIn(manifest["rift"]["assets"][0]["sha256"], args)
                    self.assertIn(str(len(payload)), args)
                elif args[:2] == ["uv-fixture", "venv"]:
                    stage = Path(args[-1])
                    python = manager._python_executable(stage)
                    python.parent.mkdir(parents=True, exist_ok=True)
                    python.write_bytes(b"fixture interpreter")
                elif args[:2] == ["uv-fixture", "sync"]:
                    self.assertIn("--frozen", args)
                    self.assertIn("--no-install-project", args)
                    self.assertEqual(env["UV_PROJECT_ENVIRONMENT"], args[args.index("--python") + 1].split("Scripts")[0].rstrip("\\/"))
                else:
                    self.fail(f"unexpected installer subprocess command: {args!r}")
                return subprocess.CompletedProcess(args, 0, "", "")

            def fake_subprocess_run(command, **_kwargs):
                args = [str(value) for value in command]
                if args[:2] == ["git", "-C"] and args[3:] == ["rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess(args, 0, manifest["rift"]["commit"] + "\n", "")
                if len(args) >= 3 and args[1] == "-c":
                    self.assertTrue(Path(args[0]).is_file(), f"environment interpreter missing: {args[0]}")
                    return subprocess.CompletedProcess(args, 0, '{"fixture":"1"}\n', "")
                self.fail(f"unexpected synchronous process: {args!r}")

            with (
                patch.object(manager, "_run", new=fake_run),
                patch.object(manager, "_uv_exe", return_value="uv-fixture"),
                patch.object(manager, "_uv_python", new=AsyncMock(return_value=base_python)),
                patch.object(installer_module.subprocess, "run", side_effect=fake_subprocess_run) as subprocess_run,
            ):
                await manager._prepare_rift()

            source = runtime_dir / "source" / "RIFT-SVC"
            environment = runtime_dir / "envs" / "rift"
            active_python = manager._python_executable(environment)
            self.assertTrue((source / "infer.py").is_file())
            self.assertEqual((source / "pretrained/fixture/config.json").read_bytes(), payload)
            asset_record = json.loads((source / "mus_library-assets.json").read_text(encoding="utf-8"))
            self.assertEqual(asset_record["rift_commit"], manifest["rift"]["commit"])
            self.assertEqual(asset_record["files"][0]["sha256"], manifest["rift"]["assets"][0]["sha256"])
            self.assertTrue(active_python.is_file())
            installation = json.loads((runtime_dir / "installation.json").read_text(encoding="utf-8"))
            self.assertEqual(installation["rift"]["commit"], manifest["rift"]["commit"])
            self.assertTrue(any(args[:2] == ["git", "clone"] for args, _cwd, _env in commands))
            self.assertTrue(any(args[:2] == ["uv-fixture", "sync"] for args, _cwd, _env in commands))
            self.assertTrue(all(".stage-" not in str(path) for path in runtime_dir.rglob("*")))
            self.assertGreaterEqual(subprocess_run.call_count, 3)
            await manager.close()

    async def test_changed_cached_asset_requires_pinned_fetch(self):
        with tempfile.TemporaryDirectory(prefix="mus-library-installer-refresh-") as folder:
            runtime_dir = Path(folder) / "runtime"
            target = runtime_dir / "cache" / "verified.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"wrong bytes")
            payload = b"correct pinned asset"
            digest = hashlib.sha256(payload).hexdigest()
            manager = installer_module.RuntimeInstaller(make_settings(runtime_dir))
            with patch.object(manager, "_run", new_callable=AsyncMock) as run:
                await manager._fetch_asset("https://assets.invalid/verified.bin", target, digest, len(payload))
            run.assert_awaited_once()
            command = run.await_args.args[0]
            self.assertIn("--sha256", command)
            self.assertIn(digest, command)
            self.assertIn("--size", command)
            self.assertIn(str(len(payload)), command)
            await manager.close()


class InstallerStatusTests(unittest.TestCase):
    def test_status_reports_live_download_bytes_and_elapsed_without_hashing(self):
        with tempfile.TemporaryDirectory(prefix="mus-library-status-download-") as folder:
            runtime_dir = Path(folder) / "runtime"
            manager = installer_module.RuntimeInstaller(make_settings(runtime_dir))
            manager._prepare_started_at = time.time() - 20
            lock_path = runtime_dir / "installer.lock"
            lock_path.parent.mkdir(parents=True)
            lock_path.write_text(json.dumps({"pid": os.getpid(), "created_at": time.time(), "token": "fixture"}), encoding="utf-8")
            partial = runtime_dir / "models" / "uvr" / "vocal.ckpt.partial"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"partial bytes")
            manager._write_status(
                "preparing",
                "downloading_model",
                0,
                "cuda",
                {},
                detail="Downloading the selected vocals model.",
                download={
                    "component": "audio",
                    "model": "vocal.ckpt",
                    "asset": "vocal.ckpt",
                    "target": "models/uvr/vocal.ckpt",
                    "size_bytes": 100,
                },
            )

            with patch.object(installer_module, "_sha256", side_effect=AssertionError("status must not hash model assets")):
                status = manager.status()

            self.assertEqual(status["stage"], "downloading_model")
            self.assertEqual(status["detail"], "Downloading the selected vocals model.")
            self.assertGreaterEqual(status["elapsed_seconds"], 19)
            self.assertEqual(status["download"]["bytes_received"], len(b"partial bytes"))
            self.assertEqual(status["download"]["bytes_total"], 100)
            self.assertNotIn("target", status["download"])
            self.assertNotIn(str(runtime_dir), json.dumps(status))

    def test_dead_preparation_owner_is_reported_without_changing_its_lock(self):
        with tempfile.TemporaryDirectory(prefix="mus-library-status-stale-") as folder:
            runtime_dir = Path(folder) / "runtime"
            manager = installer_module.RuntimeInstaller(make_settings(runtime_dir))
            lock_path = runtime_dir / "installer.lock"
            lock_path.parent.mkdir(parents=True)
            lock_data = {"pid": 987654, "created_at": time.time(), "token": "old-owner"}
            lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
            manager._write_status("preparing", "installing_dependencies", 0, "cuda", {}, detail="Installing packages.")

            with patch.object(installer_module, "_pid_alive", return_value=True):
                live = manager.status()
            self.assertEqual(live["state"], "preparing")
            self.assertTrue(live["owner_alive"])

            with patch.object(installer_module, "_pid_alive", return_value=False):
                stale = manager.status()
            self.assertEqual(stale["state"], "failed")
            self.assertEqual(stale["error_code"], "prepare_interrupted")
            self.assertEqual(stale["stage"], "interrupted")
            self.assertEqual(json.loads(lock_path.read_text(encoding="utf-8")), lock_data)


class InstallerCliTests(unittest.TestCase):
    def test_control_component_can_succeed_while_full_runtime_remains_partial(self):
        calls = []
        partial_report = {
            "state": "partial",
            "ready": False,
            "components": {
                "netease": {"ready": True},
                "login": {"ready": True},
                "audio": {"ready": False},
                "rift": {"ready": False},
            },
        }

        class FakeInstaller:
            def __init__(self):
                pass

            async def prepare(self, device, *, components):
                calls.append((device, components))
                return partial_report

        printed = []
        with (
            patch.object(manage, "RuntimeInstaller", FakeInstaller),
            patch.object(manage, "_print", side_effect=printed.append),
        ):
            result = manage.main(["prepare", "--component", "control", "--device", "cpu"])

        self.assertEqual(result, 0)
        self.assertEqual(calls, [("cpu", ["control"])])
        self.assertEqual(printed, [partial_report])
        self.assertFalse(partial_report["ready"])


class DownloaderRetryTests(unittest.TestCase):
    def test_full_size_corrupt_partial_retries_from_zero_and_installs_pinned_bytes(self):
        with tempfile.TemporaryDirectory(prefix="mus-library-download-retry-") as folder:
            target = Path(folder) / "pinned.asset"
            partial = target.with_name(target.name + ".partial")
            expected = b"verified complete fixture bytes"
            partial.write_bytes(b"X" * len(expected))
            digest = hashlib.sha256(expected).hexdigest()
            open_calls = []

            class Response:
                status = 200

                def __init__(self):
                    self.body = io.BytesIO(expected)

                def __enter__(self):
                    return self

                def __exit__(self, _type, _value, _traceback):
                    return False

                def geturl(self):
                    return "https://assets.invalid/pinned.asset"

                def read(self, size=-1):
                    return self.body.read(size)

            class Opener:
                def open(self, request, timeout):
                    open_calls.append((request, timeout))
                    if request.get_header("Range"):
                        raise HTTPError(
                            request.full_url,
                            416,
                            "Range not satisfiable",
                            {},
                            io.BytesIO(b"fixture range rejected"),
                        )
                    return Response()

            with patch.object(fetch_asset.urllib.request, "build_opener", return_value=Opener()) as build_opener:
                fetch_asset.download(
                    "https://assets.invalid/pinned.asset",
                    target,
                    digest,
                    len(expected),
                )

            self.assertEqual(target.read_bytes(), expected)
            self.assertFalse(partial.exists())
            self.assertLessEqual(len(open_calls), 2)
            self.assertTrue(open_calls, "the downloader did not request the pinned asset")
            self.assertFalse(open_calls[-1][0].get_header("Range"), "retry must restart from byte zero")
            self.assertEqual(build_opener.call_count, len(open_calls))


class NeteaseServiceBootstrapTests(unittest.IsolatedAsyncioTestCase):
    async def test_managed_service_bootstraps_private_token_and_shuts_down_cleanly(self):
        if os.name != "nt":
            self.skipTest("the pinned Netease service launcher is Windows-only")
        with tempfile.TemporaryDirectory(prefix="mus-library-service-bootstrap-") as folder:
            runtime_dir = Path(folder) / "runtime"
            manifest = installer_module._read_manifest()
            node = runtime_dir / "tools" / f"node-v{manifest['node']['version']}-win-x64" / "node.exe"
            package = (
                runtime_dir / "service" / "netease-api" / "node_modules"
                / str(manifest["netease_api"]["name"])
            )
            node.parent.mkdir(parents=True)
            node.write_bytes(b"fixture node executable")
            package.mkdir(parents=True)
            (package / "server.js").write_text("// fixture prepared API entry\n", encoding="utf-8")

            settings = make_settings(runtime_dir)
            coordinator = installer_module._NeteaseCoordinator(settings)
            service = installer_module.NeteaseService(settings, coordinator)
            self.assertEqual(service.runtime_dir, runtime_dir.resolve())

            class FakeProcess:
                def __init__(self, pid: int, returncode: int | None = None):
                    self.pid = pid
                    self.returncode = returncode
                    self.exited = asyncio.Event()
                    if returncode is not None:
                        self.exited.set()

                async def wait(self):
                    await self.exited.wait()
                    return self.returncode

                def exit(self, code: int = 0):
                    self.returncode = code
                    self.exited.set()

                def kill(self):
                    self.exit(-9)

            api_process = FakeProcess(84211)
            calls = []

            async def fake_create_subprocess_exec(*args, **kwargs):
                calls.append((args, kwargs))
                if args and str(args[0]) == "taskkill.exe":
                    api_process.exit(0)
                    return FakeProcess(84212, 0)
                return api_process

            with (
                patch.object(service, "_identity_probe", new_callable=AsyncMock, side_effect=[False, True]),
                patch.object(installer_module, "_port_open", return_value=False) as port_open,
                patch.object(installer_module.asyncio, "create_subprocess_exec", new=fake_create_subprocess_exec),
                patch.object(
                    installer_module.urllib.request,
                    "urlopen",
                    side_effect=AssertionError("service bootstrap test must not use the network"),
                ) as urlopen,
            ):
                await service.ensure_started()
                self.assertEqual(len(calls), 1)
                args, kwargs = calls[0]
                self.assertEqual(args[0], str(node))
                self.assertEqual(args[1], "-e")
                launch_code = args[2]
                self.assertIn("fs.mkdirSync(os.tmpdir()", launch_code)
                self.assertIn("path.resolve(os.tmpdir(),'anonymous_token')", launch_code)
                token_init = launch_code.index("fs.writeFileSync(token,'','utf8')")
                config_init = launch_code.index("require('NeteaseCloudMusicApi/generateConfig')()")
                api_import = launch_code.index("require('NeteaseCloudMusicApi/server')")
                self.assertLess(token_init, config_init)
                self.assertLess(config_init, api_import)
                private_tmp = runtime_dir / "service" / "tmp"
                self.assertEqual(Path(kwargs["env"]["TEMP"]), private_tmp)
                self.assertEqual(Path(kwargs["env"]["TMP"]), private_tmp)
                self.assertEqual(Path(kwargs["env"]["TMPDIR"]), private_tmp)
                self.assertTrue(private_tmp.is_dir())
                marker = runtime_dir / "service" / "netease-service.json"
                self.assertTrue(marker.is_file())
                self.assertEqual(coordinator.process, api_process)
                await service.close()

            self.assertEqual(api_process.returncode, 0)
            self.assertIsNone(coordinator.process)
            self.assertEqual(coordinator.references, 0)
            self.assertEqual(coordinator.leases, set())
            self.assertFalse((runtime_dir / "service" / "netease-service.json").exists())
            self.assertEqual(list((runtime_dir / "active-workers").glob("*.json")) if (runtime_dir / "active-workers").exists() else [], [])
            port_open.assert_called_once_with("127.0.0.1", 13306)
            urlopen.assert_not_called()
            self.assertEqual(len(calls), 2)
            self.assertEqual(str(calls[1][0][0]), "taskkill.exe")


class InstallerLockAndCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_installation_locks_serialize_acquisition(self):
        with tempfile.TemporaryDirectory(prefix="mus-library-install-lock-") as folder:
            path = Path(folder) / "installer.lock"
            first = installer_module._InstallationLock(path)
            second = installer_module._InstallationLock(path, timeout=2)
            with patch.object(installer_module, "_pid_alive", return_value=True):
                await first.acquire()
                queued = asyncio.create_task(second.acquire())
                await asyncio.sleep(0.03)
                self.assertFalse(queued.done(), "second installer acquired the lock concurrently")
                first.release()
                await asyncio.wait_for(queued, timeout=1)
                self.assertTrue(second.acquired)
                second.release()
            self.assertFalse(path.exists())

    async def test_cancelled_preparation_reaps_mocked_child_and_releases_lock(self):
        with tempfile.TemporaryDirectory(prefix="mus-library-install-cancel-") as folder:
            runtime_dir = Path(folder) / "runtime"
            manager = installer_module.RuntimeInstaller(make_settings(runtime_dir))
            states = {
                name: {"ready": False}
                for name in ("audio", "rift", "login", "netease")
            }
            child_spawned = asyncio.Event()

            class FakeProcess:
                def __init__(self, pid: int, returncode: int | None = None):
                    self.pid = pid
                    self.returncode = returncode
                    self._exited = asyncio.Event()
                    if returncode is not None:
                        self._exited.set()
                    self.kill_called = False

                async def wait(self):
                    await self._exited.wait()
                    return self.returncode

                def exit(self, code: int = 0):
                    self.returncode = code
                    self._exited.set()

                def kill(self):
                    self.kill_called = True
                    self.exit(-9)

            child = FakeProcess(83021)
            fake_calls = []

            async def fake_create_subprocess_exec(*args, **kwargs):
                fake_calls.append((args, kwargs))
                if args and str(args[0]) == "taskkill.exe":
                    child.exit(0)
                    return FakeProcess(83022, 0)
                return child

            async def block_component(_component, _device):
                await manager._spawn(["fake-worker"], cwd=None, env={}, capture=False)
                child_spawned.set()
                await asyncio.Future()

            manager.check = lambda: {"ready": False, "components": states, "fingerprints": {}}
            with (
                patch.object(manager, "_prepare_component", new=block_component),
                patch.object(installer_module, "_worker_markers", return_value=[]),
                patch.object(installer_module.asyncio, "create_subprocess_exec", new=fake_create_subprocess_exec),
            ):
                preparing = asyncio.create_task(manager.prepare(components=("login",)))
                await asyncio.wait_for(child_spawned.wait(), timeout=1)
                await manager.close()
                with self.assertRaises(asyncio.CancelledError):
                    await preparing

            self.assertEqual(child.returncode, 0)
            self.assertEqual(manager._children, set())
            self.assertTrue(any(args and str(args[0]) == "taskkill.exe" for args, _ in fake_calls) if os.name == "nt" else True)
            self.assertFalse((runtime_dir / "installer.lock").exists())
            await manager.close()


if __name__ == "__main__":
    unittest.main()
