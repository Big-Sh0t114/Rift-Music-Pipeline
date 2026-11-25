from __future__ import annotations

import asyncio
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from types import ModuleType
from pathlib import Path
from unittest.mock import patch

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from music_runtime import runtime as runtime_module
from music_runtime.workers import audio_worker as audio_worker_module
from music_runtime.runtime import MusicRuntime, _JsonLineWorker, _prepared_model_files
from music_runtime.settings import PipelineError, load_settings, public_settings, save_settings, validate_settings
from music_runtime.workers.audio_worker import _Audio


def make_settings(root: Path) -> dict:
    runtime_dir = root / "runtime"
    plugin_dir = root / "plugin"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    plugin_dir.mkdir(parents=True, exist_ok=True)
    audio_python = runtime_dir / "envs" / "audio" / "Scripts" / "python.exe"
    audio_python.parent.mkdir(parents=True, exist_ok=True)
    audio_python.write_bytes(b"test")
    ffmpeg = runtime_dir / "models" / "tools" / "ffmpeg.exe"
    ffmpeg.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.write_bytes(b"test")
    checkpoint = plugin_dir / "voice.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    return {
        "plugin_dir": plugin_dir,
        "runtime_dir": runtime_dir,
        "ffmpeg_path": ffmpeg,
        "pipeline": {"stage_timeout_seconds": 5},
        "netease": {"base_url": "http://127.0.0.1:13307", "profile": "test"},
        "separation": {
            "vocals_model": "vocal.ckpt",
            "backing_model": "backing.ckpt",
            "clean_model": "clean.pth",
        },
        "rift": {
            "checkpoint": str(checkpoint),
            "speaker": "speaker1",
            "backing_checkpoint": "",
            "backing_speaker": "",
            "key_shift": 0,
            "backing_key_shift": 0,
            "infer_steps": 4,
            "ds_cfg_strength": 0.2,
            "spk_cfg_strength": 0.8,
            "batch_size": 1,
            "robust_f0": 0,
        },
        "mix": {"lead_gain": 1.0, "backing_gain": 0.7},
        "runtime": {"device": "cpu", "rift_source": "", "rift_python": ""},
    }


class _FakeWorker:
    def __init__(self, name: str, events: list[str], other: _FakeWorker | None = None):
        self.name = name
        self.events = events
        self.other = other
        self.alive = False
        self.requests: list[dict] = []

    def _assert_exclusive(self) -> None:
        if self.other is not None and self.other.alive:
            raise AssertionError(f"{self.name} overlapped with {self.other.name}")

    def _alive(self) -> bool:
        return self.alive

    async def start(self, _command, _cwd, _startup):
        self._assert_exclusive()
        self.alive = True
        self.events.append(f"{self.name}:start")
        return {"ok": True, "ready": True, "speakers": [{"id": "speaker1", "name": "speaker1"}]}

    async def stop(self):
        if self.alive:
            self.events.append(f"{self.name}:stop")
        self.alive = False

    async def close(self):
        await self.stop()

    async def request(self, request, timeout=None):
        del timeout
        self._assert_exclusive()
        self.requests.append(request)
        self.events.append(f"{self.name}:{request['cmd']}")
        if request["cmd"] == "separate":
            output = Path(request["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            names = ("vocals", "instrumental")
            stems = {}
            for name in names:
                path = output / f"{name}.wav"
                path.write_bytes(b"fixture")
                stems[name] = str(path)
            return {"ok": True, "stems": stems}
        output = Path(request["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fixture")
        return {"ok": True, "path": str(output)}


class RuntimeReadinessTests(unittest.TestCase):
    def test_song_octave_setting_is_an_enum_and_conflicts_with_manual_shifts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(
                validate_settings({"rift": {"song_octave_shift": 12}}, root, require_paths=False)["rift"]["song_octave_shift"],
                12,
            )
            for invalid in (-6, True, 24):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(PipelineError, "song_octave_shift"):
                    validate_settings({"rift": {"song_octave_shift": invalid}}, root, require_paths=False)
            for manual_key in ("key_shift", "backing_key_shift"):
                with self.subTest(manual_key=manual_key), self.assertRaisesRegex(PipelineError, "cannot be combined"):
                    validate_settings(
                        {"rift": {"song_octave_shift": 12, manual_key: 2}}, root, require_paths=False,
                    )

    def test_netease_probe_accepts_empty_search_without_loopback_proxy(self):
        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def read(self, size: int) -> bytes:
                return json.dumps({"code": 200, "result": {"songCount": 0}}).encode("utf-8")[:size]

        calls: list[tuple[str, float]] = []
        handler_sets: list[tuple] = []

        class _Opener:
            def open(self, url: str, timeout: float):
                calls.append((url, timeout))
                return _Response()

        def build_opener(*handlers):
            handler_sets.append(handlers)
            return _Opener()

        with tempfile.TemporaryDirectory() as folder:
            runtime = MusicRuntime(make_settings(Path(folder)))
            with patch.object(urllib.request, "build_opener", side_effect=build_opener):
                self.assertTrue(asyncio.run(runtime._netease_probe()))

        self.assertEqual(len(calls), 1)
        self.assertLessEqual(calls[0][1], 0.25)
        self.assertEqual(len(handler_sets), 1)
        proxy = next(handler for handler in handler_sets[0] if isinstance(handler, urllib.request.ProxyHandler))
        self.assertEqual(proxy.proxies, {})

    def test_settings_update_preserves_plugin_table_and_hides_runtime_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            plugin_dir = Path(folder)
            (plugin_dir / "config.toml").write_text(
                '# preserved plugin setup\n[plugin]\nenable = true\nonebot_base = "http://127.0.0.1:5700"\n'
                '[pipeline]\nmax_queue = 5 # keep operator limit\n',
                encoding="utf-8",
            )
            public = save_settings(
                {"pipeline": {"enabled": False}, "runtime": {"device": "cpu"}, "mix": {"lead_gain": 2.8}},
                plugin_dir,
            )
            content = (plugin_dir / "config.toml").read_text(encoding="utf-8")
            self.assertIn("# preserved plugin setup", content)
            self.assertIn('onebot_base = "http://127.0.0.1:5700"', content)
            self.assertFalse(public["pipeline"]["enabled"])
            self.assertEqual(public["pipeline"]["max_queue"], 5)
            self.assertEqual(public["mix"]["lead_gain"], 2.8)
            self.assertIn("max_queue = 5 # keep operator limit", content)
            self.assertNotIn("runtime_dir", public)
            self.assertNotIn("ffmpeg_path", public)

    def test_missing_stored_model_paths_remain_readable_and_repairable(self):
        with tempfile.TemporaryDirectory() as folder:
            plugin_dir = Path(folder)
            missing_checkpoint = r"H:\OfflineModels\missing.ckpt"
            missing_source = r"C:\OfflineRIFT\source"
            missing_python = r"C:\OfflineRIFT\.venv\Scripts\python.exe"
            (plugin_dir / "config.toml").write_text(
                "\n".join(
                    (
                        "[pipeline]",
                        "enabled = false",
                        "[rift]",
                        f"checkpoint = {json.dumps(missing_checkpoint)}",
                        'speaker = "speaker530"',
                        "[runtime]",
                        f"rift_source = {json.dumps(missing_source)}",
                        f"rift_python = {json.dumps(missing_python)}",
                    )
                ),
                encoding="utf-8",
            )
            loaded = load_settings(plugin_dir)
            public = public_settings(loaded)
            self.assertEqual(public["rift"]["checkpoint"], missing_checkpoint)
            self.assertEqual(public["runtime"]["rift_source"], missing_source)
            self.assertEqual(public["runtime"]["rift_python"], missing_python)

            saved = save_settings({"rift": {"speaker": "speaker531"}}, plugin_dir)
            self.assertEqual(saved["rift"]["checkpoint"], missing_checkpoint)
            self.assertEqual(saved["rift"]["speaker"], "speaker531")
            with self.assertRaises(PipelineError):
                save_settings({"rift": {"checkpoint": r"H:\OfflineModels\other-missing.ckpt"}}, plugin_dir)

    def test_model_manifest_requires_all_prepared_files_and_matching_stats(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime_dir = Path(folder)
            cache = runtime_dir / "models" / "uvr"
            cache.mkdir(parents=True)
            model = cache / "vocal.ckpt"
            model.write_bytes(b"known weights")
            stat = model.stat()
            manifest = {
                "schema_version": 1,
                "installer_fingerprint": "a" * 64,
                "audio_separator_version": "0.47.1",
                "models": {
                    "vocal.ckpt": {
                        "files": [
                            {
                                "name": model.name,
                                "size_bytes": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns,
                                "sha256": "b" * 64,
                                "verified": True,
                            }
                        ]
                    }
                },
            }
            (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            files = _prepared_model_files(runtime_dir, "vocal.ckpt", expected_fingerprint="a" * 64)
            self.assertEqual(files[0]["name"], "vocal.ckpt")
            self.assertIsNone(_prepared_model_files(runtime_dir, "vocal.ckpt", expected_fingerprint="c" * 64))
            model.write_bytes(b"replaced weights")
            self.assertIsNone(_prepared_model_files(runtime_dir, "vocal.ckpt", expected_fingerprint="a" * 64))

    def test_import_does_not_load_torch_or_audio_separator(self):
        source = "import sys; sys.path.insert(0, " + repr(str(RUNTIME_ROOT)) + "); " \
            "import music_runtime.runtime; " \
            "assert 'torch' not in sys.modules; assert 'audio_separator' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-I", "-c", source],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class RuntimeProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_releasing_inference_worker_exits_process_and_allows_next_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = MusicRuntime(make_settings(root))
            self.addAsyncCleanup(runtime.close)
            script = (
                "import sys\n"
                "sys.stdin.readline()\n"
                "print('{\"ok\":true,\"ready\":true}', flush=True)\n"
                "sys.stdin.read()\n"
            )
            for worker in (runtime._rift, runtime._audio):
                await worker.start([sys.executable, "-u", "-c", script], root, {"cmd": "start"})
                process = worker.process
                self.assertIsNotNone(process)
                await runtime.release_workers()
                self.assertIsNotNone(process.returncode)
                self.assertIsNone(worker.process)
                self.assertEqual(list((runtime.runtime_dir / "active-workers").glob("*.json")), [])
                self.assertFalse(runtime._closed)

    async def test_protocol_failure_stops_worker_and_removes_active_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker_dir = root / "active-workers"
            script = (
                "import sys\n"
                "sys.stdin.readline()\n"
                "print('{\"ok\":true,\"ready\":true}', flush=True)\n"
                "sys.stdin.readline()\n"
                "print('not json', flush=True)\n"
                "sys.stdin.read()\n"
            )
            worker = _JsonLineWorker(timeout=3, marker_dir=marker_dir, component="test")
            await worker.start([sys.executable, "-u", "-c", script], root, {"cmd": "start"})
            process = worker.process
            markers = list(marker_dir.glob("test-*.json"))
            self.assertEqual(len(markers), 1)
            self.assertEqual(json.loads(markers[0].read_text(encoding="utf-8"))["pid"], process.pid)
            with self.assertRaises(PipelineError):
                await worker.request({"cmd": "break"})
            self.assertIsNone(worker.process)
            self.assertIsNotNone(process.returncode)
            self.assertEqual(list(marker_dir.glob("test-*.json")), [])

    async def test_cancellation_proves_child_exit_before_removing_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker_dir = root / "active-workers"
            script = (
                "import sys, time\n"
                "sys.stdin.readline()\n"
                "print('{\"ok\":true,\"ready\":true}', flush=True)\n"
                "sys.stdin.readline()\n"
                "time.sleep(60)\n"
            )
            worker = _JsonLineWorker(timeout=10, marker_dir=marker_dir, component="test")
            await worker.start([sys.executable, "-u", "-c", script], root, {"cmd": "start"})
            process = worker.process
            task = asyncio.create_task(worker.request({"cmd": "hang"}))
            await asyncio.sleep(0.1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIsNotNone(process.returncode)
            self.assertEqual(list(marker_dir.glob("test-*.json")), [])
            self.assertIsNone(worker.process)

    async def test_uvr_and_rift_workers_never_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root)
            runtime = MusicRuntime(settings)
            events: list[str] = []
            rift = _FakeWorker("rift", events)
            audio = _FakeWorker("audio", events, other=rift)
            rift.other = audio
            runtime._rift = rift
            runtime._audio = audio
            source = runtime.runtime_dir / "input" / "vocals.wav"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"input")
            with patch.object(runtime_module, "_prepared_model_files", return_value=[{"name": "fixture"}]), \
                    patch.object(runtime, "_audio_installer_fingerprint", return_value="d" * 64):
                stems = await runtime.separate(source, runtime.runtime_dir / "job" / "vocals", "vocals")
                self.assertEqual(set(stems), {"vocals", "instrumental"})
                self.assertFalse(rift.alive)
                self.assertTrue(audio.alive)

                async def start_rift():
                    rift._assert_exclusive()
                    rift.alive = True
                    runtime._ready_response = {"speakers": [{"id": "speaker1", "name": "speaker1"}]}

                runtime._ensure_rift_started_unlocked = start_rift
                converted = await runtime.convert(
                    source, runtime.runtime_dir / "job" / "lead.wav", "lead", song_octave_shift=12,
                )
                self.assertTrue(converted.is_file())
                self.assertTrue(rift.alive)
                self.assertFalse(audio.alive)
                request = next(item for item in rift.requests if item["cmd"] == "convert")
                self.assertEqual(request["rift"]["song_octave_shift"], 12)
                self.assertEqual(request["rift"]["song_detection_backing_manual_shift"], 0)
            await runtime.close()


class SeparatorStemPlanTests(unittest.TestCase):
    class _Model:
        def __init__(self, primary: str, secondary: str):
            self.primary_stem_name = primary
            self.secondary_stem_name = secondary

    class _Separator:
        STEM_NAME_MAP = {
            "vocals": "Vocals",
            "instrumental": "Instrumental",
            "no other": "No Other",
            "other": "Other",
            "no reverb": "No Reverb",
            "reverb": "Reverb",
        }

        def __init__(self, primary: str, secondary: str):
            self.model_instance = SeparatorStemPlanTests._Model(primary, secondary)

    def _service(self, model: str, digest: str, primary: str, secondary: str) -> _Audio:
        service = object.__new__(_Audio)
        service.model = model
        service.model_sha256 = digest
        service.separator = self._Separator(primary, secondary)
        return service

    def test_real_source_named_vocals_maps_exact_declared_stems(self):
        service = self._service(
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", "1" * 64, "Vocals", "Instrumental"
        )
        self.assertEqual(
            service._stem_plan("backing"),
            ({"Vocals": "lead", "Instrumental": "backing"}, ["lead", "backing"]),
        )

    def test_becruily_other_is_instrumental_only_for_the_pinned_vocals_model(self):
        model = "mel_band_roformer_vocals_becruily.ckpt"
        digest = "a05961310cc55fbb901290c2e8be02682942f73522b6ac76bf2ec11e347ed95a"
        service = self._service(model, digest, "vocals", "other")
        self.assertEqual(
            service._stem_plan("vocals"),
            ({"vocals": "vocals", "other": "instrumental"}, ["vocals", "instrumental"]),
        )
        with self.assertRaises(ValueError):
            service._stem_plan("backing")
        service.model_sha256 = "0" * 64
        with self.assertRaises(ValueError):
            service._stem_plan("vocals")

    def test_clean_alias_is_only_accepted_for_exact_model_hash(self):
        digest = "e644028ec82865dc0fe082bc6fea85a43f7c71cfe375caee2da2d154aa661ee7"
        service = self._service("UVR-DeEcho-DeReverb.pth", digest, "No Other", "Other")
        self.assertEqual(service._stem_plan("clean"), ({"No Other": "dry"}, ["dry"]))
        service.model_sha256 = "0" * 64
        with self.assertRaises(ValueError):
            service._stem_plan("clean")

    def test_no_reverb_label_maps_to_dry_stem(self):
        service = self._service("UVR-DeEcho-DeReverb.pth", "e644028ec82865dc0fe082bc6fea85a43f7c71cfe375caee2da2d154aa661ee7", "No Reverb", "Reverb")
        self.assertEqual(service._stem_plan("clean"), ({"No Reverb": "dry"}, ["dry"]))


class OfflineSeparatorLoadTests(unittest.TestCase):
    _BASE_FILES = {
        "vr_model_data.json": json.dumps({}).encode("utf-8"),
        "mdx_model_data.json": json.dumps({}).encode("utf-8"),
    }

    class _FakeTorch(ModuleType):
        def __init__(self):
            super().__init__("torch")
            self.cuda = self

        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            return None

    def _prepared_service(self, root: Path, model: str, model_bytes: bytes, extra_files: dict[str, bytes]):
        cache = root / "models" / "uvr"
        cache.mkdir(parents=True)
        contents = {**self._BASE_FILES, model: model_bytes, **extra_files}
        records = []
        for name, body in contents.items():
            path = cache / name
            path.write_bytes(body)
            stat_value = path.stat()
            records.append(
                {
                    "name": name,
                    "size_bytes": stat_value.st_size,
                    "mtime_ns": stat_value.st_mtime_ns,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "verified": True,
                }
            )
        manifest = {
            "schema_version": 1,
            "audio_separator_version": "0.47.1",
            "models": {model: {"files": records}},
        }
        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        service = object.__new__(_Audio)
        service.runtime_dir = root.resolve(strict=True)
        service.model_cache = cache.resolve(strict=True)
        service.output_dir = None
        service.separator = None
        service.model = ""
        service.model_identity = None
        service.model_sha256 = ""
        service._verified_model_assets = {}
        service.device = "cpu"
        service.ffmpeg_path = None
        return service, cache

    def _load_with_network_forbidden(
        self, service: _Audio, model: str, *, package_model_data: dict | None = None, model_hash: str | None = None,
    ):
        instances = []

        class FakeSeparator:
            def __init__(self, **kwargs):
                self.output_dir = kwargs["output_dir"]
                self.model_file_dir = kwargs["model_file_dir"]
                self.output_format = kwargs["output_format"]
                self.torch_device = "cuda"
                self.torch_device_cpu = "cpu"
                self.onnx_execution_provider = ["CUDAExecutionProvider"]
                instances.append(self)

            def load_model(self, model_filename):
                self.model_info = self.download_model_files(model_filename)
                if self.model_info[1] == "VR":
                    self.model_data = self.load_model_data_using_hash(self.model_info[3])
                else:
                    self.model_data = self.load_model_data_from_yaml(self.model_info[4])

            def load_model_data_from_yaml(self, yaml_name):
                path = Path(self.model_file_dir) / yaml_name
                if not path.is_file():
                    raise AssertionError("offline model YAML was not prepared")
                return path.read_text(encoding="utf-8")

            def download_model_files(self, *_args, **_kwargs):
                raise AssertionError("audio-separator remote model discovery was called")

            def list_supported_model_files(self, *_args, **_kwargs):
                raise AssertionError("audio-separator supported-model list was called")

            def download_file_if_not_exists(self, *_args, **_kwargs):
                raise AssertionError("audio-separator remote asset download was called")

            def load_model_data_using_hash(self, *_args, **_kwargs):
                raise AssertionError("audio-separator remote hash metadata loader was called")

            def get_model_hash(self, model_path):
                self.hashed_model = model_path
                return model_hash

        package = ModuleType("audio_separator")
        package.__path__ = []
        separator_module = ModuleType("audio_separator.separator")
        separator_module.Separator = FakeSeparator
        torch = self._FakeTorch()
        mocks = [
            patch.dict(
                sys.modules,
                {
                    "torch": torch,
                    "audio_separator": package,
                    "audio_separator.separator": separator_module,
                },
            )
        ]
        if package_model_data is not None:
            mocks.append(
                patch.object(
                    audio_worker_module.resources,
                    "open_text",
                    return_value=io.StringIO(json.dumps(package_model_data)),
                )
            )
        with mocks[0]:
            if len(mocks) > 1:
                with mocks[1]:
                    service._load_model(model, service.runtime_dir / "out")
            else:
                service._load_model(model, service.runtime_dir / "out")
        return instances[0]

    def test_first_roformer_load_uses_verified_yaml_without_remote_discovery(self):
        model = "mel_band_roformer_vocals_becruily.ckpt"
        yaml_name = "config_mel_band_roformer_vocals_becruily.yaml"
        with tempfile.TemporaryDirectory() as folder:
            service, cache = self._prepared_service(
                Path(folder), model, b"weight bytes", {yaml_name: b"audio:\n  sample_rate: 44100\n"}
            )
            separator = self._load_with_network_forbidden(service, model)
            self.assertEqual(separator.model_info[0], model)
            self.assertEqual(separator.model_info[1], "MDXC")
            self.assertEqual(separator.model_info[3], str(cache / model))
            self.assertEqual(separator.model_info[4], yaml_name)
            self.assertIn("sample_rate", separator.model_data)

    def test_vr_hash_metadata_comes_from_pinned_local_json(self):
        model = "UVR-DeEcho-DeReverb.pth"
        model_bytes = b"prepared dry model bytes"
        digest = "0fb9249ffe4ffc38d7b16243f394c0ff"
        with tempfile.TemporaryDirectory() as folder:
            service, _cache = self._prepared_service(Path(folder), model, model_bytes, {})
            vr_path = service.model_cache / "vr_model_data.json"
            vr_path.write_text(json.dumps({digest: {"primary_stem": "No Reverb", "vr_model_param": "4band_v3"}}), encoding="utf-8")
            # Update the fixture manifest stats/hash as installer-verified assets would be.
            manifest_path = service.model_cache / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            item = next(item for item in manifest["models"][model]["files"] if item["name"] == "vr_model_data.json")
            item.update(size_bytes=vr_path.stat().st_size, mtime_ns=vr_path.stat().st_mtime_ns, sha256=hashlib.sha256(vr_path.read_bytes()).hexdigest())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            separator = self._load_with_network_forbidden(
                service,
                model,
                package_model_data={"vr_model_data": {}, "mdx_model_data": {}},
                model_hash=digest,
            )
            self.assertEqual(separator.model_data["primary_stem"], "No Reverb")
            self.assertEqual(separator.model_info[1], "VR")
            self.assertEqual(separator.hashed_model, str(service.model_cache / model))


if __name__ == "__main__":
    unittest.main()
