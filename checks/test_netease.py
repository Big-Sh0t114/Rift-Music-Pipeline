from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from types import ModuleType
from pathlib import Path
from unittest.mock import AsyncMock, patch

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from music_runtime.providers.login import NeteaseLoginManager
from music_runtime.providers.netease import NeteaseProvider
from music_runtime.settings import PipelineError, SelectionRequired


def make_settings(root: Path, *, limit: int = 1024, base_url: str = "http://127.0.0.1:13306") -> dict:
    runtime_dir = root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return {
        "runtime_dir": runtime_dir,
        "plugin_dir": root / "plugin",
        "pipeline": {"max_download_bytes": limit},
        "netease": {"base_url": base_url, "profile": "test"},
    }


class _FakeContent:
    def __init__(self, body: bytes, *, chunk_size: int = 23):
        self.body = body
        self.chunk_size = chunk_size

    async def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]

    async def iter_chunked(self, size: int):
        step = min(size, self.chunk_size)
        for offset in range(0, len(self.body), step):
            await asyncio.sleep(0)
            yield self.body[offset : offset + step]


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None, status: int = 200):
        self.status = status
        self.headers = headers or {}
        self.content = _FakeContent(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeSession:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict[str, str] | None = None):
        self.body = body
        self.status = status
        self.headers = headers or {}
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse(self.body, self.headers, self.status)

    async def close(self):
        return None


class _GateContent(_FakeContent):
    def __init__(self, body: bytes, started: asyncio.Event, hold: asyncio.Event):
        super().__init__(body)
        self.started = started
        self.hold = hold

    async def iter_chunked(self, size: int):
        self.started.set()
        await self.hold.wait()
        async for chunk in super().iter_chunked(size):
            yield chunk


def _write_cookie(settings: dict) -> None:
    path = Path(settings["runtime_dir"]) / "credentials" / "test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"profile": "test", "cookie": "MUSIC_U=private-cookie"}), encoding="utf-8")


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_raw_set_cookie_is_normalized_before_api_use(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            credentials = Path(settings["runtime_dir"]) / "credentials" / "test.json"
            credentials.parent.mkdir(parents=True)
            credentials.write_text(
                json.dumps({
                    "profile": "test",
                    "cookie": (
                        "MUSIC_U=music-token-for-test; Max-Age=15552000; Path=/; HttpOnly;"
                        "__csrf=csrf-token-for-test; Path=/; NMTID=device-token-for-test;"
                        "Domain=.music.163.com; Secure"
                    ),
                }),
                encoding="utf-8",
            )
            provider = NeteaseProvider(settings)
            self.assertEqual(
                provider._cookie(),
                "MUSIC_U=music-token-for-test; __csrf=csrf-token-for-test; NMTID=device-token-for-test",
            )
            await provider.close()

    async def test_managed_service_lease_is_limited_to_fixed_default_port(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            external_settings = make_settings(root, base_url="http://localhost:13307")
            external = NeteaseProvider(external_settings)
            external_service = _FakeService()
            with _patched_service(external_service):
                await external._ensure_managed_service()
                self.assertIsNone(external._service)
                self.assertEqual(external.base_url, "http://localhost:13307")
                await external.close()
            self.assertEqual(external_service.ensure_count, 0)
            self.assertEqual(external_service.close_count, 0)

            managed_settings = make_settings(root, base_url="http://localhost:13306")
            managed = NeteaseProvider(managed_settings)
            managed_service = _FakeService()
            with _patched_service(managed_service):
                await managed._ensure_managed_service()
                self.assertEqual(managed.base_url, "http://127.0.0.1:13306")
                await managed.close()
            self.assertEqual(managed_service.ensure_count, 1)
            self.assertEqual(managed_service.close_count, 1)

    async def test_explicit_id_and_link_fetch_detail_without_search(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = NeteaseProvider(make_settings(Path(folder)))
            expected = {"id": "123", "title": "Song", "artist": "Artist", "duration_ms": 42_000}
            provider._api_request = AsyncMock(return_value={"code": 200, "songs": [{"id": 123, "name": "Song", "ar": [{"name": "Artist"}], "dt": 42_000}]})
            self.assertEqual(await provider.resolve("123"), expected)
            self.assertEqual(await provider.resolve("https://music.163.com/#/song?id=123"), expected)
            self.assertEqual(provider._api_request.await_args_list[0].args[0], "song/detail")
            self.assertEqual(provider._api_request.await_args_list[0].args[1], {"ids": "123"})
            await provider.close()

    async def test_same_title_versions_require_a_choice(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = NeteaseProvider(make_settings(Path(folder)))
            provider._api_request = AsyncMock(
                return_value={
                    "code": 200,
                    "result": {
                        "songs": [
                            {"id": 1, "name": "Song", "artists": [{"name": "Studio"}], "duration": 180_000},
                            {"id": 2, "name": "Song", "artists": [{"name": "Live"}], "duration": 240_000},
                        ]
                    },
                }
            )
            with self.assertRaises(SelectionRequired) as caught:
                await provider.resolve("Song")
            self.assertEqual([item["id"] for item in caught.exception.candidates], ["1", "2"])
            await provider.close()

    async def test_trial_and_empty_url_are_rejected_and_auth_expiry_is_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            _write_cookie(settings)
            provider = NeteaseProvider(settings)
            provider._api_request = AsyncMock(
                return_value={"code": 200, "data": [{"id": "42", "url": "https://m10.music.126.net/song.mp3", "freeTrialInfo": {"start": 0, "end": 30_000}}]}
            )
            with self.assertRaises(PipelineError) as trial:
                await provider.download({"id": "42"}, settings["runtime_dir"] / "trial.download")
            self.assertEqual(trial.exception.code, "trial_audio")

            provider._api_request = AsyncMock(return_value={"code": 200, "data": [{"id": "42", "url": None}]})
            with self.assertRaises(PipelineError) as empty:
                await provider.download({"id": "42"}, settings["runtime_dir"] / "empty.download")
            self.assertEqual(empty.exception.code, "download_unavailable")
            self.assertEqual(provider._api_request.await_count, 3)

            provider._api_request = AsyncMock(return_value={"code": 301, "data": []})
            with self.assertRaises(PipelineError) as expired:
                await provider.download({"id": "42"}, settings["runtime_dir"] / "expired.download")
            self.assertEqual(expired.exception.code, "login_expired")
            await provider.close()

    async def test_download_stream_is_async_bounded_and_never_sends_cookie_to_cdn(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root, limit=2048)
            _write_cookie(settings)
            provider = NeteaseProvider(settings)
            body = b"ID3" + b"a" * 300
            session = _FakeSession(body, headers={"Content-Length": str(len(body))})
            provider._session = session
            calls: list[tuple[str, dict]] = []

            async def api_request(endpoint, params, *, cookie=""):
                calls.append((endpoint, {**params, "cookie": cookie}))
                return {"code": 200, "data": [{"id": "42", "url": "http://m10.music.126.net/audio.mp3"}]}

            provider._api_request = api_request
            destination = settings["runtime_dir"] / "jobs" / "42" / "source.download"
            download = asyncio.create_task(provider.download({"id": "42"}, destination))
            ticks = 0
            while not download.done():
                ticks += 1
                await asyncio.sleep(0)
            self.assertEqual(await download, destination)
            self.assertEqual(destination.read_bytes(), body)
            self.assertGreater(ticks, 1)
            self.assertEqual(calls[0][1]["level"], "lossless")
            self.assertEqual(calls[0][1]["unblock"], "false")
            self.assertEqual(calls[0][1]["cookie"], "MUSIC_U=private-cookie")
            cdn_headers = session.calls[0][1]["headers"]
            self.assertNotIn("Cookie", cdn_headers)
            self.assertTrue(session.calls[0][0].startswith("https://m10.music.126.net/"))
            await provider.close()

    async def test_http_400_with_network_risk_uses_the_same_song_and_account_on_legacy_route(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), base_url="http://127.0.0.1:13307")
            _write_cookie(settings)
            provider = NeteaseProvider(settings)
            audio_url = "https://m10.music.126.net/song.mp3"
            song_id = "42"
            destination = Path(settings["runtime_dir"]) / "source.download"
            calls = []

            def get(url, **kwargs):
                calls.append((url, kwargs["params"], kwargs["headers"]["Cookie"]))
                if url.endswith("/song/url/v1"):
                    return _FakeResponse(b'{"code":-460}', status=400)
                payload = {"code": 200, "data": [{"id": song_id, "code": 200, "url": audio_url}]}
                return _FakeResponse(json.dumps(payload).encode())

            session = _FakeSession(b"")
            session.get = get
            provider._session = session
            provider._download_stream = AsyncMock(return_value=destination)
            try:
                self.assertEqual(await provider.download({"id": song_id}, destination), destination)
                self.assertEqual(calls, [
                    ("http://127.0.0.1:13307/song/url/v1", {"id": song_id, "level": "lossless", "unblock": "false"}, "MUSIC_U=private-cookie"),
                    ("http://127.0.0.1:13307/song/url", {"id": song_id, "br": 320000, "unblock": "false"}, "MUSIC_U=private-cookie"),
                ])
                provider._download_stream.assert_awaited_once_with(audio_url, destination, 1024)
            finally:
                await provider.close()

    async def test_legacy_url_still_requires_the_requested_song_and_a_complete_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            provider = NeteaseProvider(settings)
            url = "https://m10.music.126.net/song.mp3"
            cases = (
                ({"id": "41", "code": 200, "url": url}, "provider_invalid_response"),
                ({"id": "42", "code": 200, "url": url, "freeTrialInfo": {"end": 30_000}}, "trial_audio"),
                ({"id": "42", "code": 404, "url": None}, None),
            )
            for record, error in cases:
                with self.subTest(error=error):
                    provider._api_request = AsyncMock(side_effect=[
                        PipelineError("netease_network_risk", "-460"), {"code": 200, "data": [record]},
                    ])
                    if error:
                        with self.assertRaises(PipelineError) as caught:
                            await provider._lookup_audio("42", "MUSIC_U=private-cookie")
                        self.assertEqual(caught.exception.code, error)
                    else:
                        self.assertIsNone(await provider._lookup_audio("42", "MUSIC_U=private-cookie"))
                    self.assertEqual(provider._api_request.await_count, 2)
            await provider.close()

    async def test_download_limit_and_content_length_truncation_leave_no_partial_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root, limit=200)
            _write_cookie(settings)
            provider = NeteaseProvider(settings)
            provider._api_request = AsyncMock(return_value={"code": 200, "data": [{"id": "42", "url": "https://m10.music.126.net/audio.mp3"}]})
            provider._session = _FakeSession(b"ID3" + b"x" * 250, headers={"Content-Length": "253"})
            destination = settings["runtime_dir"] / "jobs" / "too-large.download"
            with self.assertRaises(PipelineError) as large:
                await provider.download({"id": "42"}, destination)
            self.assertEqual(large.exception.code, "download_too_large")
            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.glob("*.part")), [])

            provider.settings["pipeline"]["max_download_bytes"] = 1024
            provider._session = _FakeSession(b"ID3" + b"x" * 130, headers={"Content-Length": "300"})
            with self.assertRaises(PipelineError) as truncated:
                await provider.download({"id": "42"}, destination)
            self.assertEqual(truncated.exception.code, "download_incomplete")
            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.glob("*.part")), [])
            await provider.close()

    async def test_destination_must_stay_inside_plugin_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder))
            _write_cookie(settings)
            provider = NeteaseProvider(settings)
            provider._api_request = AsyncMock(return_value={"code": 200, "data": [{"id": "42", "url": "https://m10.music.126.net/audio.mp3"}]})
            with self.assertRaises(PipelineError) as caught:
                await provider.download({"id": "42"}, Path(folder) / "outside.download")
            self.assertEqual(caught.exception.code, "path_outside_runtime")
            await provider.close()

    async def test_cancelled_download_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), limit=1024)
            _write_cookie(settings)
            provider = NeteaseProvider(settings)
            started = asyncio.Event()
            hold = asyncio.Event()
            response = _FakeResponse(
                b"ID3" + b"x" * 300,
                headers={"Content-Length": "303"},
            )
            response.content = _GateContent(response.content.body, started, hold)

            class GateSession:
                def get(self, _url, **_kwargs):
                    return response

                async def close(self):
                    return None

            provider._session = GateSession()
            provider._api_request = AsyncMock(
                return_value={"code": 200, "data": [{"id": "42", "url": "https://m10.music.126.net/audio.mp3"}]}
            )
            destination = settings["runtime_dir"] / "jobs" / "cancelled.download"
            task = asyncio.create_task(provider.download({"id": "42"}, destination))
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.glob("*.part")), [])
            await provider.close()

class _FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None
        self._done = asyncio.get_running_loop().create_future()

    async def wait(self):
        await self._done
        return self.returncode

    def kill(self):
        if self.returncode is None:
            self.returncode = -9
            if not self._done.done():
                self._done.set_result(self.returncode)


class _FakeService:
    def __init__(self):
        self.ensure_count = 0
        self.close_count = 0

    async def ensure_started(self):
        self.ensure_count += 1
        return {"base_url": "http://127.0.0.1:13306"}

    async def close(self):
        self.close_count += 1


class _TestLoginManager(NeteaseLoginManager):
    async def _terminate_process(self, process):
        process.kill()
        await process.wait()


def _valid_png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (0).to_bytes(4, "big") + b"IHDR" + b"\0" * 4
        + (0).to_bytes(4, "big") + b"IEND" + b"\0" * 4
    )


def _patched_service(service):
    module = ModuleType("music_runtime.installer")
    module.get_netease_service = lambda _settings: service
    return patch.dict(sys.modules, {"music_runtime.installer": module})


class LoginManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_error_message_uses_manager_whitelist(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = make_settings(Path(folder), base_url="http://localhost:13307")
            login_python = settings["runtime_dir"] / "envs" / "login" / "Scripts" / "python.exe"
            login_python.parent.mkdir(parents=True, exist_ok=True)
            login_python.write_bytes(b"test")

            async def factory(*command, **_kwargs):
                job_dir = Path(command[command.index("--job-dir") + 1])
                (job_dir / "status.json").write_text(
                    json.dumps({
                        "status": "error",
                        "error_code": "account_validation_failed",
                        "message": "MUSIC_U=must-not-escape",
                    }),
                    encoding="utf-8",
                )
                return _FakeProcess(50_001)

            manager = _TestLoginManager(settings, process_factory=factory, stop_timeout=0.1)
            result = await manager.start()
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["message"], "账号验证失败，请重新进行网易云音乐登录。")
            self.assertNotIn("MUSIC_U", json.dumps(result))
            self.assertNotIn("error_code", result)
            await manager.shutdown()

    async def test_cancel_retry_stale_ids_and_no_credential_exposure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root)
            login_python = settings["runtime_dir"] / "envs" / "login" / "Scripts" / "python.exe"
            login_python.parent.mkdir(parents=True, exist_ok=True)
            login_python.write_bytes(b"test")
            service = _FakeService()
            processes: list[_FakeProcess] = []

            async def factory(*command, **_kwargs):
                index = len(processes) + 1
                job_dir = Path(command[command.index("--job-dir") + 1])
                job_dir.mkdir(parents=True, exist_ok=True)
                (job_dir / "status.json").write_text(json.dumps({"status": "waiting", "message": "MUSIC_U=must-not-escape"}), encoding="utf-8")
                (job_dir / "qr.png").write_bytes(_valid_png())
                process = _FakeProcess(20_000 + index)
                processes.append(process)
                return process

            manager = _TestLoginManager(settings, process_factory=factory, stop_timeout=0.1)
            with _patched_service(service):
                first = await manager.start()
                self.assertEqual(first["status"], "waiting")
                self.assertTrue(first["qr_ready"])
                qr = await manager.read_qr(first["job_id"])
                self.assertEqual(qr, _valid_png())
                self.assertNotIn("cookie", json.dumps(first).casefold())
                self.assertNotIn("MUSIC_U", json.dumps(first))
                canceled = await manager.cancel(first["job_id"])
                self.assertEqual(canceled["status"], "cancelled")
                self.assertIsNone(await manager.status(first["job_id"]))
                self.assertEqual(list((settings["runtime_dir"] / "active-workers").glob("login-*.json")), [])

                second = await manager.start()
                self.assertNotEqual(first["job_id"], second["job_id"])
                self.assertTrue(second["qr_ready"])
                self.assertIsNone(await manager.read_qr(first["job_id"]))
                await manager.shutdown()

            self.assertTrue(all(process.returncode is not None for process in processes))
            self.assertEqual(service.ensure_count, 2)
            self.assertEqual(service.close_count, 1)
            self.assertFalse((settings["runtime_dir"] / "credentials" / "test.json").exists())
            self.assertEqual(list((settings["runtime_dir"] / "active-workers").glob("login-*.json")), [])

    async def test_custom_loopback_login_uses_external_endpoint_without_managed_lease(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root, base_url="http://localhost:13307")
            login_python = settings["runtime_dir"] / "envs" / "login" / "Scripts" / "python.exe"
            login_python.parent.mkdir(parents=True, exist_ok=True)
            login_python.write_bytes(b"test")
            service = _FakeService()
            processes: list[_FakeProcess] = []
            commands: list[tuple[str, ...]] = []

            async def factory(*command, **_kwargs):
                commands.append(command)
                job_dir = Path(command[command.index("--job-dir") + 1])
                job_dir.mkdir(parents=True, exist_ok=True)
                (job_dir / "status.json").write_text(json.dumps({"status": "waiting"}), encoding="utf-8")
                process = _FakeProcess(40_001)
                processes.append(process)
                return process

            manager = _TestLoginManager(settings, process_factory=factory, stop_timeout=0.1)
            with _patched_service(service):
                result = await manager.start()
                self.assertEqual(result["status"], "waiting")
                self.assertEqual(commands[0][commands[0].index("--base-url") + 1], "http://127.0.0.1:13307")
                await manager.shutdown()
            self.assertEqual(service.ensure_count, 0)
            self.assertEqual(service.close_count, 0)
            self.assertEqual(processes[0].returncode, -9)

    async def test_cancelled_spawn_waits_for_child_handle_then_kills_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = make_settings(root)
            login_python = settings["runtime_dir"] / "envs" / "login" / "Scripts" / "python.exe"
            login_python.parent.mkdir(parents=True, exist_ok=True)
            login_python.write_bytes(b"test")
            service = _FakeService()
            spawn_started = asyncio.Event()
            finish_spawn = asyncio.Event()
            processes: list[_FakeProcess] = []

            async def factory(*command, **_kwargs):
                spawn_started.set()
                await finish_spawn.wait()
                job_dir = Path(command[command.index("--job-dir") + 1])
                job_dir.mkdir(parents=True, exist_ok=True)
                process = _FakeProcess(30_001)
                processes.append(process)
                return process

            manager = _TestLoginManager(settings, process_factory=factory, stop_timeout=0.1)
            with _patched_service(service):
                task = asyncio.create_task(manager.start())
                await spawn_started.wait()
                task.cancel()
                await asyncio.sleep(0)
                finish_spawn.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await manager.shutdown()
            self.assertEqual(len(processes), 1)
            self.assertIsNotNone(processes[0].returncode)
            self.assertEqual(list((settings["runtime_dir"] / "active-workers").glob("login-*.json")), [])


if __name__ == "__main__":
    unittest.main()
