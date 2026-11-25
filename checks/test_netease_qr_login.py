from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import ProxyHandler
from unittest.mock import patch


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = RUNTIME_ROOT / "music_runtime" / "workers" / "netease_qr_login.py"
PINNED_SERVER = RUNTIME_ROOT / "checks" / "fixtures" / "NeteaseCloudMusicApi" / "server.js"
_spec = importlib.util.spec_from_file_location("netease_qr_login_worker", WORKER_PATH)
assert _spec is not None and _spec.loader is not None
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


class _FakeResponse:
    def __init__(self, payload: dict):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, size: int) -> bytes:
        return self.body[:size]


class _FakeOpener:
    def __init__(self, calls: list[tuple[str, str, str]]):
        self.calls = calls

    def open(self, request, timeout: float):
        parsed = urlsplit(request.full_url)
        endpoint = parsed.path.rsplit("/", 1)[-1]
        cookie = request.get_header("Cookie") or ""
        self.calls.append((endpoint, request.full_url, cookie))
        if endpoint == "key":
            payload = {"code": 200, "data": {"unikey": "qr-key-for-test"}}
        elif endpoint == "create":
            payload = {"code": 200, "data": {"qrurl": "https://music.163.com/login?code=qr-only"}}
        elif endpoint == "check":
            payload = {"code": 803, "cookie": _SET_COOKIE_JOINED}
        elif endpoint == "status":
            # Pinned module/login_status.js returns data.code without a root code.
            payload = {"data": {"code": 200, "account": {"id": 12345}}}
        else:
            raise AssertionError(f"Unexpected endpoint: {endpoint}")
        return _FakeResponse(payload)


_SET_COOKIE_JOINED = (
    "MUSIC_U=music-token-for-test; Max-Age=15552000; Path=/; HttpOnly;"
    "__csrf=csrf-token-for-test; Path=/; NMTID=device-token-for-test;"
    "Domain=.music.163.com; Secure"
)
_PINNED_COOKIE_PARSER = r"""
const fs = require('fs')
const header = fs.readFileSync(0, 'utf8')
const cookies = {}
;(header || '').split(/;\s+|(?<!\s)\s+$/g).forEach((pair) => {
  let crack = pair.indexOf('=')
  if (crack < 1 || crack == pair.length - 1) return
  cookies[pair.slice(0, crack).trim()] = pair.slice(crack + 1).trim()
})
process.stdout.write(JSON.stringify(cookies))
"""


class NeteaseQrWorkerTests(unittest.TestCase):
    def test_cookie_normalization_matches_pinned_server_parser(self):
        if not PINNED_SERVER.is_file():
            self.skipTest("Pinned local Netease service source is not installed")
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required to exercise the pinned JavaScript cookie parser")
        source = PINNED_SERVER.read_text(encoding="utf-8")
        self.assertIn(".split(/;\\s+|(?<!\\s)\\s+$/g)", source)
        self.assertIn("pair.slice(crack + 1)", source)

        def parse_cookie(header: str) -> dict[str, str]:
            result = subprocess.run(
                [node, "-e", _PINNED_COOKIE_PARSER],
                input=header,
                text=True,
                capture_output=True,
                check=True,
                timeout=5,
            )
            return json.loads(result.stdout)

        before = parse_cookie(_SET_COOKIE_JOINED)
        self.assertEqual(before["MUSIC_U"], "music-token-for-test")
        self.assertNotIn("__csrf", before)

        normalized = worker.normalize_cookie(_SET_COOKIE_JOINED)
        after = parse_cookie(normalized)
        self.assertEqual(after, {
            "MUSIC_U": "music-token-for-test",
            "__csrf": "csrf-token-for-test",
            "NMTID": "device-token-for-test",
        })
        self.assertNotIn("Path", after)
        self.assertNotIn("Max-Age", after)
        self.assertNotIn("Domain", after)

    def test_worker_validates_effective_cookie_with_fresh_timestamp_and_pinned_body_shape(self):
        calls: list[tuple[str, str, str]] = []
        opener_arguments: list[tuple] = []

        def build_opener(*handlers):
            opener_arguments.append(handlers)
            return _FakeOpener(calls)

        with tempfile.TemporaryDirectory() as folder:
            job_dir = Path(folder) / "runtime" / "login" / "job"
            job_dir.mkdir(parents=True)
            with (
                patch.object(worker, "build_opener", side_effect=build_opener),
                patch.object(worker, "_make_qr", side_effect=lambda _url, path: path.write_bytes(b"test")),
                patch.object(worker.time, "sleep"),
            ):
                result = worker.run("http://127.0.0.1:13306", "test", job_dir, max_wait_seconds=5)

            self.assertEqual(result, 0)
            self.assertEqual([call[0] for call in calls], ["key", "create", "check", "status"])
            status_url = next(url for endpoint, url, _cookie in calls if endpoint == "status")
            timestamp = int(parse_qs(urlsplit(status_url).query)["timestamp"][0])
            self.assertLess(abs(int(time.time() * 1000) - timestamp), 60_000)
            status_cookie = next(cookie for endpoint, _url, cookie in calls if endpoint == "status")
            self.assertEqual(
                status_cookie,
                "MUSIC_U=music-token-for-test; __csrf=csrf-token-for-test; NMTID=device-token-for-test",
            )
            self.assertNotIn("music-token-for-test", status_url)
            self.assertNotIn("csrf-token-for-test", status_url)
            self.assertEqual(len(opener_arguments), 4)
            for handlers in opener_arguments:
                proxy = next(handler for handler in handlers if isinstance(handler, ProxyHandler))
                self.assertEqual(proxy.proxies, {})
                self.assertTrue(any(isinstance(handler, worker._NoRedirect) for handler in handlers))

            credential = json.loads((Path(folder) / "runtime" / "credentials" / "test.json").read_text(encoding="utf-8"))
            self.assertEqual(credential["cookie"], status_cookie)
            self.assertEqual(json.loads((job_dir / "status.json").read_text(encoding="utf-8"))["status"], "success")

    def test_transient_qr_check_failure_has_specific_safe_code(self):
        class _UnavailableOpener:
            def open(self, _request, timeout: float):
                from urllib.error import URLError

                raise URLError("synthetic network failure")

        with patch.object(worker, "build_opener", return_value=_UnavailableOpener()):
            with self.assertRaises(worker.LoginFailure) as caught:
                worker._request("http://127.0.0.1:13306", "login/qr/check", {"timestamp": 1})
        self.assertEqual(caught.exception.code, "qr_check_unavailable")
        self.assertNotIn("synthetic", str(caught.exception))

    def test_qr_check_http_errors_distinguish_transient_from_permanent(self):
        class _HttpErrorOpener:
            def __init__(self, status: int):
                self.status = status

            def open(self, request, timeout: float):
                raise HTTPError(request.full_url, self.status, "synthetic HTTP failure", None, None)

        for status, expected in (
            (429, "qr_check_unavailable"),
            (503, "qr_check_unavailable"),
            (400, "qr_check_failed"),
        ):
            with self.subTest(status=status):
                with patch.object(worker, "build_opener", return_value=_HttpErrorOpener(status)):
                    with self.assertRaises(worker.LoginFailure) as caught:
                        worker._request("http://127.0.0.1:13306", "login/qr/check", {"timestamp": 1})
                self.assertEqual(caught.exception.code, expected)
                self.assertNotIn("synthetic", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
