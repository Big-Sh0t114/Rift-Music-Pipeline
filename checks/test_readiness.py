from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from music_runtime.providers.netease import NeteaseProvider  # noqa: E402
from music_runtime.settings import PipelineError  # noqa: E402


class ProviderReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_validates_cookie_account_and_reports_expiry_without_exposure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="music-runtime-readiness-") as folder:
            root = Path(folder)
            credentials = root / "credentials"
            credentials.mkdir()
            (credentials / "operator.json").write_text(
                json.dumps({"cookie": "MUSIC_U=fixture-token; Path=/"}), encoding="utf-8"
            )
            settings = {
                "runtime_dir": root,
                "plugin_dir": root,
                "netease": {"base_url": "http://127.0.0.1:13306", "profile": "operator"},
                "pipeline": {"max_download_bytes": 1024},
            }
            provider = NeteaseProvider(settings)
            provider._ensure_managed_service = AsyncMock()
            provider._api_request = AsyncMock(return_value={
                "code": 200, "data": {"code": 200, "account": {"id": 123}},
            })
            await provider.check_account_ready()
            provider._api_request.assert_awaited_once()
            self.assertNotIn("fixture-token", str(provider._api_request.await_args.args))

            provider._api_request = AsyncMock(return_value={"code": 200, "data": {"code": 200}})
            with self.assertRaises(PipelineError) as expired:
                await provider.check_account_ready()
            self.assertEqual(expired.exception.code, "login_expired")
            self.assertNotIn("fixture-token", str(expired.exception))

    async def test_missing_cookie_has_separate_safe_readiness_reason(self) -> None:
        with tempfile.TemporaryDirectory(prefix="music-runtime-no-cookie-") as folder:
            root = Path(folder)
            provider = NeteaseProvider({
                "runtime_dir": root,
                "plugin_dir": root,
                "netease": {"base_url": "http://127.0.0.1:13306", "profile": "operator"},
                "pipeline": {"max_download_bytes": 1024},
            })
            provider._ensure_managed_service = AsyncMock()
            provider._api_request = AsyncMock()
            with self.assertRaises(PipelineError) as missing:
                await provider.check_account_ready()
            self.assertEqual(missing.exception.code, "login_required")
            provider._api_request.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
