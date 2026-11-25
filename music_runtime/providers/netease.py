"""Loopback-only Netease Cloud Music provider and bounded async downloads."""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..settings import (
    PipelineError,
    SelectionRequired,
    _canonical_loopback_url,
    load_settings,
    netease_source_fingerprint,
)


MAX_API_RESPONSE = 2 * 1024 * 1024
MAX_DOWNLOAD_CHUNK = 256 * 1024
DEFAULT_USER_AGENT = "NachoBot-Music-Library/1.0"
_SONG_ID = re.compile(r"^[0-9]{1,20}$")
_CDN_SUFFIXES = (".music.126.net", ".music.163.com")
_COOKIE_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_COOKIE_ATTRIBUTES = {
    "domain", "expires", "httponly", "max-age", "partitioned", "path", "priority",
    "sameparty", "samesite", "secure", "sourceport", "sourcescheme",
}


def _normalize_cookie(cookie: str) -> str:
    """Normalize saved QR cookies that may still contain Set-Cookie attributes."""
    if not isinstance(cookie, str) or len(cookie) > 16 * 1024 or any(char in cookie for char in "\r\n\x00"):
        return ""
    pairs: dict[str, str] = {}
    for item in cookie.split(";"):
        name, separator, value = item.strip().partition("=")
        name = name.strip()
        value = value.strip()
        if not separator or not value or name.casefold() in _COOKIE_ATTRIBUTES:
            continue
        if not _COOKIE_NAME_RE.fullmatch(name) or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
            return ""
        pairs[name] = value
    return "; ".join(f"{name}={value}" for name, value in pairs.items())


class NeteaseProvider:
    """Use one local Netease API and do not forward login cookies to its CDN."""

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = settings or load_settings()
        self.base_url = str(self.settings["netease"]["base_url"]).rstrip("/")
        self.runtime_dir = Path(self.settings["runtime_dir"]).expanduser().resolve()
        self.provider_fingerprint = netease_source_fingerprint(self.settings)
        self._session = None
        self._service = None
        self._closed = False

    async def _ensure_managed_service(self) -> None:
        configured_url = _canonical_loopback_url(self.base_url)
        managed_url = "http://127.0.0.1:13306"
        if configured_url != managed_url:
            return
        if self._service is None:
            try:
                from ..installer import get_netease_service

                service_settings = dict(self.settings)
                service_settings["netease"] = {**self.settings["netease"], "base_url": managed_url}
                self._service = get_netease_service(service_settings)
            except Exception:
                raise PipelineError("netease_runtime_missing", "The managed Netease service is not prepared.") from None
        try:
            result = await self._service.ensure_started()
        except PipelineError:
            raise
        except Exception:
            raise PipelineError("netease_runtime_unavailable", "The managed Netease service could not start.") from None
        base_url = result.get("base_url") if isinstance(result, dict) else result
        if _canonical_loopback_url(base_url) != configured_url:
            raise PipelineError("netease_service_invalid", "The managed Netease service returned an unexpected endpoint.")
        self.base_url = managed_url

    async def _get_session(self):
        if self._closed:
            raise PipelineError("provider_closed", "The Netease provider is closed.")
        if self._session is None:
            try:
                import aiohttp
            except ImportError:
                raise PipelineError("provider_runtime_missing", "The music runtime HTTP dependency is unavailable.") from None
            timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=12)
            self._session = aiohttp.ClientSession(timeout=timeout, cookie_jar=aiohttp.DummyCookieJar())
        return self._session

    async def check_account_ready(self) -> None:
        """Boundedly verify the local API and saved cookie without starting GPU workers."""
        await self._ensure_managed_service()
        cookie = self._cookie()
        try:
            payload = await asyncio.wait_for(
                self._api_request("login/status", {"timestamp": int(time.time() * 1000)}, cookie=cookie),
                timeout=6.0,
            )
        except asyncio.TimeoutError:
            raise PipelineError("netease_auth_timeout", "Netease account validation timed out.") from None
        code = payload.get("code")
        data = payload.get("data")
        if code in {301, 302, 401, 403}:
            raise PipelineError("login_expired", "The saved Netease login has expired; run the login command again.")
        account = data.get("account") if isinstance(data, dict) else None
        if (
            code not in {None, 200}
            or not isinstance(data, dict)
            or data.get("code") != 200
            or not isinstance(account, dict)
            or not account.get("id")
        ):
            raise PipelineError("login_expired", "The saved Netease login is not valid; run the login command again.")

    def _credential_path(self) -> Path:
        profile = self.settings["netease"]["profile"]
        return self.runtime_dir / "credentials" / f"{profile}.json"

    def _cookie(self) -> str:
        try:
            value = json.loads(self._credential_path().read_text(encoding="utf-8"))
            cookie = value.get("cookie") if isinstance(value, dict) else None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            cookie = None
        if not isinstance(cookie, str) or not cookie.strip() or len(cookie) > 16 * 1024:
            raise PipelineError("login_required", "No Netease login is saved; run the login command on this host.")
        cookie = _normalize_cookie(cookie)
        if not cookie:
            raise PipelineError("login_required", "The saved Netease login is invalid; run the login command again.")
        return cookie

    async def _api_request(self, endpoint: str, params: dict[str, Any], *, cookie: str = "") -> dict[str, Any]:
        await self._ensure_managed_service()
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT}
        if cookie:
            headers["Cookie"] = cookie
        try:
            async with session.get(url, params=params, headers=headers, allow_redirects=False) as response:
                status = response.status
                if status in {401, 403} and cookie:
                    raise PipelineError("login_expired", "The Netease login expired; sign in again.")
                # The local API wraps some upstream codes (including -460)
                # in HTTP 400 because they cannot be used as HTTP statuses.
                if status not in {200, 400}:
                    raise PipelineError("provider_http_error", "The local Netease service returned an error.")
                body = await response.content.read(MAX_API_RESPONSE + 1)
        except PipelineError:
            raise
        except asyncio.TimeoutError:
            raise PipelineError("provider_timeout", "The local Netease service timed out.") from None
        except Exception:
            raise PipelineError("provider_unavailable", "The local Netease service is not responding.") from None
        if len(body) > MAX_API_RESPONSE:
            raise PipelineError("provider_response_too_large", "The local Netease service response was too large.")
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if status == 400:
                raise PipelineError("provider_http_error", "The local Netease service returned an error.") from None
            raise PipelineError("provider_invalid_response", "The local Netease service returned invalid data.") from None
        if isinstance(payload, dict) and payload.get("code") == -460:
            raise PipelineError("netease_network_risk", "网易云暂时拒绝了当前网络的请求（-460），请稍后再试。")
        if status != 200:
            raise PipelineError("provider_http_error", "The local Netease service returned an error.")
        if not isinstance(payload, dict):
            raise PipelineError("provider_invalid_response", "The local Netease service returned invalid data.")
        return payload

    @staticmethod
    def _song(song: Any) -> dict[str, Any] | None:
        if not isinstance(song, dict):
            return None
        raw_id = str(song.get("id", ""))
        if not _SONG_ID.fullmatch(raw_id):
            return None
        title = song.get("name")
        artists = song.get("artists") or song.get("ar") or []
        if not isinstance(title, str) or not title.strip() or not isinstance(artists, list):
            return None
        artist_names = []
        for artist in artists[:16]:
            if isinstance(artist, dict) and isinstance(artist.get("name"), str):
                artist_names.append(artist["name"].strip()[:120])
        try:
            duration_ms = int(song.get("duration") or song.get("dt") or 0)
        except (TypeError, ValueError, OverflowError):
            duration_ms = 0
        if duration_ms < 0 or duration_ms > 24 * 60 * 60 * 1000:
            return None
        return {"id": raw_id, "title": title.strip()[:256], "artist": ", ".join(filter(None, artist_names))[:512], "duration_ms": duration_ms}

    async def _by_id(self, song_id: str) -> dict[str, Any]:
        if not _SONG_ID.fullmatch(song_id):
            raise PipelineError("invalid_song", "The Netease song id is invalid.")
        payload = await self._api_request("song/detail", {"ids": song_id})
        if payload.get("code") != 200:
            raise PipelineError("song_not_found", "Netease could not find the selected song.")
        rows = payload.get("songs")
        song = self._song(rows[0]) if isinstance(rows, list) and rows else None
        if not song or song["id"] != song_id:
            raise PipelineError("song_not_found", "Netease could not find the selected song.")
        return song

    @staticmethod
    def _explicit_id(query: str) -> str | None:
        value = query.strip()
        if _SONG_ID.fullmatch(value):
            return value
        try:
            parts = urlsplit(value if "://" in value else f"https://{value}")
        except ValueError:
            return None
        host = (parts.hostname or "").casefold()
        if host != "music.163.com" and not host.endswith(".music.163.com"):
            return None
        values = parse_qs(parts.query)
        if not values.get("id") and parts.fragment:
            values = parse_qs(parts.fragment.split("?", 1)[-1])
        candidate = (values.get("id") or [""])[0]
        return candidate if _SONG_ID.fullmatch(candidate) else None

    async def search(self, query: str) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip() or len(query) > 256:
            raise PipelineError("invalid_query", "Enter a song title to search.")
        payload = await self._api_request("search", {"keywords": query.strip(), "type": 1, "limit": 20})
        if payload.get("code") != 200:
            raise PipelineError("provider_search_failed", "Netease could not complete the song search.")
        result = payload.get("result")
        rows = result.get("songs") if isinstance(result, dict) else None
        if not isinstance(rows, list):
            raise PipelineError("provider_invalid_response", "The local Netease service returned invalid search data.")
        songs = []
        seen: set[str] = set()
        for row in rows[:20]:
            song = self._song(row)
            if song and song["id"] not in seen:
                songs.append(song)
                seen.add(song["id"])
        return songs

    @staticmethod
    def _normalize(text: str) -> str:
        folded = unicodedata.normalize("NFKC", text).casefold()
        return " ".join(re.findall(r"[\w]+", folded, flags=re.UNICODE))

    async def resolve(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip() or len(query) > 512:
            raise PipelineError("invalid_query", "Enter a song title or Netease song link.")
        explicit_id = self._explicit_id(query)
        if explicit_id:
            return await self._by_id(explicit_id)
        songs = await self.search(query)
        if not songs:
            raise PipelineError("song_not_found", "No matching song was found on Netease.")
        target = self._normalize(query)
        normalized = [(song, self._normalize(song["title"]), self._normalize(f"{song['title']} {song['artist']}")) for song in songs]
        exact = [song for song, title, full in normalized if title == target or full == target]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise SelectionRequired(exact)
        scored = []
        for song, title, full in normalized:
            title_score = difflib.SequenceMatcher(None, target, title).ratio()
            full_score = difflib.SequenceMatcher(None, target, full).ratio()
            scored.append((max(title_score, full_score), title_score, song))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, best_title, best = scored[0]
        if best_score < 0.84 or best_title < 0.78:
            raise PipelineError("song_not_found", "No confident Netease match was found.")
        if len(scored) > 1 and best_score - scored[1][0] < 0.06:
            raise SelectionRequired([row[2] for row in scored[:8]])
        if best_score < 0.94:
            raise SelectionRequired([row[2] for row in scored[:8]])
        return best

    @staticmethod
    def _cdn_url(value: str) -> str:
        try:
            parts = urlsplit(value)
        except ValueError:
            parts = None
        if parts is None or parts.scheme not in {"https", "http"} or not parts.hostname or parts.username or parts.password:
            raise PipelineError("provider_invalid_response", "Netease returned an invalid audio URL.")
        host = parts.hostname.casefold()
        if not (host == "music.126.net" or host == "music.163.com" or any(host.endswith(suffix) for suffix in _CDN_SUFFIXES)):
            raise PipelineError("provider_invalid_response", "Netease returned an unsupported audio host.")
        try:
            if parts.port not in {None, 80, 443}:
                raise PipelineError("provider_invalid_response", "Netease returned an unsupported audio host.")
        except ValueError:
            raise PipelineError("provider_invalid_response", "Netease returned an invalid audio URL.") from None
        # Only known Netease hosts are accepted, so upgrading cleartext links
        # does not change the account token or expose Cookie to a CDN.
        return value if parts.scheme == "https" else parts._replace(scheme="https").geturl()

    @staticmethod
    async def _disk_call(function, *args):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(task)
            except Exception:
                pass
            raise

    @staticmethod
    def _valid_audio_prefix(prefix: bytes) -> bool:
        if prefix.startswith((b"RIFF", b"fLaC", b"OggS", b"ID3")):
            return True
        if len(prefix) >= 8 and prefix[4:8] == b"ftyp":
            return True
        return len(prefix) >= 2 and prefix[0] == 0xFF and (prefix[1] & 0xE0) == 0xE0

    async def _download_stream(self, url: str, destination: Path, limit: int) -> Path:
        session = await self._get_session()
        timeout_cls = __import__("aiohttp").ClientTimeout
        timeout = timeout_cls(total=180, connect=10, sock_read=25)
        temporary: str | None = None
        file_obj = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("wb", prefix=f".{destination.name}.", suffix=".part", dir=destination.parent, delete=False) as output:
                temporary = output.name
            file_obj = open(temporary, "wb", buffering=0)
            current = url
            redirects = 0
            total = 0
            announced_total: int | None = None
            prefix = bytearray()
            headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "audio/*,*/*;q=0.8"}
            while True:
                async with session.get(current, headers=headers, allow_redirects=False, timeout=timeout) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        if redirects >= 3:
                            raise PipelineError("download_redirect_limit", "The Netease audio link redirected too many times.")
                        location = response.headers.get("Location")
                        if not location:
                            raise PipelineError("download_failed", "The Netease audio link was incomplete.")
                        from urllib.parse import urljoin
                        current = self._cdn_url(urljoin(current, location))
                        redirects += 1
                        continue
                    if response.status != 200:
                        raise PipelineError("download_http_error", "The Netease audio download failed.")
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        try:
                            announced_total = int(content_length)
                        except ValueError:
                            announced_total = None
                        if announced_total is not None and (announced_total <= 0 or announced_total > limit):
                            raise PipelineError("download_too_large", "The Netease audio exceeds the configured download limit.")
                    async for chunk in response.content.iter_chunked(MAX_DOWNLOAD_CHUNK):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > limit:
                            raise PipelineError("download_too_large", "The Netease audio exceeds the configured download limit.")
                        if len(prefix) < 16:
                            prefix.extend(chunk[: 16 - len(prefix)])
                        await self._disk_call(file_obj.write, chunk)
                    break
            if announced_total is not None and total != announced_total:
                raise PipelineError("download_incomplete", "Netease returned an incomplete audio file.")
            if total < 128 or not self._valid_audio_prefix(bytes(prefix)):
                raise PipelineError("download_invalid_audio", "Netease returned an invalid audio file.")
            await self._disk_call(file_obj.flush)
            await self._disk_call(os.fsync, file_obj.fileno())
            await self._disk_call(file_obj.close)
            file_obj = None
            os.replace(temporary, destination)
            temporary = None
            return destination
        except asyncio.CancelledError:
            raise
        except PipelineError:
            raise
        except asyncio.TimeoutError:
            raise PipelineError("download_timeout", "The Netease audio download timed out.") from None
        except Exception:
            raise PipelineError("download_failed", "The Netease audio download failed.") from None
        finally:
            if file_obj is not None:
                try:
                    await self._disk_call(file_obj.close)
                except Exception:
                    pass
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _playable_record(payload: dict[str, Any], song_id: str) -> dict[str, Any] | None:
        code = payload.get("code")
        if code in {301, 302, 401, 403}:
            raise PipelineError("login_expired", "The Netease login expired; sign in again.")
        if code != 200:
            return None
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return None
        record = rows[0]
        if str(record.get("id", "")) != song_id:
            raise PipelineError("provider_invalid_response", "Netease returned a different song id.")
        privilege = record.get("freeTrialPrivilege")
        if record.get("freeTrialInfo") is not None or (
            isinstance(privilege, dict) and (privilege.get("startTime") or privilege.get("endTime"))
        ):
            raise PipelineError("trial_audio", "Netease only offered a preview for this account.")
        if record.get("code") not in {None, 200} or not record.get("url"):
            return None
        return record

    async def _lookup_audio(self, song_id: str, cookie: str) -> dict[str, Any] | None:
        for level in ("lossless", "exhigh", "standard"):
            try:
                payload = await self._api_request(
                    "song/url/v1", {"id": song_id, "level": level, "unblock": "false"}, cookie=cookie,
                )
            except PipelineError as exc:
                if exc.code != "netease_network_risk":
                    raise
                # Keep the same song and account on the API's older playback endpoint.
                payload = await self._api_request(
                    "song/url", {"id": song_id, "br": 320000, "unblock": "false"}, cookie=cookie,
                )
                return self._playable_record(payload, song_id)
            record = self._playable_record(payload, song_id)
            if record is not None:
                return record
        return None

    async def download(self, song: dict[str, Any], destination: Path) -> Path:
        if not isinstance(song, dict):
            raise PipelineError("invalid_song", "A resolved Netease song record is required.")
        song_id = str(song.get("id", ""))
        if not _SONG_ID.fullmatch(song_id):
            raise PipelineError("invalid_song", "The selected Netease song id is invalid.")
        record = await self._lookup_audio(song_id, self._cookie())
        if not record:
            raise PipelineError("download_unavailable", "Netease did not provide a complete recording for this account.")
        raw_url = record.get("url")
        if not isinstance(raw_url, str) or not raw_url or len(raw_url) > 2048:
            raise PipelineError("download_unavailable", "Netease did not provide a complete recording for this account.")
        url = self._cdn_url(raw_url)
        destination = Path(destination).expanduser().resolve()
        try:
            destination.relative_to(self.runtime_dir.resolve(strict=True))
        except (OSError, ValueError):
            raise PipelineError("path_outside_runtime", "Audio downloads must stay inside the managed music runtime.") from None
        return await self._download_stream(url, destination, int(self.settings["pipeline"]["max_download_bytes"]))

    async def close(self) -> None:
        self._closed = True
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._service is not None:
            await self._service.close()
            self._service = None


__all__ = ["NeteaseProvider"]
