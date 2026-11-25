"""Authenticated, versioned HTTP service for durable music production jobs."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from aiohttp import web

from .job_store import JobStore, RequestConflict
from .pipeline import MusicPipeline
from .settings import PipelineError, load_settings


_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
_MAX_QUERY = 512
_MAX_BODY = 64 * 1024
MUSIC_SERVICE_KEY = web.AppKey("music_service", object)


class ServiceAlreadyRunning(RuntimeError):
    """Another service process owns the configured data directory."""


class ReadinessError(PipelineError):
    """A new job was rejected because a required production component is not ready."""

    readiness = True


class ServiceProcessLock:
    """OS advisory lock that remains held while this process owns SQLite."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise ServiceAlreadyRunning("another music service owns this data directory") from exc
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise ServiceAlreadyRunning("another music service owns this data directory") from exc
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()).encode("ascii"))
            handle.flush()
            self._handle = handle
        except BaseException:
            handle.close()
            raise

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class MusicService:
    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        *,
        pipeline: Any = None,
        store: JobStore | None = None,
    ):
        self.settings = settings if settings is not None else load_settings()
        self.data_dir = Path(self.settings["runtime_dir"]).expanduser().resolve()
        self.artifact_dir = self.data_dir / "artifacts"
        self._lock = ServiceProcessLock(self.data_dir / "service.lock")
        self._lock.acquire()
        try:
            self.store = store or (pipeline.store if pipeline is not None else JobStore(self.data_dir / "jobs.sqlite"))
            self.pipeline = pipeline or MusicPipeline(self.settings, store=self.store)
            self._started = False
            self._closed = False
            self._artifact_cache: dict[str, tuple[int, int, str]] = {}
            self._readiness_cache: tuple[float, dict[str, Any]] | None = None
        except BaseException:
            self._lock.release()
            raise

    async def startup(self, app: web.Application) -> None:
        if self._closed:
            raise RuntimeError("music service has already closed")
        try:
            await self.pipeline.ensure_started()
            self._started = True
            if getattr(self.pipeline, "enabled", True):
                provider = getattr(self.pipeline, "provider", None)
                ensure_managed = getattr(provider, "_ensure_managed_service", None)
                if ensure_managed is not None:
                    try:
                        await asyncio.wait_for(ensure_managed(), timeout=25.0)
                    except Exception:
                        # Keep the API alive so health/status can report setup state.
                        pass
        except BaseException:
            self._lock.release()
            raise

    async def close(self, app: web.Application | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.pipeline.close()
        finally:
            self._lock.release()

    async def health(self) -> dict[str, Any]:
        now = time.monotonic()
        cached = self._readiness_cache
        if cached is not None and cached[0] > now:
            return dict(cached[1])
        result: dict[str, Any] = {"alive": True, "ready": False}
        if not self._started:
            result["reason"] = "service_starting"
            return self._cache_readiness(result)
        if not getattr(self.pipeline, "enabled", True):
            result["reason"] = "pipeline_disabled"
            return self._cache_readiness(result)
        provider = getattr(self.pipeline, "provider", None)
        if provider is None or not hasattr(provider, "check_account_ready"):
            result["reason"] = "netease_provider_unavailable"
            return self._cache_readiness(result)
        try:
            await asyncio.wait_for(provider.check_account_ready(), timeout=8.0)
        except PipelineError as exc:
            result["reason"] = exc.code[:64]
            return self._cache_readiness(result)
        except asyncio.TimeoutError:
            result["reason"] = "netease_auth_timeout"
            return self._cache_readiness(result)
        except Exception:
            result["reason"] = "netease_provider_unavailable"
            return self._cache_readiness(result)
        runtime = getattr(self.pipeline, "runtime", None)
        if runtime is None:
            result["reason"] = "runtime_not_prepared"
            return self._cache_readiness(result)
        try:
            snapshot = await runtime.check_ready()
            components = snapshot.get("components", {}) if isinstance(snapshot, dict) else {}
            rift = components.get("rift", {}) if isinstance(components, dict) else {}
            separation = components.get("separation", {}) if isinstance(components, dict) else {}
            netease = components.get("netease", {}) if isinstance(components, dict) else {}
            if not bool(rift.get("ready")):
                result["reason"] = "rift_not_ready"
            elif not bool(separation.get("ready")):
                result["reason"] = "audio_models_not_ready"
            elif not bool(netease.get("ready")):
                result["reason"] = "netease_provider_unavailable"
            elif isinstance(snapshot, dict) and bool(snapshot.get("ready")):
                result["ready"] = True
            else:
                result["reason"] = "runtime_not_ready"
        except PipelineError as exc:
            result["reason"] = exc.code[:64]
        except Exception:
            result["reason"] = "health_check_failed"
        return self._cache_readiness(result)

    def _cache_readiness(self, result: dict[str, Any]) -> dict[str, Any]:
        self._readiness_cache = (time.monotonic() + 3.0, dict(result))
        return result

    async def submit(self, request_id: str, query: str) -> dict[str, Any]:
        try:
            existing = await asyncio.to_thread(self.store.lookup_request, request_id, query)
        except RequestConflict:
            raise
        if existing is not None:
            return existing
        health = await self.health()
        if not health["ready"]:
            # Recheck after the probe so a concurrent accepted request can
            # still replay its durable identity during a readiness transition.
            existing = await asyncio.to_thread(self.store.lookup_request, request_id, query)
            if existing is not None:
                return existing
            reason = str(health.get("reason") or "runtime_not_ready")[:64]
            raise ReadinessError(reason, "The music runtime is not ready for new jobs.")
        return await self.pipeline.submit(request_id, query)

    async def choose(self, job_id: str, candidate_id: str) -> dict[str, Any]:
        return await self.pipeline.choose(job_id, candidate_id)

    async def job(self, job_id: str) -> dict[str, Any] | None:
        value = await asyncio.to_thread(self.store.get_job_view, job_id)
        if value is None:
            return None
        public = _public_job(value)
        if value.get("status") == "ready":
            info = await asyncio.to_thread(self._artifact_info, value)
            if info is None:
                canonical_id = str(value.get("canonical_id", job_id))
                await asyncio.to_thread(self.store.invalidate_reusable_job, canonical_id)
                refreshed = await asyncio.to_thread(self.store.get_job_view, job_id)
                if refreshed is not None:
                    public = _public_job(refreshed)
            else:
                public["artifact"] = info["metadata"]
        return public

    async def artifact_path(self, job_id: str) -> tuple[Path, dict[str, Any]] | None:
        value = await asyncio.to_thread(self.store.get_job_view, job_id)
        if value is None or value.get("status") != "ready":
            return None
        info = await asyncio.to_thread(self._artifact_info, value)
        if info is None:
            canonical_id = str(value.get("canonical_id", job_id))
            await asyncio.to_thread(self.store.invalidate_reusable_job, canonical_id)
            return None
        return info["path"], info["metadata"]

    def _artifact_info(self, job: dict[str, Any]) -> dict[str, Any] | None:
        canonical_id = str(job.get("canonical_id") or job.get("id") or "")
        if not _JOB_ID.fullmatch(canonical_id):
            return None
        artifact = job.get("artifacts", {}).get("output", {}).get("wav")
        if not isinstance(artifact, str) or not artifact:
            return None
        expected = self.artifact_dir / f"{canonical_id}.wav"
        try:
            if expected.is_symlink() or not expected.is_file():
                return None
            path = expected.resolve(strict=True)
            recorded = Path(artifact).expanduser().resolve(strict=True)
            if path != recorded or path.parent != self.artifact_dir.resolve(strict=True):
                return None
            stat = path.stat()
            if stat.st_size <= 0:
                return None
            cached = self._artifact_cache.get(str(path))
            if cached and cached[:2] == (stat.st_size, stat.st_mtime_ns):
                digest = cached[2]
            else:
                hasher = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(block)
                digest = hasher.hexdigest()
                self._artifact_cache[str(path)] = (stat.st_size, stat.st_mtime_ns, digest)
            return {
                "path": path,
                "metadata": {"size": stat.st_size, "sha256": digest},
            }
        except (OSError, ValueError):
            return None


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    raw_song = job.get("song")
    song = {}
    if isinstance(raw_song, dict) and raw_song.get("id"):
        song = {key: str(raw_song.get(key, ""))[:200] for key in ("id", "title", "artist")}
    candidates: list[dict[str, str]] = []
    raw_candidates = job.get("candidates")
    if isinstance(raw_candidates, list):
        for item in raw_candidates[:8]:
            if isinstance(item, dict):
                candidates.append({key: str(item.get(key, ""))[:200] for key in ("id", "title", "artist")})
    result = {
        "id": str(job.get("id", "")),
        "status": str(job.get("status", "unknown")),
        "stage": str(job.get("stage", ""))[:64],
        "progress": str(job.get("progress", ""))[:200],
        "query": str(job.get("query", ""))[:_MAX_QUERY],
        "song": song,
        "candidates": candidates,
    }
    for key in ("error_code", "error_message"):
        value = str(job.get(key) or "")[:300]
        if value:
            result[key] = value
    return result


def _error(status: int, code: str, message: str) -> web.Response:
    response = web.json_response({"error_code": code, "error_message": message[:300]}, status=status)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _pipeline_error(exc: PipelineError) -> web.Response:
    if getattr(exc, "readiness", False):
        status = 503
    elif exc.code == "queue_full":
        status = 429
    elif exc.code == "selection_conflict":
        status = 409
    else:
        status = 422
    return _error(status, exc.code, exc.message)


def create_app(service: MusicService, token: str) -> web.Application:
    if not isinstance(token, str) or len(token.encode("utf-8")) < 32:
        raise ValueError("NACHOBOT_MUSIC_API_TOKEN must contain at least 32 bytes")
    expected = ("Bearer " + token).encode("utf-8")

    @web.middleware
    async def authenticate(request: web.Request, handler):
        supplied = request.headers.get("Authorization", "").encode("utf-8", errors="ignore")
        if not hmac.compare_digest(supplied, expected):
            return _error(401, "unauthorized", "A valid bearer token is required.")
        try:
            return await handler(request)
        except web.HTTPException as exc:
            return _error(exc.status, "http_error", "The requested route is unavailable.")

    async def health(request: web.Request) -> web.Response:
        return web.json_response(await service.health(), headers={"Cache-Control": "no-store"})

    async def read_json(request: web.Request) -> dict[str, Any] | web.Response:
        if request.content_type != "application/json":
            return _error(415, "invalid_content_type", "Send a JSON request body.")
        if request.content_length is not None and request.content_length > _MAX_BODY:
            return _error(413, "request_too_large", "The request body is too large.")
        try:
            value = await request.json(loads=json.loads)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return _error(400, "invalid_json", "The request body must be valid JSON.")
        if not isinstance(value, dict):
            return _error(400, "invalid_json", "The JSON request body must be an object.")
        return value

    async def submit(request: web.Request) -> web.Response:
        body = await read_json(request)
        if isinstance(body, web.Response):
            return body
        request_id = body.get("request_id")
        query = body.get("query")
        if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
            return _error(400, "invalid_request_id", "request_id must be 32 lowercase hexadecimal characters.")
        if (
            not isinstance(query, str)
            or not query
            or len(query) > _MAX_QUERY
            or query != query.strip()
            or any(ord(char) < 32 and char not in "\t" for char in query)
        ):
            return _error(400, "invalid_query", "query must be trimmed text of at most 512 characters.")
        try:
            job = await service.submit(request_id, query)
        except sqlite3.Error:
            return _error(503, "state_unavailable", "The durable job store is unavailable.")
        except ValueError as exc:
            if isinstance(exc, RequestConflict):
                return _error(409, "request_id_conflict", "request_id is already bound to a different query.")
            return _error(400, "invalid_request", "The request could not be accepted.")
        except PipelineError as exc:
            return _pipeline_error(exc)
        return web.json_response(_public_job(job), status=202, headers={"Cache-Control": "no-store"})

    async def get_job(request: web.Request) -> web.Response:
        job_id = request.match_info["job_id"]
        if not _JOB_ID.fullmatch(job_id):
            return _error(404, "job_not_found", "No matching music job was found.")
        job = await service.job(job_id)
        if job is None:
            return _error(404, "job_not_found", "No matching music job was found.")
        return web.json_response(job, headers={"Cache-Control": "no-store"})

    async def choose(request: web.Request) -> web.Response:
        job_id = request.match_info["job_id"]
        if not _JOB_ID.fullmatch(job_id):
            return _error(404, "job_not_found", "No matching music job was found.")
        body = await read_json(request)
        if isinstance(body, web.Response):
            return body
        candidate_id = body.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or len(candidate_id) > 128:
            return _error(400, "invalid_candidate_id", "candidate_id must be non-empty text.")
        try:
            job = await service.choose(job_id, candidate_id)
        except PipelineError as exc:
            if exc.code == "job_not_found":
                return _error(404, exc.code, exc.message)
            return _pipeline_error(exc)
        return web.json_response(_public_job(job), headers={"Cache-Control": "no-store"})

    async def artifact(request: web.Request) -> web.StreamResponse:
        job_id = request.match_info["job_id"]
        if not _JOB_ID.fullmatch(job_id):
            return _error(404, "job_not_found", "No matching music job was found.")
        job = await asyncio.to_thread(service.store.get_job_view, job_id)
        if job is None:
            return _error(404, "job_not_found", "No matching music job was found.")
        if job.get("status") != "ready":
            return _error(409, "artifact_not_ready", "The job has not completed yet.")
        info = await service.artifact_path(job_id)
        if info is None:
            return _error(404, "artifact_unavailable", "The completed WAV artifact is unavailable.")
        path, metadata = info
        response = web.FileResponse(path, headers={
            "Content-Type": "audio/wav",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Artifact-SHA256": metadata["sha256"],
        })
        return response

    app = web.Application(client_max_size=_MAX_BODY, middlewares=[authenticate])
    app[MUSIC_SERVICE_KEY] = service
    app.router.add_get("/v1/health", health)
    app.router.add_post("/v1/jobs", submit)
    app.router.add_get("/v1/jobs/{job_id}", get_job)
    app.router.add_post("/v1/jobs/{job_id}/selection", choose)
    app.router.add_get("/v1/jobs/{job_id}/artifact", artifact)
    app.on_startup.append(service.startup)
    app.on_cleanup.append(service.close)
    return app


def serve(host: str | None = None, port: int | None = None) -> None:
    token = os.environ.get("NACHOBOT_MUSIC_API_TOKEN", "")
    if len(token.encode("utf-8")) < 32:
        raise RuntimeError("Set NACHOBOT_MUSIC_API_TOKEN to a random value of at least 32 bytes")
    listen_host = host or os.environ.get("MUSIC_RUNTIME_HOST", "127.0.0.1")
    if listen_host not in {"127.0.0.1", "localhost", "::1", "0.0.0.0", "::"}:
        raise RuntimeError("MUSIC_RUNTIME_HOST must be a literal loopback or wildcard address")
    listen_port = int(port if port is not None else os.environ.get("MUSIC_RUNTIME_PORT", "8765"))
    if not 1 <= listen_port <= 65535:
        raise RuntimeError("MUSIC_RUNTIME_PORT must be between 1 and 65535")
    service = MusicService()
    try:
        web.run_app(create_app(service, token), host=listen_host, port=listen_port, access_log=None)
    finally:
        if not service._closed:
            asyncio.run(service.close())


__all__ = ["MusicService", "MUSIC_SERVICE_KEY", "ServiceAlreadyRunning", "ServiceProcessLock", "create_app", "serve"]
