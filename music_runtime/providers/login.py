"""Safe, generation-aware Netease QR login manager for the music setup UI."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..settings import PipelineError, _canonical_loopback_url, load_settings


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_QR_BYTES = 2 * 1024 * 1024
MAX_STATUS_BYTES = 16 * 1024
ProcessFactory = Callable[..., Awaitable[Any]]
_SAFE_MESSAGES = {
    "starting": "正在申请网易云音乐二维码…",
    "waiting": "请用网易云音乐 App 扫描二维码。",
    "scanned": "已扫码，请在网易云音乐 App 确认登录。",
    "success": "网易云音乐登录成功，凭据已安全保存。",
    "expired": "二维码已过期，请重新生成。",
    "error": "网易云音乐登录失败，请检查服务后重试。",
    "cancelled": "登录已取消。",
}
_SAFE_ERROR_MESSAGES = {
    "qr_request_failed": "网易云音乐二维码申请失败，请检查登录服务后重试。",
    "qr_check_unavailable": "网易云音乐登录状态服务暂时不可用，请稍后重试。",
    "qr_check_failed": "二维码登录状态查询失败，请重新申请二维码。",
    "account_validation_unavailable": "登录已确认，但账号验证服务暂时不可用，请稍后重试。",
    "account_validation_failed": "账号验证失败，请重新进行网易云音乐登录。",
    "internal_error": _SAFE_MESSAGES["error"],
}
_TERMINAL = {"success", "expired", "error", "cancelled"}


@dataclass
class _LoginJob:
    job_id: str
    directory: Path
    started_at: float
    process: Any = None
    status: str = "starting"
    error_code: str | None = None
    monitor: asyncio.Task[Any] | None = field(default=None, repr=False)
    active_marker: Path | None = field(default=None, repr=False)

    @property
    def qr_path(self) -> Path:
        return self.directory / "qr.png"

    @property
    def status_path(self) -> Path:
        return self.directory / "status.json"


class NeteaseLoginManager:
    """Manage one fixed QR helper child without exposing its credential output."""

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        *,
        process_factory: ProcessFactory | None = None,
        poll_interval: float = 0.25,
        stop_timeout: float = 5.0,
    ) -> None:
        self.settings = settings or load_settings()
        self.runtime_dir = Path(self.settings["runtime_dir"]).expanduser().resolve()
        self.root = self.runtime_dir / "login"
        self.helper = Path(__file__).resolve().parents[1] / "workers" / "netease_qr_login.py"
        self.python = self._login_python()
        self._process_factory = process_factory
        self._poll_interval = max(0.03, float(poll_interval))
        self._stop_timeout = max(0.1, float(stop_timeout))
        self._lock = asyncio.Lock()
        self._current: _LoginJob | None = None
        self._closed = False
        self._lifecycle_failed = False
        self._service = None

    def _login_python(self) -> Path:
        folder = self.runtime_dir / "envs" / "login"
        return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    async def startup(self) -> None:
        """Re-open the manager for a new UI lifespan without starting processes."""
        async with self._lock:
            self._closed = False

    async def start(self) -> dict[str, Any]:
        async with self._lock:
            if self._closed:
                self._closed = False
            await self._stop_current_unlocked("cancelled")
            job_id = secrets.token_urlsafe(18)
            directory = (self.root / job_id).resolve()
            try:
                directory.relative_to(self.root.resolve())
                directory.mkdir(parents=True, exist_ok=False)
            except (OSError, ValueError):
                raise PipelineError("login_storage_failed", "The Netease login workspace could not be created.") from None
            job = _LoginJob(job_id=job_id, directory=directory, started_at=asyncio.get_running_loop().time())
            self._current = job
            try:
                if not self.helper.is_file() or not self.python.is_file():
                    job.status = "error"
                    self._write_safe_status(job, "error")
                    return self._snapshot(job)
                configured_url = _canonical_loopback_url(str(self.settings["netease"]["base_url"]))
                if configured_url is None:
                    raise PipelineError("netease_service_invalid", "The configured Netease endpoint is invalid.")
                base_url = configured_url
                if configured_url == "http://127.0.0.1:13306":
                    if self._service is None:
                        from ..installer import get_netease_service

                        service_settings = dict(self.settings)
                        service_settings["netease"] = {**self.settings["netease"], "base_url": configured_url}
                        self._service = get_netease_service(service_settings)
                    service_info = await self._service.ensure_started()
                    service_base = service_info.get("base_url") if isinstance(service_info, dict) else service_info
                    if _canonical_loopback_url(service_base) != configured_url:
                        raise PipelineError("netease_service_invalid", "The managed Netease service is unavailable.")
                    base_url = configured_url
                profile = str(self.settings["netease"]["profile"])
                command = [
                    str(self.python), "-u", str(self.helper),
                    "--base-url", base_url,
                    "--profile", profile,
                    "--job-dir", str(job.directory),
                ]
                env = dict(os.environ)
                for name in tuple(env):
                    if name.upper() in {"PYTHONPATH", "PYTHONHOME"}:
                        env.pop(name, None)
                env["PYTHONUTF8"] = "1"
                factory = self._process_factory or asyncio.create_subprocess_exec
                kwargs: dict[str, Any] = {
                    "cwd": str(self.runtime_dir),
                    "stdin": asyncio.subprocess.DEVNULL,
                    "stdout": asyncio.subprocess.DEVNULL,
                    "stderr": asyncio.subprocess.DEVNULL,
                    "env": env,
                }
                if os.name == "nt":
                    kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                else:
                    kwargs["start_new_session"] = True
                spawn_task = asyncio.create_task(factory(*command, **kwargs))
                try:
                    job.process = await asyncio.shield(spawn_task)
                except asyncio.CancelledError:
                    # Process creation can complete after the request task is
                    # cancelled. Reap the creation result so the outer cancel
                    # handler can terminate the exact child before returning.
                    try:
                        job.process = await asyncio.shield(spawn_task)
                    except Exception:
                        pass
                    raise
                try:
                    self._write_active_marker(job)
                except PipelineError:
                    await self._terminate_process(job.process)
                    raise
                job.status = "starting"
                job.monitor = asyncio.create_task(self._wait_job(job))
            except asyncio.CancelledError:
                await self._stop_current_unlocked("cancelled")
                raise
            except PipelineError:
                job.status = "error"
                self._write_safe_status(job, "error")
            except Exception:
                job.status = "error"
                self._write_safe_status(job, "error")
            return self._snapshot(job)

    async def status(self, job_id: str) -> dict[str, Any] | None:
        async with self._lock:
            job = self._current
            if job is None or not secrets.compare_digest(str(job_id), job.job_id):
                return None
            self._refresh_status(job)
            return self._snapshot(job)

    async def read_qr(self, job_id: str) -> bytes | None:
        async with self._lock:
            job = self._current
            if job is None or not secrets.compare_digest(str(job_id), job.job_id):
                return None
            self._refresh_status(job)
            if job.status not in {"waiting", "scanned"}:
                return None
            try:
                payload = job.qr_path.read_bytes()
            except OSError:
                return None
            if len(payload) > MAX_QR_BYTES or not self._complete_png(payload):
                return None
            return payload

    async def cancel(self, job_id: str) -> dict[str, Any] | None:
        async with self._lock:
            job = self._current
            if job is None or not secrets.compare_digest(str(job_id), job.job_id):
                return None
            await self._stop_current_unlocked("cancelled")
            return self._snapshot(job)

    async def shutdown(self) -> None:
        async with self._lock:
            self._closed = True
            await self._stop_current_unlocked("cancelled")
            service = self._service
            self._service = None
            if service is not None:
                try:
                    await service.close()
                except Exception:
                    self._lifecycle_failed = True

    async def _wait_job(self, job: _LoginJob) -> None:
        process = job.process
        if process is None:
            return
        try:
            await process.wait()
            self._remove_active_marker(job)
            self._refresh_status(job)
            if job.status not in _TERMINAL:
                job.status = "error"
                self._write_safe_status(job, "error")
            if job.status in _TERMINAL:
                self._cleanup_qr(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._lifecycle_failed = True
            job.status = "error"
            self._write_safe_status(job, "error")

    def _write_active_marker(self, job: _LoginJob) -> None:
        process = job.process
        if process is None or not isinstance(getattr(process, "pid", None), int):
            raise PipelineError("runtime_marker_failed", "The login helper could not register its active process.")
        marker_dir = self.runtime_dir / "active-workers"
        nonce = secrets.token_urlsafe(18)
        marker = marker_dir / f"login-{process.pid}-{nonce}.json"
        temporary: str | None = None
        try:
            marker_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="\n", prefix=".login-", suffix=".tmp",
                dir=marker_dir, delete=False,
            ) as output:
                temporary = output.name
                json.dump({"component": "login", "pid": process.pid, "nonce": nonce}, output, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, marker)
            temporary = None
            job.active_marker = marker
        except OSError:
            raise PipelineError("runtime_marker_failed", "The login helper could not register its active process.") from None
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def _remove_active_marker(self, job: _LoginJob) -> None:
        marker = job.active_marker
        job.active_marker = None
        if marker is None:
            return
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            self._lifecycle_failed = True

    def _refresh_status(self, job: _LoginJob) -> None:
        if job.status in _TERMINAL:
            return
        try:
            raw = job.status_path.read_bytes()
            if len(raw) > MAX_STATUS_BYTES:
                return
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            if job.process is not None and getattr(job.process, "returncode", None) is not None:
                job.status = "error"
            return
        state = payload.get("status") if isinstance(payload, dict) else None
        if state in {"starting", "waiting", "scanned", *_TERMINAL}:
            job.status = state
            error_code = payload.get("error_code") if isinstance(payload, dict) and state == "error" else None
            job.error_code = error_code if isinstance(error_code, str) and error_code in _SAFE_ERROR_MESSAGES else None
            if job.status in _TERMINAL:
                self._cleanup_qr(job)

    def _cleanup_qr(self, job: _LoginJob) -> None:
        try:
            job.qr_path.unlink(missing_ok=True)
        except OSError:
            self._lifecycle_failed = True

    def _write_safe_status(self, job: _LoginJob, status: str) -> None:
        # No helper stdout/stderr or arbitrary exception text is ever saved.
        try:
            job.directory.mkdir(parents=True, exist_ok=True)
            value = json.dumps({"status": status}, separators=(",", ":")).encode("utf-8")
            temp = job.status_path.with_suffix(".tmp")
            temp.write_bytes(value)
            os.replace(temp, job.status_path)
        except OSError:
            pass

    async def _stop_current_unlocked(self, final_status: str) -> None:
        job = self._current
        if job is None:
            return
        process = job.process
        if process is not None and getattr(process, "returncode", None) is None:
            await self._terminate_process(process)
        monitor = job.monitor
        job.monitor = None
        if monitor is not None and monitor is not asyncio.current_task():
            if not monitor.done():
                monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
            except Exception:
                self._lifecycle_failed = True
        job.status = final_status
        self._write_safe_status(job, final_status)
        self._cleanup_qr(job)
        if process is not None and getattr(process, "returncode", None) is None:
            # Do not start a replacement generation until the old child is proven gone.
            await process.wait()
        self._remove_active_marker(job)
        self._current = None

    async def _terminate_process(self, process: Any) -> None:
        if getattr(process, "returncode", None) is not None:
            return
        try:
            if os.name == "nt":
                killer = await asyncio.create_subprocess_exec(
                    "taskkill.exe", "/PID", str(process.pid), "/T", "/F",
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                await asyncio.wait_for(killer.wait(), timeout=self._stop_timeout)
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except (OSError, asyncio.TimeoutError):
            try:
                process.kill()
            except Exception:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=self._stop_timeout)
        except asyncio.TimeoutError:
            try:
                if os.name == "nt":
                    killer = await asyncio.create_subprocess_exec(
                        "taskkill.exe", "/PID", str(process.pid), "/T", "/F",
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    await asyncio.wait_for(killer.wait(), timeout=self._stop_timeout)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except (OSError, asyncio.TimeoutError):
                process.kill()
            await process.wait()

    def _snapshot(self, job: _LoginJob) -> dict[str, Any]:
        self._refresh_status(job)
        qr_ready = job.status in {"waiting", "scanned"} and self._complete_png_path(job.qr_path)
        terminal = job.status in _TERMINAL
        return {
            "job_id": job.job_id,
            "status": job.status,
            "qr_ready": qr_ready,
            "terminal": terminal,
            "retryable": terminal and job.status != "success",
            "lifecycle_failed": self._lifecycle_failed,
            "message": (
                _SAFE_ERROR_MESSAGES.get(job.error_code, _SAFE_MESSAGES["error"])
                if job.status == "error"
                else _SAFE_MESSAGES.get(job.status, _SAFE_MESSAGES["error"])
            ),
        }

    @staticmethod
    def _complete_png_path(path: Path) -> bool:
        try:
            payload = path.read_bytes()
        except OSError:
            return False
        return len(payload) <= MAX_QR_BYTES and NeteaseLoginManager._complete_png(payload)

    @staticmethod
    def _complete_png(payload: bytes) -> bool:
        if not payload.startswith(PNG_SIGNATURE):
            return False
        offset = len(PNG_SIGNATURE)
        while offset + 12 <= len(payload):
            length = int.from_bytes(payload[offset : offset + 4], "big")
            end = offset + 12 + length
            if end > len(payload):
                return False
            chunk_type = payload[offset + 4 : offset + 8]
            offset = end
            if chunk_type == b"IEND":
                return offset == len(payload)
        return False


__all__ = ["NeteaseLoginManager"]
