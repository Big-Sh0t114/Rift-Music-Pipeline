"""Explicit, plugin-local preparation for the optional music runtime.

Importing this module does not create directories, start processes, or access
the network. Dependency and model preparation only occurs in ``prepare``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from .settings import PipelineError, load_settings


PLUGIN_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PLUGIN_DIR / "runtime-manifest.json"
_SAFE_STATUS = {"missing", "preparing", "ready", "partial", "failed", "cancelled"}
_COMPONENTS = ("audio", "rift", "login", "netease")
_STALE_LOCK_SECONDS = 60
_STALE_WORKER_SECONDS = 60
_COMMAND_TIMEOUT = 60 * 60
_SERVICE_START_TIMEOUT = 20.0
_SERVICE_PORT = 13306
_SERVICE_HOST = "127.0.0.1"
_SERVICE_IDENTITY_PATH = "/"
_NO_PROXY_KEYS = {
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "NODE_OPTIONS", "NODE_PATH",
}
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _read_manifest() -> dict[str, Any]:
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise PipelineError("runtime_manifest_missing", "The pinned music runtime manifest is unavailable.") from None
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise PipelineError("runtime_manifest_invalid", "The pinned music runtime manifest is invalid.")
    return value


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_lock_digest(path: Path) -> bytes:
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).digest()


def _sha256(path: Path) -> str:
    try:
        stat_value = path.stat()
    except OSError:
        return ""
    cache_key = (str(path.resolve()), stat_value.st_size, stat_value.st_mtime_ns)
    cached = _HASH_CACHE.get(cache_key)
    if cached:
        return cached
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return ""
    value = digest.hexdigest()
    _HASH_CACHE[cache_key] = value
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        with temporary.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_pid_alive(pid: int) -> bool:
    """Query Windows process state without sending a termination signal."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        if ctypes.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: no process has this PID
            return False
        return True  # Access denied or unknown query failure: fail closed.
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _under_runtime(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise PipelineError("unsafe_runtime_path", "The installer target is outside the managed runtime folder.") from None
    return resolved


def _worker_markers(runtime_dir: Path) -> list[tuple[Path, int | None, bool]]:
    marker_dir = runtime_dir / "active-workers"
    try:
        files = tuple(marker_dir.glob("*.json"))
    except OSError:
        return [(marker_dir, None, True)]
    result: list[tuple[Path, int | None, bool]] = []
    now = time.time()
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            pid = raw.get("pid") if isinstance(raw, dict) else None
            if type(pid) is not int or pid <= 0:
                result.append((path, None, True))
                continue
            alive = _pid_alive(pid)
            age = now - path.stat().st_mtime
            if not alive and age >= _STALE_WORKER_SECONDS:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    result.append((path, pid, True))
            else:
                result.append((path, pid, alive or age < _STALE_WORKER_SECONDS))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            result.append((path, None, True))
    return result


class _InstallationLock:
    """Cross-process exclusive lock; only provably dead, old owners are reaped."""

    def __init__(self, path: Path, timeout: float = 30.0):
        self.path = path
        self.timeout = timeout
        self.token = secrets.token_hex(16)
        self.acquired = False

    def _try_once(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"pid": os.getpid(), "created_at": time.time(), "token": self.token}).encode("utf-8")
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                pid = existing.get("pid")
                created_at = float(existing.get("created_at", 0))
                stale = type(pid) is int and pid > 0 and not _pid_alive(pid) and time.time() - created_at >= _STALE_LOCK_SECONDS
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                stale = False
            if stale:
                try:
                    self.path.unlink()
                except OSError:
                    return False
                return self._try_once()
            return False
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        self.acquired = True
        return True

    async def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout
        while not self._try_once():
            if time.monotonic() >= deadline:
                raise PipelineError("installer_busy", "Another music runtime preparation is still running.")
            await asyncio.sleep(0.2)

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if existing.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        self.acquired = False


class RuntimeInstaller:
    """Prepare isolated plugin environments on explicit operator request."""

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = settings if settings is not None else load_settings()
        self.runtime_dir = Path(self.settings["runtime_dir"]).expanduser().resolve()
        self.manifest = _read_manifest()
        self._children: set[asyncio.subprocess.Process] = set()
        self._prepare_task: asyncio.Task[Any] | None = None
        self._closed = False
        self._prepare_started_at: float | None = None
        self._prepare_device = str(self.settings.get("runtime", {}).get("device", "auto"))
        self._prepare_components: dict[str, Any] = {}

    @property
    def fingerprint(self) -> str:
        return _canonical_digest(self.manifest)

    def status(self) -> dict[str, Any]:
        """Return the last safe persisted progress snapshot without probing models."""
        path = self.runtime_dir / "installer-status.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {
                "state": "missing",
                "stage": "idle",
                "progress": 0,
                "device": str(self.settings.get("runtime", {}).get("device", "auto")),
                "components": {},
                "fingerprint": self.fingerprint,
                "fingerprints": self._component_fingerprints(str(self.settings.get("runtime", {}).get("device", "auto"))),
                "detail": None,
                "elapsed_seconds": 0,
                "updated_at": None,
                "download": None,
                "owner_alive": False,
                "error_code": None,
                "error": None,
            }
        if not isinstance(raw, dict) or raw.get("state") not in _SAFE_STATUS:
            return {
                "state": "missing",
                "stage": "idle",
                "progress": 0,
                "components": {},
                "fingerprint": self.fingerprint,
                "fingerprints": self._component_fingerprints(str(self.settings.get("runtime", {}).get("device", "auto"))),
                "detail": None,
                "elapsed_seconds": 0,
                "updated_at": None,
                "download": None,
                "owner_alive": None,
                "error_code": None,
                "error": None,
            }
        try:
            progress = int(raw.get("progress", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            progress = 0
        state = raw.get("state")
        stage = str(raw.get("stage", "idle"))[:64]
        detail = str(raw.get("detail") or "")[:300] or None
        error_code = str(raw.get("error_code"))[:64] if raw.get("error_code") else None
        error = str(raw.get("error"))[:300] if raw.get("error") else None
        owner_alive: bool | None = None
        download = self._download_status(raw.get("download"))
        if state == "preparing":
            owner_alive = self._preparation_owner_alive()
            if owner_alive is False:
                state = "failed"
                stage = "interrupted"
                detail = "准备进程已退出，依赖准备没有完成。"
                progress = 0
                download = None
                error_code = "prepare_interrupted"
                error = "The preparation process ended before it could finish."
        try:
            updated_at = int(raw.get("updated_at")) if raw.get("updated_at") is not None else None
            started_at = int(raw.get("started_at", updated_at or time.time()))
        except (TypeError, ValueError, OverflowError):
            updated_at = None
            started_at = int(time.time())
        elapsed_seconds = max(0, int(time.time()) - started_at) if state == "preparing" else 0
        fingerprints = raw.get("fingerprints") if isinstance(raw.get("fingerprints"), dict) else {}
        fingerprints = dict(fingerprints)
        fingerprints["audio"] = self._component_fingerprints(
            str(self.settings.get("runtime", {}).get("device", "auto"))
        )["audio"]
        return {
            "state": state,
            "stage": stage,
            "progress": max(0, min(100, progress)),
            "device": str(raw.get("device", "auto"))[:16],
            "components": raw.get("components") if isinstance(raw.get("components"), dict) else {},
            "fingerprint": self.fingerprint,
            "fingerprints": fingerprints,
            "detail": detail,
            "elapsed_seconds": elapsed_seconds,
            "updated_at": updated_at,
            "download": download,
            "owner_alive": owner_alive,
            "error_code": error_code,
            "error": error,
        }

    def _preparation_owner_alive(self) -> bool | None:
        """Read the preparation lock without removing or modifying it."""
        try:
            owner = json.loads((self.runtime_dir / "installer.lock").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        pid = owner.get("pid") if isinstance(owner, dict) else None
        if type(pid) is not int or pid <= 0:
            return None
        return _pid_alive(pid)

    def _download_status(self, value: Any) -> dict[str, Any] | None:
        """Read live byte counts from a managed partial file without hashing it."""
        if not isinstance(value, dict):
            return None
        relative = PurePosixPath(str(value.get("target", "")))
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            return None
        try:
            target = _under_runtime(self.runtime_dir.joinpath(*relative.parts), self.runtime_dir)
        except (OSError, PipelineError):
            return None
        try:
            total = int(value["size_bytes"]) if value.get("size_bytes") is not None else None
        except (TypeError, ValueError, OverflowError):
            return None
        if total is not None and total < 0:
            return None
        partial = target.with_name(target.name + ".partial")
        source = partial if partial.is_file() else target
        try:
            with source.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                received = stream.tell()
        except OSError:
            received = 0
        if total is not None:
            received = min(received, total)
        model = str(value.get("model", ""))[:128]
        asset = str(value.get("asset", ""))[:128]
        if any(separator in model or separator in asset for separator in ("/", "\\")):
            return None
        result: dict[str, Any] = {
            "component": str(value.get("component", ""))[:16],
            "asset": asset,
            "bytes_received": max(0, received),
            "bytes_total": total,
        }
        if model:
            result["model"] = model
        return result

    def check(self) -> dict[str, Any]:
        """Check real local installs and prepared asset hashes without importing ML packages."""
        components = {
            "audio": self._check_audio(),
            "rift": self._check_rift(),
            "login": self._check_login(),
            "netease": self._check_netease(),
        }
        fingerprints = self._component_fingerprints(str(self.settings.get("runtime", {}).get("device", "auto")))
        ready = all(bool(item.get("ready")) for item in components.values())
        return {
            "ready": ready,
            "components": components,
            "fingerprint": self.fingerprint,
            "fingerprints": fingerprints,
        }

    async def prepare(
        self,
        device: str = "auto",
        *,
        components: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Prepare selected components; no install runs before this explicit call."""
        if device not in {"auto", "cpu", "cuda"}:
            raise PipelineError("invalid_device", "Choose auto, cpu, or cuda for runtime preparation.")
        selected = self._normalize_components(components)
        if self._closed:
            self._closed = False
        self._prepare_task = asyncio.current_task()
        lock = _InstallationLock(self.runtime_dir / "installer.lock")
        await lock.acquire()
        self._prepare_started_at = time.time()
        try:
            resolved_device = await self._resolve_device(device) if "audio" in selected else device
            current = self.check()
            self._prepare_device = resolved_device
            self._prepare_components = current["components"]
            self._write_status("preparing", "validation", 0, resolved_device, current["components"], detail="正在检查受管环境和已缓存组件。")
            if any(item[2] for item in _worker_markers(self.runtime_dir)):
                # Existing active workers may finish normally; activation waits for a restart.
                if any(component in selected and not current["components"][component]["ready"] for component in {"audio", "rift"}):
                    raise PipelineError("runtime_in_use", "Stop active music jobs before replacing their runtime.")

            steps = [
                ("netease", "netease", 0),
                ("login", "login", 0),
                ("audio", "audio", 0),
                ("rift", "rift", 0),
            ]
            for component, stage, progress in steps:
                if component not in selected:
                    continue
                snapshot = self.check()["components"][component]
                if snapshot.get("ready") and not (component == "audio" and resolved_device != snapshot.get("device")):
                    continue
                self._prepare_components = self.check()["components"]
                self._write_status("preparing", stage, progress, resolved_device, self._prepare_components, detail=f"正在准备 {component} 组件。")
                await self._prepare_component(component, resolved_device)

            report = self.check()
            state = "ready" if report["ready"] else "partial"
            self._write_status(state, "complete" if state == "ready" else "waiting_for_components", 100, resolved_device, report["components"], fingerprints=report["fingerprints"])
            return {"state": state, **report}
        except asyncio.CancelledError:
            await self._terminate_children()
            self._write_status("cancelled", "cancelled", 0, device, {}, error_code="prepare_cancelled", error="Runtime preparation was cancelled.")
            raise
        except PipelineError as exc:
            previous = self.status()
            failure_detail = previous.get("detail") if previous.get("state") == "preparing" else None
            self._write_status("failed", "failed", 0, device, {}, detail=failure_detail, error_code=exc.code, error=exc.message)
            raise
        except Exception:
            previous = self.status()
            failure_detail = previous.get("detail") if previous.get("state") == "preparing" else None
            self._write_status("failed", "failed", 0, device, {}, detail=failure_detail, error_code="prepare_failed", error="Music runtime preparation failed. Review the selected component and try again.")
            raise PipelineError("prepare_failed", "Music runtime preparation failed. Review the selected component and try again.") from None
        finally:
            lock.release()
            self._prepare_task = None
            self._prepare_started_at = None

    async def close(self) -> None:
        """Cancel and reap only subprocesses created by this installer instance."""
        task = self._prepare_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._terminate_children()
        self._closed = True

    def _normalize_components(self, requested: Sequence[str] | None) -> tuple[str, ...]:
        if requested is None:
            return _COMPONENTS
        values: list[str] = []
        for item in requested:
            if item == "all":
                values.extend(_COMPONENTS)
            elif item == "control":
                values.extend(("netease", "login"))
            elif item in _COMPONENTS:
                values.append(item)
            else:
                raise PipelineError("invalid_component", "Choose all, control, audio, rift, login, or netease.")
        if "login" in values and "netease" not in values:
            values.insert(values.index("login"), "netease")
        return tuple(dict.fromkeys(values))

    def _component_fingerprints(self, device: str) -> dict[str, str]:
        audio_device = self._audio_fingerprint_device(device)
        lock_hashes: dict[str, str] = {}
        for component, rel in (
            (
                "audio",
                self.manifest.get("audio", {}).get("locks", {}).get(audio_device)
                if audio_device in {"cpu", "cuda"}
                else None,
            ),
            ("rift", self.manifest.get("rift", {}).get("lock")),
        ):
            if rel:
                lock = PLUGIN_DIR / str(rel)
                try:
                    lock_hashes[component] = hashlib.sha256(lock.read_bytes()).hexdigest()
                except OSError:
                    lock_hashes[component] = "missing"
        return {
            component: _canonical_digest(
                {
                    "manifest": self.fingerprint,
                    "component": component,
                    "device": audio_device if component == "audio" else device,
                    "lock": lock_hashes.get(component, ""),
                }
            )
            for component in ("audio", "rift", "login", "netease")
        }

    def _audio_fingerprint_device(self, requested: str) -> str:
        if requested in {"cpu", "cuda"}:
            return requested
        configured = self.settings.get("runtime", {}).get("device", "auto")
        if configured in {"cpu", "cuda"}:
            return str(configured)
        return self._installed_audio_device()

    def _installed_audio_device(self) -> str:
        """Resolve auto from durable audio component metadata, never the last global status."""
        record = self._read_install_record().get("audio")
        record_device = record.get("device") if isinstance(record, dict) else None
        if record_device not in {"cpu", "cuda"}:
            record_device = None
        manifest_device = None
        manifest_path = self.runtime_dir / "models" / "uvr" / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidate = manifest.get("device") if isinstance(manifest, dict) else None
            if candidate in {"cpu", "cuda"}:
                manifest_device = candidate
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        if record_device and manifest_device and record_device != manifest_device:
            return "auto"
        return str(record_device or manifest_device or "auto")

    def _write_status(
        self,
        state: str,
        stage: str,
        progress: int,
        device: str,
        components: Mapping[str, Any],
        *,
        fingerprints: Mapping[str, str] | None = None,
        detail: str | None = None,
        download: Mapping[str, Any] | None = None,
        error_code: str | None = None,
        error: str | None = None,
    ) -> None:
        value = {
            "state": state if state in _SAFE_STATUS else "failed",
            "stage": str(stage)[:64],
            "progress": max(0, min(100, int(progress))),
            "device": str(device)[:16],
            "components": dict(components),
            "fingerprint": self.fingerprint,
            "fingerprints": dict(fingerprints or self._component_fingerprints(device)),
            "detail": str(detail)[:300] if detail else None,
            "download": dict(download) if download else None,
            "started_at": int(self._prepare_started_at or time.time()),
            "error_code": str(error_code)[:64] if error_code else None,
            "error": str(error)[:300] if error else None,
            "updated_at": int(time.time()),
        }
        _atomic_json(self.runtime_dir / "installer-status.json", value)

    def _write_prepare_progress(
        self,
        stage: str,
        detail: str,
        *,
        download: Mapping[str, Any] | None = None,
    ) -> None:
        self._write_status(
            "preparing",
            stage,
            0,
            self._prepare_device,
            self._prepare_components,
            detail=detail,
            download=download,
        )

    async def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if os.name == "nt" and shutil.which("nvidia-smi"):
            result = await self._run_capture(["nvidia-smi", "-L"], timeout=4, allow_failure=True)
            if result.returncode == 0 and result.stdout.strip():
                return "cuda"
        return "cpu"

    async def _prepare_component(self, component: str, device: str) -> None:
        if component == "netease":
            await self._prepare_netease()
        elif component == "login":
            await self._prepare_python_environment("login", "runtime_specs/login/requirements.lock.txt", ("qrcode",))
        elif component == "audio":
            await self._prepare_audio(device)
        elif component == "rift":
            await self._prepare_rift()

    def _uv_exe(self) -> str:
        uv = shutil.which("uv")
        if not uv:
            raise PipelineError("uv_missing", "Install the pinned uv tool before preparing the music runtime.")
        return uv

    async def _uv_python(self) -> Path:
        uv = self._uv_exe()
        python_manifest = self.manifest["python"]
        env = self._child_env()
        install_dir = self.runtime_dir / "tools" / "python"
        install_dir.mkdir(parents=True, exist_ok=True)
        env["UV_PYTHON_INSTALL_DIR"] = str(install_dir)
        env["UV_CACHE_DIR"] = str(self.runtime_dir / "cache" / "uv")
        env["UV_NO_PROGRESS"] = "1"
        result = await self._run_capture([uv, "--version"], env=env, timeout=15)
        if not result.stdout.strip().startswith(f"uv {python_manifest['uv_version']} "):
            raise PipelineError("uv_version_mismatch", f"Runtime preparation requires uv {python_manifest['uv_version']}.")
        await self._run(
            [
                uv,
                "python",
                "install",
                str(python_manifest["version"]),
                "--install-dir",
                str(install_dir),
                "--no-registry",
                "--no-bin",
            ],
            env=env,
            timeout=600,
        )
        found = await self._run_capture([uv, "python", "find", str(python_manifest["version"]), "--managed-python"], env=env, timeout=30)
        candidate = Path(found.stdout.strip().splitlines()[-1]).resolve()
        if not candidate.is_file():
            raise PipelineError("python_missing", "The pinned managed Python interpreter was not installed.")
        version = await self._run_capture([str(candidate), "--version"], timeout=15)
        if version.stdout.strip() != f"Python {python_manifest['version']}":
            raise PipelineError("python_version_mismatch", "The managed Python version does not match the runtime pin.")
        return candidate

    async def _prepare_python_environment(
        self,
        name: str,
        lock_relative: str,
        required_distributions: Sequence[str],
        *,
        torch_backend: str | None = None,
    ) -> Path:
        lockfile = PLUGIN_DIR / lock_relative
        if not lockfile.is_file():
            raise PipelineError("runtime_lock_missing", f"The pinned {name} dependency lock is missing.")
        if self._environment_ready(name, lock_relative, required_distributions):
            return self._python_executable(self.runtime_dir / "envs" / name)
        if os.name != "nt":
            raise PipelineError("platform_unsupported", "The pinned runtime environment is currently prepared for Windows x64.")
        uv = self._uv_exe()
        self._write_prepare_progress(
            "managed_python",
            f"正在检查或安装受管 Python {self.manifest['python']['version']}。",
        )
        base_python = await self._uv_python()
        envs = self.runtime_dir / "envs"
        envs.mkdir(parents=True, exist_ok=True)
        stage = _under_runtime(envs / f".{name}.stage-{uuid.uuid4().hex}", self.runtime_dir)
        try:
            self._write_prepare_progress("creating_environment", f"正在创建 {name} 的隔离 Python 环境。")
            await self._run([uv, "venv", "--python", str(base_python), str(stage)], env=self._child_env(), timeout=120)
            stage_python = self._python_executable(stage)
            sync_command = [uv, "pip", "sync", "--python", str(stage_python), "--require-hashes"]
            if torch_backend is not None:
                sync_command.extend(("--torch-backend", torch_backend))
            sync_command.append(str(lockfile))
            dependency_detail = f"正在安装 {name} 的锁定 Python 依赖。"
            if torch_backend == "cu128":
                dependency_detail = "正在安装音频依赖、PyTorch 与 CUDA 12.8 GPU 包。"
            elif torch_backend == "cpu":
                dependency_detail = "正在安装音频依赖与 CPU 版 PyTorch。"
            self._write_prepare_progress("installing_dependencies", dependency_detail)
            await self._run(sync_command, env=self._child_env(), timeout=_COMMAND_TIMEOUT)
            self._write_prepare_progress("validating_environment", f"正在校验 {name} 环境中的必需 Python 包。")
            if not self._validate_environment(stage_python, required_distributions):
                raise PipelineError("runtime_environment_invalid", f"The staged {name} environment did not pass package validation.")
            self._assert_no_active_workers()
            target = envs / name
            self._activate_directory(
                stage,
                target,
                validator=lambda path: self._validate_environment(self._python_executable(path), required_distributions),
            )
            active_python = self._python_executable(target)
            self._write_install_record(name, {"ready": True, "lock_sha256": hashlib.sha256(lockfile.read_bytes()).hexdigest(), "python_version": self.manifest["python"]["version"]})
            return active_python
        finally:
            self._remove_owned_directory(stage)

    def _environment_ready(self, name: str, lock_relative: str, required: Sequence[str]) -> bool:
        env_dir = self.runtime_dir / "envs" / name
        python = self._python_executable(env_dir)
        if not python.is_file():
            return False
        lockfile = PLUGIN_DIR / lock_relative
        try:
            lock_hash = hashlib.sha256(lockfile.read_bytes()).hexdigest()
        except OSError:
            return False
        record = self._read_install_record().get(name)
        if not isinstance(record, dict) or record.get("lock_sha256") != lock_hash:
            return False
        return self._validate_environment(python, required, quiet=True)

    def _validate_environment(self, python: Path, required: Sequence[str], *, quiet: bool = False) -> bool:
        if not python.is_file():
            return False
        code = (
            "import importlib.metadata as m, json, sys; "
            f"names={json.dumps(list(required))}; "
            "result={}; "
            "[result.__setitem__(n,m.version(n)) for n in names]; "
            "print(json.dumps(result))"
        )
        try:
            result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=12, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    def _python_executable(self, env_dir: Path) -> Path:
        return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def _read_install_record(self) -> dict[str, Any]:
        try:
            raw = json.loads((self.runtime_dir / "installation.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write_install_record(self, component: str, value: Mapping[str, Any]) -> None:
        record = self._read_install_record()
        record[component] = dict(value)
        record["schema_version"] = 1
        record["fingerprint"] = self.fingerprint
        _atomic_json(self.runtime_dir / "installation.json", record)

    def _activate_directory(self, stage: Path, target: Path, *, validator: Callable[[Path], bool] | None = None) -> None:
        stage = _under_runtime(stage, self.runtime_dir)
        target = _under_runtime(target, self.runtime_dir)
        self._assert_no_active_workers()
        backup = target.with_name(f".{target.name}.old-{uuid.uuid4().hex}")
        moved_old = False
        activated = False
        try:
            if target.exists():
                os.replace(target, backup)
                moved_old = True
            os.replace(stage, target)
            activated = True
            if validator is not None and not validator(target):
                raise PipelineError("runtime_validation_failed", "The activated runtime did not pass package validation.")
        except BaseException as exc:
            if activated and target.exists():
                self._remove_owned_directory(target)
            if moved_old and not target.exists() and backup.exists():
                os.replace(backup, target)
            if isinstance(exc, PipelineError):
                raise
            raise PipelineError("runtime_activation_failed", "The prepared environment could not be activated safely.") from None
        if moved_old:
            self._remove_owned_directory(backup)

    def _remove_owned_directory(self, path: Path) -> None:
        target = _under_runtime(path, self.runtime_dir)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    def _assert_no_active_workers(self) -> None:
        active = _worker_markers(self.runtime_dir)
        if active:
            raise PipelineError("runtime_in_use", "Stop active music jobs before replacing their runtime.")

    def _child_env(self, *, no_proxy: bool = False) -> dict[str, str]:
        env = dict(os.environ)
        for key in _NO_PROXY_KEYS:
            env.pop(key, None)
        env["PYTHONUTF8"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["UV_NO_PROGRESS"] = "1"
        env["UV_CACHE_DIR"] = str(self.runtime_dir / "cache" / "uv")
        if no_proxy:
            env["NO_PROXY"] = "*"
        return env

    async def _run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = _COMMAND_TIMEOUT,
        allow_failure: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        process = await self._spawn(command, cwd=cwd, env=env, capture=False)
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._stop_process(process)
            raise PipelineError("runtime_command_timeout", "A pinned dependency preparation step exceeded its time limit.") from None
        except asyncio.CancelledError:
            await self._stop_process(process)
            raise
        if process.returncode and not allow_failure:
            raise PipelineError("runtime_command_failed", "A pinned dependency preparation step failed.")
        return subprocess.CompletedProcess(list(command), int(process.returncode or 0), "", "")

    async def _run_capture(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30,
        allow_failure: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        process = await self._spawn(command, cwd=cwd, env=env, capture=True)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._stop_process(process)
            raise PipelineError("runtime_command_timeout", "A pinned dependency check exceeded its time limit.") from None
        except asyncio.CancelledError:
            await self._stop_process(process)
            raise
        out = stdout.decode("utf-8", "replace") if stdout else ""
        err = stderr.decode("utf-8", "replace") if stderr else ""
        if process.returncode and not allow_failure:
            # Never persist or return subprocess output, which can contain local paths.
            raise PipelineError("runtime_command_failed", "A pinned dependency check failed.")
        return subprocess.CompletedProcess(list(command), int(process.returncode or 0), out, err)

    async def _spawn(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None,
        env: Mapping[str, str] | None,
        capture: bool,
    ) -> asyncio.subprocess.Process:
        if self._closed:
            raise PipelineError("installer_closed", "The runtime installer has been closed.")
        kwargs: dict[str, Any] = {
            "cwd": str(cwd) if cwd else None,
            "stdin": asyncio.subprocess.DEVNULL,
            "stdout": asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
            "stderr": asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
            "env": dict(env or self._child_env()),
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        spawn_task = asyncio.create_task(asyncio.create_subprocess_exec(*map(str, command), **kwargs))
        try:
            process = await asyncio.shield(spawn_task)
        except asyncio.CancelledError as cancelled:
            try:
                process = await spawn_task
            except BaseException:
                raise cancelled from None
            self._children.add(process)
            try:
                await self._stop_process(process)
            except BaseException:
                pass
            raise cancelled
        except OSError:
            raise PipelineError("runtime_command_unavailable", "A required pinned runtime tool could not be started.") from None
        self._children.add(process)
        return process

    async def _stop_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            self._children.discard(process)
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
                await asyncio.wait_for(killer.wait(), timeout=10)
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except (OSError, asyncio.TimeoutError):
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
        self._children.discard(process)

    async def _terminate_children(self) -> None:
        children = tuple(self._children)
        if children:
            await asyncio.gather(*(self._stop_process(child) for child in children), return_exceptions=True)

    def _check_audio(self) -> dict[str, Any]:
        configured_device = str(self.settings.get("runtime", {}).get("device", "auto"))
        selected_device = self._audio_fingerprint_device(configured_device)
        lock_relative = (
            self.manifest.get("audio", {}).get("locks", {}).get(selected_device)
            if selected_device in {"cpu", "cuda"}
            else None
        )
        required = ("audio-separator", "imageio-ffmpeg")
        env_ready = bool(lock_relative and self._environment_ready("audio", lock_relative, required))
        ffmpeg = self.runtime_dir / "models" / "tools" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        ffmpeg_ready = ffmpeg.is_file()
        models_ready, model_errors = self._check_audio_assets(selected_device)
        return {"ready": env_ready and ffmpeg_ready and models_ready, "environment_ready": env_ready, "ffmpeg_ready": ffmpeg_ready, "assets_ready": models_ready, "device": selected_device, "missing": model_errors}

    def _check_audio_assets(self, expected_device: str | None = None) -> tuple[bool, list[str]]:
        model_dir = self.runtime_dir / "models" / "uvr"
        manifest_path = model_dir / "manifest.json"
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False, ["model_manifest"]
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            return False, ["model_manifest"]
        manifest_device = str(value.get("device", "auto"))
        if expected_device in {"cpu", "cuda"} and manifest_device != expected_device:
            return False, ["audio_device"]
        if expected_device == "auto" and manifest_device in {"cpu", "cuda"}:
            return False, ["audio_device"]
        expected_fingerprint = self._component_fingerprints(manifest_device).get("audio")
        if value.get("installer_fingerprint") != expected_fingerprint:
            return False, ["installer_fingerprint"]
        entries = value.get("models")
        if not isinstance(entries, dict):
            return False, ["model_manifest"]
        missing: list[str] = []
        for filename in self.settings.get("separation", {}).values():
            entry = entries.get(str(filename))
            if not isinstance(entry, dict) or not isinstance(entry.get("files"), list):
                missing.append(str(filename))
                continue
            for item in entry["files"]:
                if not isinstance(item, dict):
                    missing.append(str(filename))
                    continue
                name = str(item.get("name", ""))
                path = _under_runtime(model_dir / name, model_dir)
                try:
                    stat_value = path.stat()
                except OSError:
                    missing.append(str(filename))
                    continue
                if (
                    not path.is_file()
                    or stat_value.st_size != item.get("size_bytes")
                    or stat_value.st_mtime_ns != item.get("mtime_ns")
                    or not item.get("sha256")
                    or _sha256(path) != item.get("sha256")
                ):
                    missing.append(str(filename))
                    break
        return not missing, list(dict.fromkeys(missing))

    def _check_login(self) -> dict[str, Any]:
        lock_relative = "runtime_specs/login/requirements.lock.txt"
        ready = self._environment_ready("login", lock_relative, ("qrcode", "Pillow"))
        return {"ready": ready, "environment_ready": ready}

    def _check_netease(self) -> dict[str, Any]:
        node_root = self.runtime_dir / "tools" / f"node-v{self.manifest['node']['version']}-win-x64"
        node = node_root / "node.exe"
        api_root = self.runtime_dir / "service" / "netease-api"
        package_path = api_root / "node_modules" / str(self.manifest["netease_api"]["name"]) / "package.json"
        try:
            node_manifest = json.loads((api_root / "package.json").read_text(encoding="utf-8"))
            api_manifest = json.loads(package_path.read_text(encoding="utf-8"))
            version_ok = api_manifest.get("version") == self.manifest["netease_api"]["version"]
            lockfile = PLUGIN_DIR / str(self.manifest["netease_api"]["package_lock"])
            lock_ok = lockfile.is_file() and node_manifest.get("dependencies", {}).get(str(self.manifest["netease_api"]["name"])) == self.manifest["netease_api"]["version"]
            record = self._read_install_record().get("netease")
            recorded = isinstance(record, dict) and record.get("lock_sha256") == hashlib.sha256(lockfile.read_bytes()).hexdigest() if lock_ok else False
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            version_ok = lock_ok = recorded = False
        ready = os.name == "nt" and node.is_file() and version_ok and lock_ok and recorded
        return {"ready": bool(ready), "node_ready": node.is_file(), "api_ready": bool(version_ok and recorded)}

    def _check_rift(self) -> dict[str, Any]:
        runtime = self.settings.get("runtime", {})
        source_value = runtime.get("rift_source") or self.runtime_dir / "source" / "RIFT-SVC"
        python_value = runtime.get("rift_python") or self._python_executable(self.runtime_dir / "envs" / "rift")
        source = Path(source_value).expanduser()
        python = Path(python_value).expanduser()
        source_ready = self._rift_source_ready(source)
        python_ready = self._validate_rift_environment(python)
        python_override = bool(runtime.get("rift_python"))
        if python_override:
            # Overrides are read-only inputs; their environment fingerprint is validated, never repaired here.
            env_ready = python_ready
        else:
            env_ready = python_ready and self._rift_environment_record_ready()
        return {"ready": bool(source_ready and env_ready), "source_ready": bool(source_ready), "environment_ready": bool(env_ready), "external_override": bool(runtime.get("rift_source") or python_override)}

    def _rift_source_ready(self, source: Path) -> bool:
        pinned = str(self.manifest["rift"]["commit"])
        try:
            result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=8, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
            source_ready = result.returncode == 0 and result.stdout.strip().casefold() == pinned.casefold() and (source / "infer.py").is_file()
        except (OSError, subprocess.SubprocessError):
            source_ready = False
        return bool(source_ready and self._rift_assets_ready(source))

    def _rift_assets_ready(self, source: Path) -> bool:
        for asset in self.manifest.get("rift", {}).get("assets", []):
            path = source / str(asset["path"])
            if not path.is_file():
                return False
            try:
                stat_value = path.stat()
            except OSError:
                return False
            if stat_value.st_size != int(asset["size_bytes"]) or _sha256(path) != str(asset["sha256"]):
                return False
        return True

    def _rift_environment_record_ready(self) -> bool:
        installation = self._read_install_record()
        record = installation.get("rift")
        lock = PLUGIN_DIR / str(self.manifest["rift"]["lock"])
        try:
            lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
        except OSError:
            return False
        return (
            isinstance(record, dict)
            and record.get("lock_sha256") == lock_hash
            and record.get("python_version") == self.manifest["python"]["version"]
            and installation.get("fingerprint") == self.fingerprint
        )

    def _has_distribution(self, python: Path, distribution: str) -> bool:
        code = "import importlib.metadata as m; m.version(" + repr(distribution) + ")"
        try:
            result = subprocess.run([str(python), "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    async def _prepare_netease(self) -> None:
        if self._check_netease()["ready"]:
            return
        if os.name != "nt" or self.manifest["node"]["archive"] != "node-v24.11.1-win-x64.zip":
            raise PipelineError("platform_unsupported", "The managed Netease service package is currently prepared for Windows x64.")
        service_state = self.runtime_dir / "service" / "netease-service.json"
        try:
            current = json.loads(service_state.read_text(encoding="utf-8"))
            service_pid = current.get("pid") if isinstance(current, dict) else None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            service_pid = None
        if type(service_pid) is int and _pid_alive(service_pid):
            raise PipelineError("netease_service_in_use", "Stop the managed Netease service before repairing its files.")

        node = self.manifest["node"]
        tools_dir = self.runtime_dir / "tools"
        target = tools_dir / f"node-v{node['version']}-win-x64"
        if not (target / "node.exe").is_file():
            stage = _under_runtime(tools_dir / f".node-stage-{uuid.uuid4().hex}", self.runtime_dir)
            archive = self.runtime_dir / "cache" / str(node["archive"])
            try:
                await self._fetch_asset(str(node["url"]), archive, str(node["sha256"]), None)
                await asyncio.to_thread(self._extract_node_archive, archive, stage)
                self._assert_no_active_workers()
                self._activate_directory(stage, target)
            finally:
                self._remove_owned_directory(stage)

        specs = PLUGIN_DIR / "runtime_specs" / "netease"
        package_stage = _under_runtime(self.runtime_dir / "service" / f".netease-stage-{uuid.uuid4().hex}", self.runtime_dir)
        package_target = self.runtime_dir / "service" / "netease-api"
        package_stage.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(specs / "package.json", package_stage / "package.json")
            shutil.copy2(specs / "package-lock.json", package_stage / "package-lock.json")
            npm_cli = target / "node_modules" / "npm" / "bin" / "npm-cli.js"
            install_env = self._child_env()
            install_env["npm_config_cache"] = str(self.runtime_dir / "cache" / "npm")
            install_env["npm_config_registry"] = str(self.manifest["netease_api"]["registry"])
            install_env["npm_config_audit"] = "false"
            install_env["npm_config_fund"] = "false"
            await self._run([target / "node.exe", npm_cli, "ci", "--omit=dev", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=package_stage, env=install_env, timeout=_COMMAND_TIMEOUT)
            package = json.loads((package_stage / "node_modules" / str(self.manifest["netease_api"]["name"]) / "package.json").read_text(encoding="utf-8"))
            if package.get("version") != self.manifest["netease_api"]["version"]:
                raise PipelineError("netease_package_mismatch", "The Netease API package does not match the pinned version.")
            self._assert_no_active_workers()
            self._activate_directory(package_stage, package_target)
            lock_hash = hashlib.sha256((specs / "package-lock.json").read_bytes()).hexdigest()
            self._write_install_record("netease", {"ready": True, "version": self.manifest["netease_api"]["version"], "lock_sha256": lock_hash})
        finally:
            self._remove_owned_directory(package_stage)

    @staticmethod
    def _extract_node_archive(archive: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(archive) as source:
            root = Path(destination).resolve()
            for item in source.infolist():
                member = PurePosixPath(item.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise PipelineError("node_archive_invalid", "The pinned Node archive contains an invalid path.")
                parts = member.parts[1:] if member.parts and member.parts[0].startswith("node-v") else member.parts
                if not parts:
                    continue
                target = (root / Path(*parts)).resolve()
                try:
                    target.relative_to(root)
                except ValueError:
                    raise PipelineError("node_archive_invalid", "The pinned Node archive contains an invalid path.") from None
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(item, "r") as input_file, target.open("wb") as output_file:
                        shutil.copyfileobj(input_file, output_file, length=1024 * 1024)

    async def _fetch_asset(self, url: str, target: Path, sha256: str, size_bytes: int | None) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or not sha256 or len(sha256) != 64:
            raise PipelineError("asset_pin_invalid", "A pinned runtime asset has an invalid HTTPS URL or digest.")
        target = _under_runtime(target, self.runtime_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and (size_bytes is None or target.stat().st_size == size_bytes) and _sha256(target) == sha256.casefold():
            return
        downloader = PLUGIN_DIR / "scripts" / "fetch_asset.py"
        if not downloader.is_file():
            raise PipelineError("asset_downloader_missing", "The pinned runtime asset downloader is missing.")
        command = [sys.executable, str(downloader), "--url", url, "--target", str(target), "--sha256", sha256]
        if size_bytes is not None:
            command.extend(("--size", str(size_bytes)))
        await self._run(command, env=self._child_env(), timeout=_COMMAND_TIMEOUT)

    async def _prepare_audio(self, device: str) -> None:
        resolved = device if device in {"cpu", "cuda"} else "cpu"
        lock = str(self.manifest["audio"]["locks"][resolved])
        python = await self._prepare_python_environment(
            "audio",
            lock,
            ("audio-separator", "imageio-ffmpeg"),
            torch_backend="cpu" if resolved == "cpu" else "cu128",
        )
        self._assert_no_active_workers()
        model_dir = self.runtime_dir / "models" / "uvr"
        model_dir.mkdir(parents=True, exist_ok=True)
        configured = self.settings.get("separation", {})
        files: dict[str, list[dict[str, Any]]] = {}
        for filename in dict.fromkeys(str(value) for value in configured.values()):
            entry = self.manifest.get("models", {}).get(filename)
            if not isinstance(entry, dict):
                raise PipelineError("model_not_pinned", f"The selected separation model {filename} has no pinned asset manifest.")
            manifest_files: list[dict[str, Any]] = []
            for item in entry.get("files", []):
                relative = str(item.get("name", ""))
                target = _under_runtime(model_dir / relative, model_dir)
                size_bytes = int(item["size_bytes"]) if item.get("size_bytes") is not None else None
                self._write_prepare_progress(
                    "downloading_model",
                    f"正在下载分离模型 {filename}：{Path(relative).name}。",
                    download={
                        "component": "audio",
                        "model": filename,
                        "asset": Path(relative).name,
                        "target": target.relative_to(self.runtime_dir).as_posix(),
                        "size_bytes": size_bytes,
                    },
                )
                await self._fetch_asset(str(item["url"]), target, str(item["sha256"]), size_bytes)
                stat_value = target.stat()
                digest = _sha256(target)
                if digest != str(item["sha256"]).casefold():
                    raise PipelineError("asset_hash_mismatch", "A selected model asset failed its pinned checksum.")
                manifest_files.append({"name": relative, "size_bytes": stat_value.st_size, "mtime_ns": stat_value.st_mtime_ns, "sha256": digest, "verified": True})
            files[filename] = manifest_files
        fingerprint = self._component_fingerprints(resolved)["audio"]
        audio_manifest = {
            "schema_version": 1,
            "installer_fingerprint": fingerprint,
            "audio_separator_version": str(self.manifest["audio"]["audio_separator"]["version"]),
            "device": resolved,
            "models": {name: {"files": values} for name, values in files.items()},
        }
        _atomic_json(model_dir / "manifest.json", audio_manifest)
        self._write_prepare_progress("preparing_ffmpeg", "正在准备音频包中的 FFmpeg 可执行文件。")
        ffmpeg_source = await self._run_capture([str(python), "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"], timeout=20)
        source_path = Path(ffmpeg_source.stdout.strip()).resolve(strict=True)
        ffmpeg_target = self.runtime_dir / "models" / "tools" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        ffmpeg_target.parent.mkdir(parents=True, exist_ok=True)
        if not ffmpeg_target.is_file() or _sha256(source_path) != _sha256(ffmpeg_target):
            temporary = ffmpeg_target.with_name(f".{ffmpeg_target.name}.{uuid.uuid4().hex}.tmp")
            shutil.copy2(source_path, temporary)
            os.replace(temporary, ffmpeg_target)
        self._write_install_record("audio", {"ready": True, "lock_sha256": hashlib.sha256((PLUGIN_DIR / lock).read_bytes()).hexdigest(), "device": resolved, "audio_separator_version": self.manifest["audio"]["audio_separator"]["version"]})

    async def _prepare_rift(self) -> None:
        runtime = self.settings.get("runtime", {})
        external_source = Path(runtime["rift_source"]).expanduser() if runtime.get("rift_source") else None
        external_python = Path(runtime["rift_python"]).expanduser() if runtime.get("rift_python") else None
        managed_source = _under_runtime(self.runtime_dir / "source" / "RIFT-SVC", self.runtime_dir)
        managed_python = self._python_executable(self.runtime_dir / "envs" / "rift")

        if external_source is not None and not self._rift_source_ready(external_source):
            raise PipelineError("rift_override_invalid", "The configured RIFT source is not the pinned revision with its required model files.")
        if external_python is not None and not self._validate_rift_environment(external_python):
            raise PipelineError("rift_override_invalid", "The configured RIFT Python environment is missing required pinned runtime packages.")
        if external_source is not None and external_python is not None:
            return
        if os.name != "nt":
            raise PipelineError("platform_unsupported", "The pinned RIFT environment is currently prepared for Windows x64.")

        source_stage: Path | None = None
        python_stage: Path | None = None
        source_project = external_source or managed_source
        try:
            if external_source is None and not self._rift_source_ready(managed_source):
                source_stage = _under_runtime(
                    managed_source.with_name(f".RIFT-SVC.stage-{uuid.uuid4().hex}"),
                    self.runtime_dir,
                )
                await self._prepare_rift_source(source_stage)
                source_project = source_stage

            python_value = external_python or managed_python
            if external_python is None and not (
                self._validate_rift_environment(python_value) and self._rift_environment_record_ready()
            ):
                python_stage = await self._prepare_rift_environment(source_project)

            self._assert_no_active_workers()
            if source_stage is not None:
                self._activate_directory(source_stage, managed_source, validator=self._rift_source_ready)
                source_stage = None
            if python_stage is not None:
                self._activate_directory(
                    python_stage,
                    self.runtime_dir / "envs" / "rift",
                    validator=lambda env_dir: self._validate_rift_environment(self._python_executable(env_dir)),
                )
                python_stage = None
                lock = PLUGIN_DIR / str(self.manifest["rift"]["lock"])
                self._write_install_record(
                    "rift",
                    {
                        "ready": True,
                        "commit": str(self.manifest["rift"]["commit"]),
                        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                        "python_version": self.manifest["python"]["version"],
                    },
                )
            if not self._check_rift()["ready"]:
                raise PipelineError("rift_validation_failed", "The prepared RIFT source and isolated environment did not pass readiness checks.")
        finally:
            if source_stage is not None:
                self._remove_owned_directory(source_stage)
            if python_stage is not None:
                self._remove_owned_directory(python_stage)

    async def _prepare_rift_source(self, stage: Path) -> None:
        rift = self.manifest["rift"]
        repository = str(rift["repository"])
        commit = str(rift["commit"])
        stage.parent.mkdir(parents=True, exist_ok=True)
        env = self._child_env()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        await self._run(
            ["git", "clone", "--no-checkout", "--filter=blob:none", repository, str(stage)],
            env=env,
            timeout=300,
        )
        await self._run(["git", "-C", str(stage), "checkout", "--detach", commit], env=env, timeout=300)
        lock = PLUGIN_DIR / str(rift["lock"])
        source_lock = stage / "uv.lock"
        try:
            same_lock = source_lock.is_file() and _normalized_lock_digest(source_lock) == _normalized_lock_digest(lock)
        except OSError:
            same_lock = False
        if not same_lock:
            raise PipelineError("rift_lock_mismatch", "The pinned RIFT source lock does not match the packaged frozen dependency lock.")
        records: list[dict[str, Any]] = []
        for asset in rift.get("assets", []):
            relative = str(asset["path"])
            target = _under_runtime(stage / relative, self.runtime_dir)
            await self._fetch_asset(str(asset["url"]), target, str(asset["sha256"]), int(asset["size_bytes"]))
            stat_value = target.stat()
            digest = _sha256(target)
            if stat_value.st_size != int(asset["size_bytes"]) or digest != str(asset["sha256"]).casefold():
                raise PipelineError("rift_asset_hash_mismatch", "A pinned RIFT model module failed its checksum validation.")
            records.append({"name": relative, "size_bytes": stat_value.st_size, "mtime_ns": stat_value.st_mtime_ns, "sha256": digest, "verified": True})
        self._write_rift_asset_manifest(stage, records)
        if not self._rift_source_ready(stage):
            raise PipelineError("rift_source_validation_failed", "The pinned RIFT source did not pass revision and asset checks.")

    async def _prepare_rift_environment(self, source_project: Path) -> Path:
        rift = self.manifest["rift"]
        lock = PLUGIN_DIR / str(rift["lock"])
        source_lock = source_project / "uv.lock"
        try:
            lock_hash = _normalized_lock_digest(lock)
            source_lock_hash = _normalized_lock_digest(source_lock)
        except OSError:
            raise PipelineError("runtime_lock_missing", "The pinned RIFT dependency lock is unavailable.") from None
        if lock_hash != source_lock_hash:
            raise PipelineError("rift_lock_mismatch", "The RIFT project lock does not match the packaged frozen dependency lock.")

        uv = self._uv_exe()
        base_python = await self._uv_python()
        envs = self.runtime_dir / "envs"
        envs.mkdir(parents=True, exist_ok=True)
        stage = _under_runtime(envs / f".rift.stage-{uuid.uuid4().hex}", self.runtime_dir)
        child_env = self._child_env()
        child_env["UV_PROJECT_ENVIRONMENT"] = str(stage)
        try:
            await self._run([uv, "venv", "--python", str(base_python), str(stage)], env=child_env, timeout=120)
            stage_python = self._python_executable(stage)
            await self._run(
                [
                    uv,
                    "sync",
                    "--project",
                    str(source_project),
                    "--frozen",
                    "--no-install-project",
                    "--no-dev",
                    "--no-install-package",
                    "gradio",
                    "--no-install-package",
                    "hydra-core",
                    "--no-install-package",
                    "omegaconf",
                    "--no-install-package",
                    "schedulefree",
                    "--no-install-package",
                    "tensorboardx",
                    "--no-install-package",
                    "thop",
                    "--python",
                    str(stage_python),
                ],
                cwd=source_project,
                env=child_env,
                timeout=_COMMAND_TIMEOUT,
            )
            if not self._validate_rift_environment(stage_python):
                raise PipelineError("rift_environment_invalid", "The staged pinned RIFT environment did not pass package validation.")
            return stage
        except BaseException:
            self._remove_owned_directory(stage)
            raise

    def _validate_rift_environment(self, python: Path) -> bool:
        required = (
            "torch", "torchaudio", "torchcodec", "librosa", "pyloudnorm", "tqdm", "einops",
            "jaxtyping", "x-transformers", "torchdiffeq", "transformers", "lightning",
            "pytorch-lightning", "pyworld", "praat-parselmouth", "soundfile", "matplotlib", "wandb",
        )
        return self._validate_environment(python, required)

    def _write_rift_asset_manifest(self, source: Path, records: list[dict[str, Any]]) -> None:
        _atomic_json(source / "mus_library-assets.json", {"schema_version": 1, "rift_commit": self.manifest["rift"]["commit"], "files": records})


class _NeteaseCoordinator:
    def __init__(self, settings: Mapping[str, Any]):
        self.settings = dict(settings)
        self.runtime_dir = Path(settings["runtime_dir"]).expanduser().resolve()
        self.base_url = str(settings.get("netease", {}).get("base_url", "http://127.0.0.1:13306")).rstrip("/")
        self.lock = asyncio.Lock()
        self.process: asyncio.subprocess.Process | None = None
        self.references = 0
        self.leases: set[str] = set()
        self.process_owner = False


class NeteaseService:
    """A lazy lease on the managed loopback API; only its own child is stopped."""

    def __init__(self, settings: Mapping[str, Any], coordinator: _NeteaseCoordinator):
        self.settings = dict(settings)
        self._coordinator = coordinator
        self.runtime_dir = coordinator.runtime_dir
        self.base_url = coordinator.base_url
        self._lease_id = secrets.token_urlsafe(12)
        self._registered = False
        self._closed = False

    async def ensure_started(self) -> dict[str, str]:
        coordinator = self._coordinator
        async with coordinator.lock:
            if self._closed:
                self._closed = False
            if self.base_url != f"http://{_SERVICE_HOST}:{_SERVICE_PORT}":
                raise PipelineError("netease_url_unsupported", "The managed Netease API must use 127.0.0.1:13306.")
            if await self._identity_probe():
                self._register_lease()
                return {"base_url": self.base_url}
            if _port_open(_SERVICE_HOST, _SERVICE_PORT):
                raise PipelineError("netease_port_in_use", "Port 13306 is in use by a service that did not pass the Netease API identity check.")
            if coordinator.process is not None and coordinator.process.returncode is None:
                await self._wait_for_identity()
                self._register_lease()
                return {"base_url": self.base_url}
            manifest = _read_manifest()
            node = self.runtime_dir / "tools" / f"node-v{manifest['node']['version']}-win-x64" / "node.exe"
            api_root = self.runtime_dir / "service" / "netease-api"
            package = api_root / "node_modules" / str(manifest["netease_api"]["name"])
            if not node.is_file() or not (package / "server.js").is_file():
                raise PipelineError("netease_not_prepared", "Prepare the Netease service before starting account login or song search.")
            if os.name != "nt":
                raise PipelineError("platform_unsupported", "The managed Netease service is currently available on Windows x64.")
            service_dir = self.runtime_dir / "service"
            service_dir.mkdir(parents=True, exist_ok=True)
            private_tmp = service_dir / "tmp"
            private_tmp.mkdir(parents=True, exist_ok=True)
            env = _service_environment()
            env["PORT"] = str(_SERVICE_PORT)
            env["HOST"] = _SERVICE_HOST
            env["TEMP"] = str(private_tmp)
            env["TMP"] = str(private_tmp)
            env["TMPDIR"] = str(private_tmp)
            env["MUS_LIBRARY_SERVICE_VERSION"] = str(manifest["netease_api"]["version"])
            server_code = (
                "const fs=require('fs'),path=require('path'),os=require('os');"
                "fs.mkdirSync(os.tmpdir(),{recursive:true});"
                "const token=path.resolve(os.tmpdir(),'anonymous_token');"
                "if(!fs.existsSync(token))fs.writeFileSync(token,'','utf8');"
                "(async()=>{"
                "await require('NeteaseCloudMusicApi/generateConfig')();"
                "require('NeteaseCloudMusicApi/server').serveNcmApi("
                f"{{port:{_SERVICE_PORT},host:'{_SERVICE_HOST}',checkVersion:false}});"
                "})().catch(()=>process.exit(1));"
            )
            kwargs: dict[str, Any] = {
                "cwd": str(api_root),
                "stdin": asyncio.subprocess.DEVNULL,
                "stdout": asyncio.subprocess.DEVNULL,
                "stderr": asyncio.subprocess.DEVNULL,
                "env": env,
                "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            }
            spawn_task = asyncio.create_task(asyncio.create_subprocess_exec(str(node), "-e", server_code, **kwargs))
            try:
                process = await asyncio.shield(spawn_task)
            except asyncio.CancelledError as cancelled:
                try:
                    process = await spawn_task
                except BaseException:
                    raise cancelled from None
                try:
                    await _stop_owned_process(process)
                except BaseException:
                    pass
                raise cancelled
            except OSError:
                raise PipelineError("netease_service_start_failed", "The managed Netease service could not be started.") from None
            coordinator.process = process
            coordinator.process_owner = True
            try:
                await asyncio.sleep(0.15)
                await self._wait_for_identity(process=process)
                _atomic_json(service_dir / "netease-service.json", {"pid": process.pid, "version": manifest["netease_api"]["version"], "started_at": int(time.time())})
                self._register_lease()
            except BaseException:
                await _stop_owned_process(process)
                coordinator.process = None
                coordinator.process_owner = False
                try:
                    (service_dir / "netease-service.json").unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            return {"base_url": self.base_url}

    async def _wait_for_identity(self, *, process: asyncio.subprocess.Process | None = None) -> None:
        deadline = time.monotonic() + _SERVICE_START_TIMEOUT
        while time.monotonic() < deadline:
            if await self._identity_probe():
                return
            if process is not None and process.returncode is not None:
                break
            await asyncio.sleep(0.25)
        raise PipelineError("netease_service_start_failed", "The managed Netease API did not pass its startup identity check.")

    async def _identity_probe(self) -> bool:
        url = self.base_url + _SERVICE_IDENTITY_PATH
        def probe() -> bool:
            request = urllib.request.Request(url, headers={"Accept": "text/html", "User-Agent": "NachoBot-mus-library"})
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=1.2) as response:
                    if response.status != 200:
                        return False
                    payload = response.read(64 * 1024 + 1)
                    if len(payload) > 64 * 1024:
                        return False
                    return "网易云音乐 API" in payload.decode("utf-8", "replace")
            except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
                return False
        return await asyncio.to_thread(probe)

    def _register_lease(self) -> None:
        if self._registered:
            return
        self._registered = True
        self._coordinator.references += 1
        self._coordinator.leases.add(self._lease_id)

    async def close(self) -> None:
        if self._closed:
            return
        coordinator = self._coordinator
        async with coordinator.lock:
            self._closed = True
            if self._registered:
                self._registered = False
                coordinator.references = max(0, coordinator.references - 1)
                coordinator.leases.discard(self._lease_id)
            if coordinator.references == 0 and coordinator.process_owner and coordinator.process is not None:
                await _stop_owned_process(coordinator.process)
                coordinator.process = None
                coordinator.process_owner = False
                try:
                    (self.runtime_dir / "service" / "netease-service.json").unlink(missing_ok=True)
                except OSError:
                    pass


def _service_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in _NO_PROXY_KEYS:
        env.pop(key, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    return env


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex((host, port)) == 0


async def _stop_owned_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
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
            await asyncio.wait_for(killer.wait(), timeout=10)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (OSError, asyncio.TimeoutError):
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


def get_netease_service(settings: Mapping[str, Any] | None = None) -> NeteaseService:
    """Return a lazy reference-counted service lease shared in this Python process."""
    resolved = dict(settings) if settings is not None else load_settings()
    runtime_dir = Path(resolved["runtime_dir"]).expanduser().resolve()
    key = str(runtime_dir)
    shared_module_key = "_nachobot_mus_library_service_registry"
    shared = sys.modules.get(shared_module_key)
    if shared is None:
        shared = ModuleType(shared_module_key)
        shared.__dict__["coordinators"] = {}
        sys.modules[shared_module_key] = shared
    registry: dict[str, _NeteaseCoordinator] = shared.__dict__["coordinators"]
    coordinator = registry.get(key)
    if coordinator is None:
        coordinator = _NeteaseCoordinator(resolved)
        registry[key] = coordinator
    return NeteaseService(resolved, coordinator)


__all__ = ["NeteaseService", "RuntimeInstaller", "get_netease_service"]
