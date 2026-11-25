"""Plugin-owned asynchronous process facade for isolated audio runtimes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import threading
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from .settings import PipelineError, load_settings
from .phrase_assets import phrase_assets_ready
from .octave_probe_assets import (
    OCTAVE_PROBE_IMPLEMENTATION,
    OCTAVE_PROBE_SCHEMA_VERSION,
    octave_probe_assets_state,
)
from .workers.rift_profile import read_profile

if TYPE_CHECKING:
    from .installer import RuntimeInstaller


MAX_LINE_BYTES = 64 * 1024
MAX_AUDIO_BYTES = 512 * 1024 * 1024
DEFAULT_COMMAND_TIMEOUT = 900.0
_SPEAKER_CACHE: dict[tuple[str, int, int, str], list[dict[str, Any]]] = {}
_SPEAKER_CACHE_LOCK = threading.RLock()
_ASSET_MANIFEST_LIMIT = 2 * 1024 * 1024
logger = logging.getLogger(__name__)


def _under(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError:
        raise PipelineError("path_outside_runtime", "Audio job paths must stay inside the managed music runtime.") from None
    return resolved


def _worker_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if key.upper() in {"PYTHONPATH", "PYTHONHOME"}:
            env.pop(key, None)
    env["PYTHONUTF8"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _uvr_manifest(runtime_dir: Path) -> dict[str, Any] | None:
    cache = runtime_dir / "models" / "uvr"
    manifest_path = cache / "manifest.json"
    try:
        if manifest_path.stat().st_size > _ASSET_MANIFEST_LIMIT:
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        return None
    fingerprint = manifest.get("installer_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or not re.fullmatch(r"[a-fA-F0-9]{64}", fingerprint)
        or manifest.get("audio_separator_version") != "0.47.1"
    ):
        return None
    return manifest


def _prepared_model_files(
    runtime_dir: Path,
    model_name: str,
    *,
    expected_fingerprint: str | None = None,
) -> list[dict[str, Any]] | None:
    """Return installer-verified files for one model, or None if incomplete."""
    cache = runtime_dir / "models" / "uvr"
    manifest = _uvr_manifest(runtime_dir)
    if manifest is None or (
        expected_fingerprint and manifest.get("installer_fingerprint") != expected_fingerprint
    ):
        return None
    models = manifest.get("models")
    entry = models.get(model_name) if isinstance(models, dict) else None
    files = entry.get("files") if isinstance(entry, dict) else None
    if not isinstance(files, list) or not files:
        return None
    cleaned: list[dict[str, Any]] = []
    found_model = False
    for item in files[:256]:
        if not isinstance(item, dict) or item.get("verified") is not True:
            return None
        name = item.get("name")
        size = item.get("size_bytes")
        mtime_ns = item.get("mtime_ns")
        digest = item.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or type(size) is not int
            or size <= 0
            or type(mtime_ns) is not int
            or not isinstance(digest, str)
            or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)
        ):
            return None
        file_path = (cache / name).resolve(strict=False)
        try:
            file_path.relative_to(cache.resolve(strict=True))
            stat = file_path.stat()
        except (OSError, ValueError):
            return None
        if not file_path.is_file() or stat.st_size != size or stat.st_mtime_ns != mtime_ns:
            return None
        if file_path.name.casefold() == model_name.casefold():
            found_model = True
        cleaned.append({"name": name, "size_bytes": size, "mtime_ns": mtime_ns, "sha256": digest.lower()})
    if len(files) > 256 or not found_model:
        return None
    return cleaned


def _ffmpeg_executable(settings: dict[str, Any], runtime_dir: Path) -> Path | None:
    value = settings.get("ffmpeg_path")
    candidate = Path(value).expanduser() if value else runtime_dir / "models" / "tools" / "ffmpeg.exe"
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.is_file() else None


class _JsonLineWorker:
    def __init__(self, *, timeout: float = DEFAULT_COMMAND_TIMEOUT, marker_dir: Path | None = None, component: str = "worker"):
        self.timeout = timeout
        self.process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task | None = None
        self.marker_dir = marker_dir
        self.component = component
        self._active_marker: Path | None = None

    async def start(self, command: list[str], cwd: Path, startup: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            if self._alive():
                return {"ok": True, "ready": True}
            await self._stop_unlocked()
            flags = 0
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                process = await asyncio.create_subprocess_exec(
                    *command, cwd=str(cwd), stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    env=_worker_env(), creationflags=flags,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *command, cwd=str(cwd), stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    env=_worker_env(), start_new_session=True,
                )
            self.process = process
            self._stderr_task = asyncio.create_task(self._drain_stderr(process.stderr))
            try:
                self._write_active_marker(process)
                response = await self._exchange_unlocked(startup, self.timeout)
            except BaseException:
                await self._stop_unlocked()
                raise
            if not response.get("ok") or not response.get("ready"):
                await self._stop_unlocked()
                raise PipelineError("worker_start_failed", "The audio worker did not become ready.")
            return response

    def _write_active_marker(self, process: asyncio.subprocess.Process) -> None:
        if self.marker_dir is None:
            return
        nonce = secrets.token_urlsafe(18)
        marker = self.marker_dir / f"{self.component}-{process.pid}-{nonce}.json"
        temporary: str | None = None
        try:
            self.marker_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="\n", prefix=f".{self.component}-", suffix=".tmp",
                dir=self.marker_dir, delete=False,
            ) as output:
                temporary = output.name
                json.dump({"component": self.component, "pid": process.pid, "nonce": nonce}, output, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, marker)
            temporary = None
            self._active_marker = marker
        except OSError:
            raise PipelineError("runtime_marker_failed", "The audio worker could not register its active process.") from None
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    async def request(self, value: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        async with self._lock:
            if not self._alive():
                raise PipelineError("worker_not_ready", "The requested audio worker is not running.")
            try:
                response = await self._exchange_unlocked(value, timeout or self.timeout)
            except asyncio.CancelledError:
                await self._stop_unlocked()
                raise
            except asyncio.TimeoutError:
                await self._stop_unlocked()
                raise PipelineError("stage_timeout", "The audio worker exceeded its time limit.") from None
            except PipelineError:
                await self._stop_unlocked()
                raise
            except Exception:
                await self._stop_unlocked()
                raise PipelineError("worker_failed", "The audio worker stopped unexpectedly.") from None
            if not response.get("ok"):
                raise PipelineError(str(response.get("code", "worker_failed")), str(response.get("message", "Audio processing failed.")))
            return response

    def _alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def _exchange_unlocked(self, value: dict[str, Any], timeout: float) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise PipelineError("worker_not_ready", "The requested audio worker is not running.")
        payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(payload) > MAX_LINE_BYTES:
            raise PipelineError("worker_request_too_large", "The audio worker request exceeds its limit.")

        async def exchange() -> dict[str, Any]:
            process.stdin.write(payload)
            await process.stdin.drain()
            line = await process.stdout.readline()
            if not line:
                raise asyncio.IncompleteReadError(b"", 1)
            if len(line) > MAX_LINE_BYTES:
                raise PipelineError("worker_response_too_large", "The audio worker response exceeds its limit.")
            try:
                result = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise PipelineError("worker_invalid_response", "The audio worker returned an invalid response.") from None
            if not isinstance(result, dict):
                raise PipelineError("worker_invalid_response", "The audio worker returned an invalid response.")
            return result

        return await asyncio.wait_for(exchange(), timeout=max(0.1, timeout))

    async def _drain_stderr(self, stream) -> None:
        if stream is None:
            return
        try:
            while await stream.read(4096):
                # GPU libraries and upstream click/tqdm output can include
                # private file paths. Drain without capturing or logging.
                pass
        except (OSError, asyncio.CancelledError):
            return

    async def close(self) -> None:
        async with self._lock:
            await self._stop_unlocked()

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_unlocked()

    async def _stop_unlocked(self) -> None:
        process = self.process
        if process is None:
            return
        if process.returncode is None:
            try:
                if os.name == "nt":
                    killer = await asyncio.create_subprocess_exec(
                        "taskkill.exe", "/PID", str(process.pid), "/T", "/F",
                        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
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
                    if os.name == "nt":
                        killer = await asyncio.create_subprocess_exec(
                            "taskkill.exe", "/PID", str(process.pid), "/T", "/F",
                            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                        await asyncio.wait_for(killer.wait(), timeout=10)
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                except (OSError, asyncio.TimeoutError):
                    process.kill()
                await process.wait()
        # Do not permit replacement until the old child has definitely exited.
        if process.returncode is None:
            await process.wait()
        if process.stdin:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.stdin.wait_closed(), timeout=2)
            except (asyncio.TimeoutError, OSError, BrokenPipeError):
                pass
        if process.stdout:
            try:
                await asyncio.wait_for(process.stdout.read(), timeout=2)
            except (asyncio.TimeoutError, OSError):
                pass
        stderr_task = self._stderr_task
        self._stderr_task = None
        if stderr_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(stderr_task), timeout=2)
            except asyncio.TimeoutError:
                stderr_task.cancel()
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass
            except (OSError, asyncio.CancelledError):
                pass
        marker = self._active_marker
        self._active_marker = None
        if marker is not None:
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                # The process has exited, so a stale marker is safe for the
                # installer to resolve by checking its PID.
                pass
        self.process = None


class MusicRuntime:
    """Coordinates persistent UVR and RIFT child processes.

    Construction and import are cheap and never import Torch or audio-separator.
    ``ensure_started`` starts the configured RIFT worker when a checkpoint is
    configured; UVR starts on the first separation request.
    """

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = settings or load_settings()
        self.runtime_dir = Path(self.settings["runtime_dir"]).expanduser().resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.worker_dir = Path(__file__).resolve().parent / "workers"
        active_dir = self.runtime_dir / "active-workers"
        self._rift = _JsonLineWorker(
            timeout=float(self.settings["pipeline"]["stage_timeout_seconds"]),
            marker_dir=active_dir,
            component="rift",
        )
        self._audio = _JsonLineWorker(
            timeout=float(self.settings["pipeline"]["stage_timeout_seconds"]),
            marker_dir=active_dir,
            component="audio",
        )
        self._closed = False
        self._ready_response: dict[str, Any] | None = None
        self._lock = asyncio.Lock()
        self._voice_profiles: dict[str, tuple[dict, str]] = {}
        self._profile_error = False
        rift = self.settings["rift"]
        for role in ("lead", "backing"):
            profile_path = rift.get(f"{role}_voice_profile")
            if profile_path:
                speaker = rift.get("backing_speaker") if role == "backing" else None
                try:
                    self._voice_profiles[role] = read_profile(profile_path, speaker or rift.get("speaker"))
                except (OSError, ValueError):
                    self._profile_error = True

    def _audio_installer_fingerprint(self) -> str | None:
        try:
            from .installer import RuntimeInstaller

            snapshot = RuntimeInstaller(self.settings).status()
        except Exception:
            return None
        fingerprints = snapshot.get("fingerprints") if isinstance(snapshot, dict) else None
        value = fingerprints.get("audio") if isinstance(fingerprints, dict) else None
        return value if isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) else None

    def _rift_paths(self) -> tuple[Path, Path]:
        runtime = self.settings["runtime"]
        source = Path(runtime.get("rift_source") or self.runtime_dir / "source" / "RIFT-SVC").expanduser()
        python = Path(runtime.get("rift_python") or self._managed_python("rift")).expanduser()
        return source.resolve(), python.resolve()

    def _managed_python(self, name: str) -> Path:
        if os.name == "nt":
            return self.runtime_dir / "envs" / name / "Scripts" / "python.exe"
        return self.runtime_dir / "envs" / name / "bin" / "python"

    def _audio_python(self) -> Path:
        return self._managed_python("audio")

    def _rift_checkpoint(self) -> Path | None:
        value = self.settings["rift"].get("checkpoint")
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(self.settings["plugin_dir"]) / path
        try:
            return path.resolve(strict=True)
        except OSError:
            return None

    @property
    def model_fingerprints(self) -> dict[str, dict[str, str]]:
        """Stable configuration/model identity tokens without exposing paths."""
        separation: dict[str, str] = {}
        cache = self.runtime_dir / "models" / "uvr"
        audio_fingerprint = self._audio_installer_fingerprint()
        for key, filename in self.settings["separation"].items():
            path = cache / str(filename)
            identity_parts: list[str] = [
                f"audio-separator-0.47.1|{filename}|{self.settings['runtime']['device']}|{audio_fingerprint or 'unprepared'}"
            ]
            try:
                stat = path.stat()
                identity_parts.append(f"{stat.st_size}|{stat.st_mtime_ns}")
            except OSError:
                identity_parts.append("missing")
            files = _prepared_model_files(
                self.runtime_dir,
                str(filename),
                expected_fingerprint=audio_fingerprint,
            )
            if files:
                identity_parts.extend(f"{item['name']}|{item['size_bytes']}|{item['mtime_ns']}|{item['sha256']}" for item in files)
            separation[key] = hashlib.sha256("\n".join(identity_parts).encode("utf-8")).hexdigest()
        rift: dict[str, str] = {}
        settings = self.settings["rift"]
        for role, checkpoint_key in (("lead", "checkpoint"), ("backing", "backing_checkpoint")):
            value = settings.get(checkpoint_key) or settings.get("checkpoint")
            if not value:
                rift[role] = hashlib.sha256(b"rift-checkpoint-missing").hexdigest()
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = Path(self.settings["plugin_dir"]) / path
            try:
                path = path.resolve(strict=True)
                stat = path.stat()
                identity = f"rift:{path}|{stat.st_size}|{stat.st_mtime_ns}|{self.settings['runtime']['device']}"
            except OSError:
                identity = f"rift:{path}|missing|{self.settings['runtime']['device']}"
            if role in self._voice_profiles:
                identity += "|profile:" + self._voice_profiles[role][1]
            if role == "lead" and settings.get("octave_probe") is True:
                probe_state = octave_probe_assets_state(self.runtime_dir / "models/octave_probe", verify=True)
                identity += (
                    f"|{OCTAVE_PROBE_IMPLEMENTATION}|schema:{OCTAVE_PROBE_SCHEMA_VERSION}"
                    f"|target:{settings.get('octave_probe_target_voice', 'unknown')}"
                    f"|assets:{probe_state['state']}:{probe_state['fingerprint']}"
                )
            rift[role] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return {"separation": separation, "rift": rift}

    def target_model_fingerprints(self) -> dict[str, dict[str, str]]:
        """Return cheap model/config identity for job de-duplication."""
        return self.model_fingerprints

    async def ensure_started(self) -> None:
        async with self._lock:
            if self._closed:
                raise PipelineError("runtime_closed", "The music runtime has been closed.")
            await self._audio.stop()
            await self._ensure_rift_started_unlocked()

    async def _ensure_rift_started_unlocked(self) -> None:
        if self._rift._alive():
            return
        checkpoint = self._rift_checkpoint()
        if checkpoint is None:
            raise PipelineError("checkpoint_missing", "Choose a RIFT-SVC checkpoint before voice conversion.")
        source, python = self._rift_paths()
        if not source.is_dir() or not (source / "infer.py").is_file():
            raise PipelineError("rift_runtime_missing", "The pinned RIFT-SVC runtime is not prepared.")
        if not python.is_file():
            raise PipelineError("rift_python_missing", "The isolated RIFT-SVC Python environment is not prepared.")
        if not self._rift_pretrained_ready(source):
            raise PipelineError("rift_assets_missing", "The RIFT-SVC pretrained modules are not prepared.")
        startup = {
            "cmd": "start",
            "settings": {"checkpoint": str(checkpoint), "device": self.settings["runtime"]["device"]},
        }
        command = [str(python), "-u", str(self.worker_dir / "rift_worker.py"), "--source", str(source), "--runtime-dir", str(self.runtime_dir)]
        self._ready_response = await self._rift.start(command, source, startup)

    @staticmethod
    def _rift_pretrained_ready(source: Path) -> bool:
        required = (
            source / "pretrained" / "nsf_hifigan_44.1k_hop512_128bin_2024.02" / "model.ckpt",
            source / "pretrained" / "nsf_hifigan_44.1k_hop512_128bin_2024.02" / "config.json",
            source / "pretrained" / "rmvpe" / "model.pt",
            source / "pretrained" / "content-vec-best" / "config.json",
            source / "pretrained" / "content-vec-best" / "pytorch_model.bin",
        )
        return all(path.is_file() and path.stat().st_size > 0 for path in required)

    async def check_ready(self) -> dict[str, Any]:
        """Cheap readiness snapshot; does not start either worker or install assets."""
        checkpoint = self._rift_checkpoint()
        source, python = self._rift_paths()
        rift_settings = self.settings["rift"]
        phrases_required = bool(rift_settings.get("adaptive_quality") and rift_settings.get("timbre_slicing", True))
        phrase_models_ready = not phrases_required or phrase_assets_ready(self.runtime_dir / "models/phrases")
        probe_state = (
            octave_probe_assets_state(self.runtime_dir / "models/octave_probe", verify=True)
            if rift_settings.get("octave_probe") is True else None
        )
        backing_value = rift_settings.get("backing_checkpoint") or rift_settings.get("checkpoint")
        backing_checkpoint = Path(backing_value).expanduser() if backing_value else None
        if backing_checkpoint is not None and not backing_checkpoint.is_absolute():
            backing_checkpoint = Path(self.settings["plugin_dir"]) / backing_checkpoint
        speaker = str(rift_settings.get("speaker") or "")
        backing_speaker = str(rift_settings.get("backing_speaker") or speaker)
        rift_ready = bool(
            checkpoint
            and backing_checkpoint is not None
            and backing_checkpoint.is_file()
            and speaker
            and backing_speaker
            and source.is_dir()
            and (source / "infer.py").is_file()
            and python.is_file()
            and self._rift_pretrained_ready(source)
            and not self._profile_error
            and phrase_models_ready
        )
        audio_python = self._audio_python()
        ffmpeg = _ffmpeg_executable(self.settings, self.runtime_dir)
        installer_fingerprint = self._audio_installer_fingerprint()
        model_readiness = {
            key: _prepared_model_files(
                self.runtime_dir,
                str(filename),
                expected_fingerprint=installer_fingerprint,
            ) is not None
            for key, filename in self.settings["separation"].items()
        }
        separation_ready = bool(
            installer_fingerprint and audio_python.is_file() and ffmpeg and all(model_readiness.values())
        )
        netease_parts = urlsplit(str(self.settings["netease"]["base_url"]))
        netease_host = (netease_parts.hostname or "").casefold()
        if netease_host in {"localhost", "::1"}:
            netease_host = "127.0.0.1"
        components = {
            "rift": {
                "ready": rift_ready,
                "configured": bool(checkpoint),
                "checkpoint": bool(checkpoint),
                "speaker": bool(speaker and backing_speaker),
                "pretrained": self._rift_pretrained_ready(source),
                "voice_profiles_valid": not self._profile_error,
                "phrase_models_ready": phrase_models_ready,
            },
            "separation": {
                "ready": separation_ready,
                "configured": True,
                "models": model_readiness,
                "ffmpeg": ffmpeg is not None,
                "installer_fingerprint": bool(installer_fingerprint),
            },
            "netease": {
                "ready": await self._netease_probe(),
                "managed": netease_host == "127.0.0.1" and netease_parts.port == 13306,
            },
        }
        result = {
            "ready": bool(components["rift"]["ready"] and components["separation"]["ready"]),
            "components": components,
            "model_fingerprints": self.model_fingerprints,
        }
        if probe_state is not None:
            result["octave_probe"] = {
                "enabled": True,
                "state": probe_state["state"],
                "fingerprint": probe_state["fingerprint"],
                "ready": probe_state["state"] == "prepared" and phrase_assets_ready(
                    self.runtime_dir / "models/phrases"
                ),
            }
        return result

    async def _netease_probe(self) -> bool:
        # Cheap loopback-only check; never requests song URLs or credentials.
        try:
            import urllib.request

            base_url = str(self.settings["netease"]["base_url"]).rstrip("/")
            parts = urlsplit(base_url)
            if (
                parts.scheme != "http"
                or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parts.username
                or parts.password
                or parts.path
                or parts.query
                or parts.fragment
                or parts.port is None
            ):
                return False
            url = base_url + "/search?keywords=nachobot-runtime-check&limit=1"

            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *_args, **_kwargs):
                    return None

            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

            def probe() -> bool:
                try:
                    with opener.open(url, timeout=0.25) as response:
                        if response.status != 200:
                            return False
                        body = response.read(64 * 1024 + 1)
                        if len(body) > 64 * 1024:
                            return False
                        payload = json.loads(body)
                        if not isinstance(payload, dict) or payload.get("code") != 200:
                            return False
                        result = payload.get("result")
                        if not isinstance(result, dict):
                            return False
                        return isinstance(result.get("songs"), list) or (
                            type(result.get("songCount")) is int and result["songCount"] == 0
                        )
                except Exception:
                    return False

            return await asyncio.to_thread(probe)
        except Exception:
            return False

    async def separate(self, input: Path, output_dir: Path, kind: str) -> dict[str, Path]:
        async with self._lock:
            if self._closed:
                raise PipelineError("runtime_closed", "The music runtime has been closed.")
            if kind not in {"vocals", "backing", "clean"}:
                raise PipelineError("invalid_stage", "The requested separation stage is not supported.")
            source = _under(Path(input), self.runtime_dir, must_exist=True)
            if not source.is_file() or source.stat().st_size > MAX_AUDIO_BYTES:
                raise PipelineError("invalid_audio", "The source audio is missing or exceeds the worker limit.")
            target = Path(output_dir).expanduser().resolve()
            try:
                target.relative_to(self.runtime_dir)
            except ValueError:
                raise PipelineError("path_outside_runtime", "Audio job paths must stay inside the managed music runtime.") from None
            target.mkdir(parents=True, exist_ok=True)
            python = self._audio_python()
            if not python.is_file():
                raise PipelineError("audio_runtime_missing", "The isolated UVR environment is not prepared.")
            model = str(self.settings["separation"][f"{kind}_model"])
            installer_fingerprint = self._audio_installer_fingerprint()
            if _prepared_model_files(
                self.runtime_dir,
                model,
                expected_fingerprint=installer_fingerprint,
            ) is None or installer_fingerprint is None:
                raise PipelineError("separation_assets_missing", "Prepare the configured separator model before processing audio.")
            ffmpeg = _ffmpeg_executable(self.settings, self.runtime_dir)
            if ffmpeg is None:
                raise PipelineError("ffmpeg_missing", "The plugin-managed FFmpeg executable is not prepared.")
            # Keep heavyweight GPU allocations sequential: release RIFT before
            # loading UVR, then reuse UVR only across separation calls.
            await self._rift.stop()
            if not self._audio._alive():
                startup = {
                    "cmd": "start",
                    "settings": {
                        "model_cache": str(self.runtime_dir / "models" / "uvr"),
                        "device": self.settings["runtime"]["device"],
                        "ffmpeg_path": str(ffmpeg),
                    },
                }
                command = [str(python), "-u", str(self.worker_dir / "audio_worker.py"), "--runtime-dir", str(self.runtime_dir)]
                await self._audio.start(command, self.runtime_dir, startup)
            response = await self._audio.request({"cmd": "separate", "input": str(source), "output_dir": str(target), "kind": kind, "model": model})
            paths = response.get("stems")
            if not isinstance(paths, dict):
                raise PipelineError("worker_invalid_response", "The separator returned an invalid stem map.")
            expected = {"vocals": {"vocals", "instrumental"}, "backing": {"lead", "backing"}, "clean": {"dry"}}[kind]
            if set(paths) != expected:
                raise PipelineError("stem_map_invalid", "The separator could not map the requested output stems.")
            result: dict[str, Path] = {}
            for name, value in paths.items():
                path = _under(Path(str(value)), target, must_exist=True)
                if not path.is_file() or path.stat().st_size > MAX_AUDIO_BYTES:
                    raise PipelineError("invalid_audio", "A separated stem is missing or exceeds the worker limit.")
                result[name] = path
            return result

    async def convert(self, input: Path, output: Path, role: str, *, song_octave_shift: int = 0) -> Path:
        async with self._lock:
            if self._closed:
                raise PipelineError("runtime_closed", "The music runtime has been closed.")
            if self._profile_error:
                raise PipelineError("rift_profile_invalid", "A configured RIFT voice profile cannot be read or does not match its speaker.")
            rift = self.settings["rift"]
            if rift.get("adaptive_quality") and rift.get("timbre_slicing", True) and not phrase_assets_ready(self.runtime_dir / "models/phrases"):
                raise PipelineError("rift_phrase_models_missing", "Run prepare-phrases before using RIFT phrase clustering.")
            if role not in {"lead", "backing"}:
                raise PipelineError("invalid_role", "The requested voice role is not supported.")
            source = _under(Path(input), self.runtime_dir, must_exist=True)
            target = Path(output).expanduser().resolve()
            try:
                target.relative_to(self.runtime_dir)
            except ValueError:
                raise PipelineError("path_outside_runtime", "Audio job paths must stay inside the managed music runtime.") from None
            if not source.is_file() or source.stat().st_size > MAX_AUDIO_BYTES or target.suffix.casefold() != ".wav":
                raise PipelineError("invalid_audio", "The source or destination audio path is invalid.")
            await self._audio.stop()
            await self._ensure_rift_started_unlocked()
            checkpoint_key = "backing_checkpoint" if role == "backing" else "checkpoint"
            speaker_key = "backing_speaker" if role == "backing" else "speaker"
            shift_key = "backing_key_shift" if role == "backing" else "key_shift"
            if type(song_octave_shift) is not int or song_octave_shift not in {-12, 0, 12}:
                raise PipelineError("invalid_octave_shift", "The song octave shift must be -12, 0, or 12.")
            configured_song_shift = rift.get("song_octave_shift", 0)
            if type(configured_song_shift) is not int or configured_song_shift not in {-12, 0, 12}:
                raise PipelineError("invalid_settings", "[rift].song_octave_shift must be -12, 0, or 12.")
            manual_shift = int(rift.get(shift_key, 0))
            if configured_song_shift and manual_shift:
                raise PipelineError(
                    "invalid_settings",
                    "[rift].song_octave_shift cannot be combined with a nonzero per-role key shift.",
                )
            # A per-role manual displacement suppresses a propagated automatic
            # whole-song decision. Validated settings reject a configured lock
            # alongside either manual override.
            effective_song_shift = configured_song_shift or (song_octave_shift if manual_shift == 0 else 0)
            octave_probe_eligible = bool(
                role == "lead"
                and rift.get("octave_probe") is True
                and rift.get("adaptive_quality") is True
                and rift.get("timbre_slicing", True) is True
                and rift.get("domain_alignment", True) is True
                and rift.get("allow_octave_jump") is True
                and configured_song_shift == 0
                and manual_shift == 0
                and int(rift.get("backing_key_shift", 0)) == 0
                and effective_song_shift == 0
            )
            octave_probe_assets = (
                octave_probe_assets_state(self.runtime_dir / "models/octave_probe", verify=True)
                if octave_probe_eligible else None
            )
            checkpoint_value = rift.get(checkpoint_key) or rift.get("checkpoint")
            checkpoint = Path(checkpoint_value).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = Path(self.settings["plugin_dir"]) / checkpoint
            checkpoint = checkpoint.resolve(strict=True)
            speaker = rift.get(speaker_key) or rift.get("speaker")
            if not speaker:
                raise PipelineError("speaker_missing", "Choose a speaker from the configured RIFT checkpoint.")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.parent.resolve(strict=True).relative_to(self.runtime_dir)
            except ValueError:
                raise PipelineError("path_outside_runtime", "Audio job paths must stay inside the managed music runtime.") from None
            domain_ffmpeg = _ffmpeg_executable(self.settings, self.runtime_dir)
            response = await self._rift.request(
                {
                    "cmd": "convert",
                    "role": role,
                    "rift": {
                        "checkpoint": str(checkpoint),
                        "speaker": str(speaker),
                        "key_shift": int(rift[shift_key]),
                        "song_octave_shift": effective_song_shift,
                        # The worker uses this other-role context to avoid making
                        # a lead-only automatic octave decision when backing has
                        # an explicit manual displacement.
                        "song_detection_backing_manual_shift": int(rift.get("backing_key_shift", 0)),
                        "infer_steps": int(rift["infer_steps"]),
                        "ds_cfg_strength": float(rift["ds_cfg_strength"]),
                        "spk_cfg_strength": float(rift["spk_cfg_strength"]),
                        "cfg_rescale": float(rift.get("cfg_rescale", 0.7)),
                        "batch_size": int(rift["batch_size"]),
                        "robust_f0": int(rift["robust_f0"]),
                        "adaptive_quality": bool(rift.get("adaptive_quality", False)),
                        "timbre_slicing": bool(rift.get("timbre_slicing", True)),
                        "domain_alignment": bool(rift.get("domain_alignment", True)),
                        "domain_formants": bool(rift.get("domain_formants", True)),
                        "allow_octave_jump": bool(rift.get("allow_octave_jump", False)),
                        "pitch_range": rift.get(f"{role}_pitch_range", []),
                        **({"voice_profile": self._voice_profiles[role][0]} if role in self._voice_profiles else {}),
                        **({"domain_ffmpeg": str(domain_ffmpeg)} if domain_ffmpeg else {}),
                        "device": str(self.settings["runtime"]["device"]),
                        **({
                            "octave_probe": True,
                            "octave_probe_target_voice": str(rift.get("octave_probe_target_voice", "unknown")),
                            "octave_probe_asset_state": octave_probe_assets["state"],
                            "octave_probe_asset_fingerprint": octave_probe_assets["fingerprint"],
                        } if octave_probe_assets is not None else {}),
                    },
                    "input": str(source),
                    "output": str(target),
                },
                timeout=float(self.settings["pipeline"]["stage_timeout_seconds"]),
            )
            path = _under(Path(str(response.get("path", ""))), self.runtime_dir, must_exist=True)
            if not path.is_file() or path.stat().st_size > MAX_AUDIO_BYTES:
                raise PipelineError("worker_invalid_response", "RIFT returned an invalid output file.")
            return path

    async def speakers(self) -> list[dict[str, str]]:
        async with self._lock:
            if self._closed:
                raise PipelineError("runtime_closed", "The music runtime has been closed.")
            await self._audio.stop()
            await self._ensure_rift_started_unlocked()
            if self._ready_response:
                values = self._ready_response.get("speakers", [])
                if isinstance(values, list):
                    return values[:2048]
            response = await self._rift.request({"cmd": "speakers"})
            values = response.get("speakers", [])
            return values[:2048] if isinstance(values, list) else []

    async def release_workers(self) -> None:
        """Exit inference processes while keeping the runtime usable for the next job."""
        async with self._lock:
            self._ready_response = None
            stopping = asyncio.gather(
                self._audio.stop(), self._rift.stop(), return_exceptions=True,
            )
            try:
                audio_result, rift_result = await asyncio.shield(stopping)
            except asyncio.CancelledError:
                await stopping
                raise
            for component, result in (("UVR", audio_result), ("RIFT", rift_result)):
                if isinstance(result, Exception):
                    logger.warning("Could not stop the music %s worker: %s", component, type(result).__name__)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            await asyncio.gather(self._rift.close(), self._audio.close(), return_exceptions=True)


def _speaker_snapshot_sync(settings: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoint_value = settings.get("rift", {}).get("checkpoint")
    if not checkpoint_value:
        return []
    checkpoint = Path(checkpoint_value).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path(settings["plugin_dir"]) / checkpoint
    try:
        checkpoint = checkpoint.resolve(strict=True)
        stat = checkpoint.stat()
    except OSError:
        return []
    runtime_dir = Path(settings["runtime_dir"]).expanduser().resolve()
    runtime_settings = settings.get("runtime", {})
    python = Path(runtime_settings.get("rift_python") or (runtime_dir / "envs" / "rift" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))).expanduser()
    if not python.is_file():
        return []
    cache_key = (str(checkpoint), stat.st_size, stat.st_mtime_ns, str(python.resolve()))
    with _SPEAKER_CACHE_LOCK:
        cached = _SPEAKER_CACHE.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]
    command = [str(python), "-I", "-u", str(Path(__file__).resolve().parent / "workers" / "rift_metadata_worker.py"), "--checkpoint", str(checkpoint)]
    env = _worker_env()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    kwargs = {"creationflags": flags} if os.name == "nt" else {"start_new_session": True}
    try:
        result = subprocess.run(
            command,
            cwd=str(checkpoint.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=180,
            env=env,
            check=False,
            **kwargs,
        )
        if result.returncode != 0 or len(result.stdout) > 128 * 1024:
            return []
        payload = json.loads(result.stdout)
        speakers = payload.get("speakers") if isinstance(payload, dict) and payload.get("ok") else None
        if not isinstance(speakers, list) or len(speakers) > 2048:
            return []
        clean = [
            {"id": str(item["id"]), "name": str(item["name"])}
            for item in speakers
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("name"), str)
            and len(item["id"]) <= 128
            and len(item["name"]) <= 128
        ]
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        return []
    with _SPEAKER_CACHE_LOCK:
        _SPEAKER_CACHE[cache_key] = [dict(item) for item in clean]
    return clean


async def discover_speakers(settings: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Return cached checkpoint labels from a CPU/meta-only child process.

    This helper never constructs an inference model, imports CUDA code in Core,
    or downloads RIFT assets; call it only when speaker choices are requested.
    """
    resolved = settings or load_settings()
    return await asyncio.to_thread(_speaker_snapshot_sync, resolved)


def __getattr__(name: str):
    if name == "RuntimeInstaller":
        from .installer import RuntimeInstaller

        return RuntimeInstaller
    raise AttributeError(name)


__all__ = ["MusicRuntime", "RuntimeInstaller", "discover_speakers"]
