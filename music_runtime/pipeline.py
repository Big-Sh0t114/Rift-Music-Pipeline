"""Durable, sender-free music production orchestration."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import audio
from .job_store import JobStore, QueueLimitError
from .settings import PipelineError, SelectionRequired, netease_source_fingerprint


VOICE_PROCESSING_VERSION = 9
MAX_SONG_OCTAVE_DECISION_BYTES = 1024


def _safe_song(song: dict[str, Any]) -> dict[str, Any]:
    """Keep only public metadata; provider URLs may contain temporary tokens."""
    return {
        "id": str(song.get("id", song.get("song_id", "")))[:128],
        "title": str(song.get("title", ""))[:200],
        "artist": str(song.get("artist", ""))[:200],
        "duration_ms": int(song.get("duration_ms", 0) or 0),
    }


def _song_identity(song: dict[str, Any]) -> str:
    identity = _safe_song(song)["id"]
    if identity:
        return identity
    return audio.settings_fingerprint([song.get("title", ""), song.get("artist", "")])


def _contained(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PipelineError("unsafe_path", "A pipeline stage returned a path outside its job directory.") from exc
    return resolved


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(part) for key, part in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(part) for part in value]
    return str(value)


def configured_source_fingerprint(settings: dict[str, Any]) -> str:
    """Derive the provider identity without opening a session or loading Core."""
    return netease_source_fingerprint(settings)


def _fingerprint_sections(settings: dict[str, Any]) -> dict[str, Any]:
    """Keep disabled optional probe defaults out of legacy cache identities."""
    sections = {key: settings.get(key, {}) for key in ("separation", "rift", "mix", "runtime")}
    rift = dict(sections.get("rift", {}))
    if rift.get("octave_probe") is not True:
        rift.pop("octave_probe", None)
        rift.pop("octave_probe_target_voice", None)
    sections["rift"] = rift
    return sections


def configured_target_fingerprint(
    settings: dict[str, Any],
    model_fingerprints: dict[str, Any] | None = None,
) -> str:
    target = {
        "voice_processing_version": VOICE_PROCESSING_VERSION,
        "sections": _fingerprint_sections(settings),
        "model_fingerprints": model_fingerprints or {},
        "provider": {
            "base_url": settings.get("netease", {}).get("base_url"),
            "profile": settings.get("netease", {}).get("profile"),
        },
        "source_fingerprint": configured_source_fingerprint(settings),
    }
    return audio.settings_fingerprint(target)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as target:
            temporary = target.name
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


def _automatic_song_octave_shift(decision_path: Path) -> int:
    """Read the worker's tiny automatic-only sidecar for backing conversion."""
    try:
        if decision_path.stat().st_size > MAX_SONG_OCTAVE_DECISION_BYTES:
            return 0
        with decision_path.open("rb") as source:
            payload = source.read(MAX_SONG_OCTAVE_DECISION_BYTES + 1)
        if len(payload) > MAX_SONG_OCTAVE_DECISION_BYTES:
            return 0
        report = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 0
    if not isinstance(report, dict) or report.get("song_octave_source") != "automatic":
        return 0
    shift = report.get("song_octave_shift")
    return shift if type(shift) is int and shift in {-12, 12} else 0


class MusicPipeline:
    """A single-GPU producer with durable SQLite checkpoints."""

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        store: JobStore | None = None,
        provider: Any = None,
        runtime: Any = None,
    ):
        self.settings = settings
        runtime_dir = Path(settings["runtime_dir"]).resolve()
        self.job_root = runtime_dir / "jobs"
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.store = store or JobStore(runtime_dir / "jobs.sqlite")
        self.provider = provider
        self.runtime = runtime
        self._started = False
        self._stopping = False
        self._gpu_lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task] = {}
        self._component_lock = asyncio.Lock()
        self._runtime_manifest: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.settings.get("pipeline", {}).get("enabled", False))

    async def ensure_started(self) -> None:
        async with self._component_lock:
            if not self._started:
                self.store.recover_after_restart()
                if self.provider is None:
                    from .providers.netease import NeteaseProvider

                    self.provider = NeteaseProvider(self.settings)
                if self.runtime is None:
                    from .runtime import MusicRuntime

                    self.runtime = MusicRuntime(self.settings)
                self._started = True
                self._stopping = False
                for job in self.store.list_jobs(limit=500):
                    if self.enabled and job["status"] == "queued":
                        self._spawn(job["id"])

    async def submit(
        self,
        request_id: str,
        query: str,
    ) -> dict[str, Any]:
        await self.ensure_started()
        if not self.enabled:
            raise PipelineError("pipeline_disabled", "New music jobs are disabled by the operator.")
        params = self._target_parameters()
        try:
            job, created = self.store.create_request_job(
                request_id, query, params, queue_limit=int(self.settings["pipeline"]["max_queue"])
            )
        except QueueLimitError as exc:
            raise PipelineError("queue_full", "The music production queue is full; try again later.") from exc
        if created:
            self._spawn(job["id"])
        elif job["status"] == "queued":
            self._spawn(job["id"])
        return self.store.get_job_view(job["id"]) or job

    async def choose(
        self,
        job_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        await self.ensure_started()
        try:
            job, started = self.store.select_candidate(job_id, candidate_id)
        except KeyError:
            raise PipelineError("job_not_found", "No matching music job was found.") from None
        except ValueError as exc:
            raise PipelineError("selection_conflict", str(exc)) from None
        canonical_id = str(job.get("canonical_id", job["id"]))
        if started or job["status"] == "queued":
            self._spawn(canonical_id)
        return self.store.get_job_view(job_id) or job

    async def close(self) -> None:
        self._stopping = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        if self.provider is not None:
            try:
                await self.provider.close()
            except Exception:
                pass
        if self.runtime is not None:
            try:
                await self.runtime.close()
            except Exception:
                pass
        self._started = False

    def _target_parameters(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        model_fingerprints = {}
        if self.runtime is not None:
            try:
                model_fingerprints = self.runtime.target_model_fingerprints()
            except Exception:
                model_fingerprints = getattr(self.runtime, "model_fingerprints", {}) or {}
        target = {
            "voice_processing_version": VOICE_PROCESSING_VERSION,
            "sections": _fingerprint_sections(self.settings),
            "model_fingerprints": model_fingerprints,
            "provider": {
                "base_url": self.settings.get("netease", {}).get("base_url"),
                "profile": self.settings.get("netease", {}).get("profile"),
            },
            "source_fingerprint": getattr(
                self.provider, "provider_fingerprint", configured_source_fingerprint(self.settings)
            ),
        }
        target_fp = audio.settings_fingerprint(target)
        return {"target_fingerprint": target_fp, "target": _json_safe(target)}

    def _spawn(self, job_id: str) -> None:
        current = self._tasks.get(job_id)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(self._run_job(job_id), name=f"mus-library-{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda done, jid=job_id: self._forget_task(jid, done))

    def _forget_task(self, job_id: str, task: asyncio.Task) -> None:
        if self._tasks.get(job_id) is task:
            self._tasks.pop(job_id, None)

    def _job_work_dir(self, job_id: str) -> Path:
        path = (self.job_root / job_id).resolve()
        if self.job_root.resolve() not in path.parents:
            raise PipelineError("unsafe_path", "Invalid job directory.")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cancelled(self, job_id: str) -> bool:
        row = self.store.get_job(job_id)
        return self._stopping or row is None or row["status"] == "cancelled"

    def _stage_fingerprint(self, stage: str, inputs: dict[str, str], settings: Any) -> str:
        return audio.settings_fingerprint({"stage": stage, "inputs": inputs, "settings": settings})

    async def _content_fingerprint(self, path: Path, job_id: str) -> str:
        fingerprint = await audio.run_blocking(audio.content_fingerprint, path)
        if self._cancelled(job_id):
            raise asyncio.CancelledError
        return fingerprint

    async def _content_fingerprints(self, paths: dict[str, Path], job_id: str) -> dict[str, str]:
        names = list(paths)
        values = await asyncio.gather(
            *(audio.run_blocking(audio.content_fingerprint, paths[name]) for name in names)
        )
        if self._cancelled(job_id):
            raise asyncio.CancelledError
        return dict(zip(names, values, strict=True))

    async def _cached(
        self, job_id: str, stage: str, fingerprint: str, expected: tuple[str, ...]
    ) -> dict[str, Path] | None:
        cached = self.store.cached_artifact(job_id, stage, fingerprint)
        if cached is None:
            cached = self.store.cached_fingerprint(stage, fingerprint)
        if not cached or set(cached) != set(expected):
            return None
        paths: dict[str, Path] = {}
        job_root = self._job_work_dir(job_id)
        try:
            for key, value in cached.items():
                source = _contained(Path(value), self.job_root)
                try:
                    source.relative_to(job_root)
                    target = source
                except ValueError:
                    reuse_dir = job_root / "stages" / stage / f"reuse-{time.time_ns()}-{uuid.uuid4().hex[:6]}"
                    reuse_dir.mkdir(parents=True, exist_ok=False)
                    target = reuse_dir / f"{key}{source.suffix}"
                    await audio.run_blocking(shutil.copyfile, source, target)
                    if self._cancelled(job_id):
                        raise asyncio.CancelledError from None
                    target_stat = await audio.run_blocking(Path.stat, target)
                    if target_stat.st_size <= 0:
                        return None
                paths[key] = target
            if self._cancelled(job_id):
                raise asyncio.CancelledError
            self.store.record_artifact(
                job_id, stage, fingerprint, {key: str(path) for key, path in paths.items()}
            )
            return paths
        except asyncio.CancelledError:
            raise
        except PipelineError:
            return None
        except OSError:
            return None

    async def _perform_stage(
        self,
        job_id: str,
        stage: str,
        fingerprint: str,
        expected: tuple[str, ...],
        producer: Callable[[Path], Awaitable[dict[str, Path]]],
        *,
        audio_outputs: tuple[str, ...] = (),
        allow_silence: tuple[str, ...] = (),
    ) -> dict[str, Path]:
        cached = await self._cached(job_id, stage, fingerprint, expected)
        if cached:
            try:
                def validate_cached() -> None:
                    for key, path in cached.items():
                        if key in audio_outputs:
                            audio.validate_wav(path, reject_silence=key not in allow_silence)
                        elif not path.is_file() or path.stat().st_size <= 0:
                            raise audio.AudioError("Cached output is empty")

                await audio.run_blocking(validate_cached)
                if self._cancelled(job_id):
                    raise asyncio.CancelledError
                return cached
            except asyncio.CancelledError:
                raise
            except (audio.AudioError, OSError):
                pass
        if self._cancelled(job_id):
            raise asyncio.CancelledError
        timeout = max(10, int(self.settings["pipeline"].get("stage_timeout_seconds", 900)))
        root = self._job_work_dir(job_id)
        attempt_dir = root / "stages" / stage / f"attempt-{time.time_ns()}-{uuid.uuid4().hex[:6]}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        outputs = await asyncio.wait_for(producer(attempt_dir), timeout=timeout)
        if self._cancelled(job_id):
            raise asyncio.CancelledError
        if set(outputs) != set(expected):
            raise PipelineError("invalid_stage_output", f"The {stage} stage returned incomplete output.")
        normalized: dict[str, Path] = {}
        for key, raw in outputs.items():
            path = _contained(Path(raw), root)
            normalized[key] = path

        def validate_outputs() -> None:
            for key, path in normalized.items():
                if key in audio_outputs:
                    audio.validate_wav(path, reject_silence=key not in allow_silence)
                elif not path.is_file() or path.stat().st_size <= 0:
                    raise PipelineError("invalid_stage_output", f"The {stage} stage returned an empty file.")

        try:
            await audio.run_blocking(validate_outputs)
        except audio.AudioError as exc:
            raise PipelineError("invalid_stage_output", f"The {stage} stage returned invalid audio.") from exc
        if self._cancelled(job_id):
            raise asyncio.CancelledError
        self.store.record_artifact(job_id, stage, fingerprint, {key: str(path) for key, path in normalized.items()})
        return normalized

    async def _run_job(self, job_id: str) -> None:
        async with self._gpu_lock:
            try:
                try:
                    await self._produce(job_id)
                finally:
                    await self.runtime.release_workers()
            except asyncio.CancelledError:
                current = self.store.get_job(job_id)
                if current and current["status"] != "cancelled" and not self._stopping:
                    self.store.update(job_id, status="paused", stage="paused", progress="paused safely")
                raise
            except PipelineError as exc:
                if self.store.get_job(job_id):
                    self.store.update(
                        job_id,
                        status="failed",
                        error_code=exc.code,
                        error_message=exc.message,
                        progress="failed",
                    )
            except asyncio.TimeoutError:
                self.store.update(
                    job_id,
                    status="failed",
                    error_code="stage_timeout",
                    error_message="A processing stage exceeded its time limit.",
                    progress="failed",
                )
            except Exception as exc:
                self.store.update(
                    job_id,
                    status="failed",
                    error_code="pipeline_failed",
                    error_message=f"Music processing failed ({type(exc).__name__}).",
                    progress="failed",
                )

    async def _prepare_runtime(self) -> dict[str, Any]:
        ready = await self.runtime.check_ready()
        if not isinstance(ready, dict) or not ready.get("ready"):
            components = ready.get("components", {}) if isinstance(ready, dict) else {}
            missing = [str(key) for key, value in components.items() if not value] if isinstance(components, dict) else []
            detail = ", ".join(missing[:4])
            raise PipelineError("runtime_not_ready", "Audio models are not ready" + (f": {detail}" if detail else "."))
        return ready

    async def _produce(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if not job or self._cancelled(job_id):
            return
        if job["status"] == "ready":
            return
        if job["status"] == "selection_required":
            return

        song: dict[str, Any] | None = job.get("song") or None
        self.store.update(job_id, status="resolving", stage="resolving", progress="Finding a song")
        if song and song.get("id"):
            try:
                resolved = await self.provider.resolve(str(song["id"]))
            except SelectionRequired:
                resolved = None
            except PipelineError:
                resolved = None
            song = resolved or None
        if song is None:
            try:
                song = await self.provider.resolve(job["query"])
            except SelectionRequired as exc:
                self.store.update(
                    job_id,
                    status="selection_required",
                    stage="selection_required",
                    candidates=[_safe_song(item) for item in exc.candidates],
                    error_code=exc.code,
                    error_message=exc.message,
                )
                return
        if not isinstance(song, dict):
            raise PipelineError("song_not_found", "The music provider returned no song result.")
        safe_song = _safe_song(song)
        if not safe_song["title"]:
            raise PipelineError("song_not_found", "The music provider returned an invalid song result.")
        source_fp = str(getattr(self.provider, "provider_fingerprint", "netease:unknown"))
        manifest = await self._prepare_runtime()
        target = self._target_parameters(manifest)
        dedup_key = hashlib.sha256(
            f"{_song_identity(safe_song)}|{source_fp}|{target['target_fingerprint']}".encode("utf-8")
        ).hexdigest()
        reusable = self.store.find_dedup_key(dedup_key)
        if reusable and reusable["status"] in {"ready", "delivered"}:
            library_artifact = reusable.get("artifacts", {}).get("output", {}).get("wav")
            try:
                if not library_artifact:
                    raise audio.AudioError("Library output is missing")
                await audio.run_blocking(audio.validate_wav, library_artifact)
                if self._cancelled(job_id):
                    raise asyncio.CancelledError
            except asyncio.CancelledError:
                raise
            except (audio.AudioError, OSError):
                self.store.invalidate_reusable_job(reusable["id"])
        surviving_job_id = self.store.bind_dedup_key(job_id, dedup_key)
        if surviving_job_id != job_id:
            return
        self.store.update(job_id, song=safe_song)
        if self._cancelled(job_id):
            return

        job_root = self._job_work_dir(job_id)
        download_dir = job_root / "download"
        download_dir.mkdir(parents=True, exist_ok=True)

        duration_ms = int(safe_song.get("duration_ms", 0) or 0)
        max_duration = int(self.settings["pipeline"].get("max_duration_seconds", 600))
        if duration_ms and duration_ms > max_duration * 1000:
            raise PipelineError("duration_limit", "The selected song is longer than the configured limit.")

        download_fp = self._stage_fingerprint("download", {"song": _song_identity(safe_song)}, source_fp)
        cached_download = await self._cached(job_id, "download", download_fp, ("source",))
        if cached_download:
            source = cached_download["source"]
        else:
            self.store.update(job_id, status="downloading", stage="downloading", progress="Downloading song")
            part = download_dir / "source.part"
            part.unlink(missing_ok=True)
            raw_source = await asyncio.wait_for(
                self.provider.download(song, part),
                timeout=max(10, int(self.settings["pipeline"].get("stage_timeout_seconds", 900))),
            )
            source = _contained(Path(raw_source), job_root)
            max_bytes = int(self.settings["pipeline"].get("max_download_bytes", 100 * 1024 * 1024))
            source_size = await audio.run_blocking(lambda: source.stat().st_size)
            if self._cancelled(job_id):
                raise asyncio.CancelledError
            if source_size <= 0 or source_size > max_bytes:
                raise PipelineError("download_size", "The downloaded audio is empty or exceeds the configured size limit.")
            stable_source = download_dir / "source.download"
            if source != stable_source:
                await audio.run_blocking(shutil.copyfile, source, stable_source)
                if self._cancelled(job_id):
                    raise asyncio.CancelledError
                source = stable_source
            self.store.record_artifact(job_id, "download", download_fp, {"source": str(source)})

        conversion_fp = self._stage_fingerprint(
            "converting", {"source": await self._content_fingerprint(source, job_id)}, {"ffmpeg": self.settings.get("ffmpeg_path")}
        )
        async def convert_source(attempt_dir: Path) -> dict[str, Path]:
            self.store.update(job_id, status="converting", stage="converting", progress="Converting to WAV")
            target_path = attempt_dir / "source.wav"
            await audio.convert_to_wav(source, target_path, ffmpeg=self.settings.get("ffmpeg_path"), timeout_seconds=max_duration + 300)
            return {"wav": target_path}

        # The initial conversion stage has one output and its fingerprint keys
        # are independent of provider URLs or credentials.
        source_wav_outputs = await self._perform_stage(
            job_id, "converting", conversion_fp, ("wav",), convert_source, audio_outputs=("wav",)
        )
        source_wav = source_wav_outputs["wav"]
        source_info = await audio.run_blocking(audio.validate_wav, source_wav, max_duration=max_duration)
        if self._cancelled(job_id):
            raise asyncio.CancelledError
        if duration_ms and abs(float(source_info["duration"]) - duration_ms / 1000) > max(3.0, duration_ms / 10000):
            raise PipelineError("duration_mismatch", "Downloaded audio duration does not match the selected song.")

        separation_settings = self.settings.get("separation", {})
        model_fingerprints = getattr(self.runtime, "model_fingerprints", {}) or {}
        separation_models = model_fingerprints.get("separation", {}) if isinstance(model_fingerprints, dict) else {}
        rift_models = model_fingerprints.get("rift", {}) if isinstance(model_fingerprints, dict) else {}
        vocals_fp = self._stage_fingerprint(
            "separating",
            {"source": await self._content_fingerprint(source_wav, job_id)},
            {"model": separation_models.get("vocals_model", separation_settings.get("vocals_model"))},
        )
        async def separate_vocals(attempt_dir: Path) -> dict[str, Path]:
            self.store.update(job_id, status="separating", stage="separating", progress="Separating vocals and instrumental")
            result = await self.runtime.separate(source_wav, attempt_dir, "vocals")
            return {"vocals": Path(result["vocals"]), "instrumental": Path(result["instrumental"])}
        vocals = await self._perform_stage(
            job_id, "separating", vocals_fp, ("vocals", "instrumental"), separate_vocals,
            audio_outputs=("vocals", "instrumental"),
        )

        split_fp = self._stage_fingerprint(
            "splitting",
            {"vocals": await self._content_fingerprint(vocals["vocals"], job_id)},
            {"model": separation_models.get("backing_model", separation_settings.get("backing_model"))},
        )
        async def split_backing(attempt_dir: Path) -> dict[str, Path]:
            self.store.update(job_id, status="splitting", stage="splitting", progress="Splitting lead and backing vocals")
            result = await self.runtime.separate(vocals["vocals"], attempt_dir, "backing")
            return {"lead": Path(result["lead"]), "backing": Path(result["backing"])}
        split = await self._perform_stage(
            job_id, "splitting", split_fp, ("lead", "backing"), split_backing,
            audio_outputs=("lead", "backing"), allow_silence=("backing",),
        )
        backing_info = await audio.run_blocking(audio.validate_wav, split["backing"], reject_silence=False)
        if self._cancelled(job_id):
            raise asyncio.CancelledError
        backing_silent = float(backing_info.get("rms", 0.0)) < 0.5

        clean_fp = self._stage_fingerprint(
            "cleaning",
            await self._content_fingerprints({"lead": split["lead"], "backing": split["backing"]}, job_id),
            {"model": separation_models.get("clean_model", separation_settings.get("clean_model"))},
        )
        async def clean_vocals(attempt_dir: Path) -> dict[str, Path]:
            self.store.update(job_id, status="cleaning", stage="cleaning", progress="Removing reverb and echo")
            lead_result = await self.runtime.separate(split["lead"], attempt_dir / "lead", "clean")
            attempt_dir.joinpath("backing").mkdir(parents=True, exist_ok=True)
            if backing_silent:
                backing_dry = attempt_dir / "backing" / "dry.wav"
                await audio.run_blocking(shutil.copyfile, split["backing"], backing_dry)
            else:
                backing_result = await self.runtime.separate(split["backing"], attempt_dir / "backing", "clean")
                backing_dry = Path(backing_result["dry"])
            return {"lead": Path(lead_result["dry"]), "backing": backing_dry}
        clean = await self._perform_stage(
            job_id, "cleaning", clean_fp, ("lead", "backing"), clean_vocals,
            audio_outputs=("lead", "backing"), allow_silence=("backing",) if backing_silent else (),
        )

        rift_settings = self.settings.get("rift", {})
        adaptive_quality = rift_settings.get("adaptive_quality", False)
        voice_outputs = ("lead", "backing", "lead_qa", "backing_qa") if adaptive_quality else ("lead", "backing")
        conversion_fp = self._stage_fingerprint(
            "voice_converting",
            await self._content_fingerprints({"lead": clean["lead"], "backing": clean["backing"]}, job_id),
            {"settings": _fingerprint_sections({"rift": rift_settings})["rift"],
             "models": rift_models, "backing_silent": backing_silent,
             "voice_processing_version": VOICE_PROCESSING_VERSION},
        )
        async def convert_vocals(attempt_dir: Path) -> dict[str, Path]:
            self.store.update(job_id, status="voice_converting", stage="voice_converting", progress="Converting lead and backing voices")
            await self.runtime.ensure_started()
            lead_out = attempt_dir / "lead.wav"
            await self.runtime.convert(clean["lead"], lead_out, "lead")
            automatic_song_shift = 0
            if (
                adaptive_quality
                and rift_settings.get("timbre_slicing", True)
                and rift_settings.get("domain_alignment", True)
                and rift_settings.get("allow_octave_jump", False)
                and not rift_settings.get("song_octave_shift", 0)
                and not rift_settings.get("key_shift", 0)
                and not rift_settings.get("backing_key_shift", 0)
            ):
                automatic_song_shift = await audio.run_blocking(
                    _automatic_song_octave_shift, lead_out.with_suffix(".song-octave.json"),
                )
            backing_out = attempt_dir / "backing.wav"
            if backing_silent:
                await audio.run_blocking(shutil.copyfile, clean["backing"], backing_out)
            else:
                if automatic_song_shift:
                    await self.runtime.convert(
                        clean["backing"], backing_out, "backing", song_octave_shift=automatic_song_shift,
                    )
                else:
                    # Preserve the longstanding three-argument runtime protocol
                    # unless a validated lead decision needs to cross to backing.
                    await self.runtime.convert(clean["backing"], backing_out, "backing")
            outputs = {"lead": lead_out, "backing": backing_out}
            if adaptive_quality:
                outputs["lead_qa"] = lead_out.with_suffix(".qa.json")
                outputs["backing_qa"] = backing_out.with_suffix(".qa.json")
                if backing_silent:
                    await audio.run_blocking(outputs["backing_qa"].write_text, '{"status":"skipped_silent"}', encoding="utf-8")
            return outputs
        converted = await self._perform_stage(
            job_id, "voice_converting", conversion_fp, voice_outputs, convert_vocals,
            audio_outputs=("lead", "backing"), allow_silence=("backing",) if backing_silent else (),
        )

        mix_settings = self.settings.get("mix", {})
        true_peak_ceiling = -1.0 if adaptive_quality else None
        mix_fp = self._stage_fingerprint(
            "mixing",
            await self._content_fingerprints(
                {"instrumental": vocals["instrumental"], "lead": converted["lead"], "backing": converted["backing"]},
                job_id,
            ),
            {"mix": mix_settings, "peak_limit": 0.98, "true_peak_ceiling": true_peak_ceiling,
             "pcm_mix_version": 2},
        )
        async def mix_tracks(attempt_dir: Path) -> dict[str, Path]:
            self.store.update(job_id, status="mixing", stage="mixing", progress="Mixing aligned tracks")
            output = attempt_dir / "mixed.wav"
            pre_master = attempt_dir / "pre-master.wav" if true_peak_ceiling is not None else output
            await audio.run_blocking(
                audio.mix_wavs,
                vocals["instrumental"], converted["lead"], converted["backing"], pre_master,
                peak_limit=0.98,
                gains=(float(mix_settings.get("lead_gain", 1.0)), float(mix_settings.get("backing_gain", 0.7))),
            )
            if true_peak_ceiling is not None:
                try:
                    await audio.limit_master_true_peak(pre_master, output, ffmpeg=self.settings.get("ffmpeg_path"),
                                                       target_dbtp=true_peak_ceiling)
                except audio.AudioError as exc:
                    raise PipelineError("mastering_failed", str(exc)) from exc
                finally:
                    pre_master.unlink(missing_ok=True)
            return {"wav": output}
        mixed = await self._perform_stage(
            job_id, "mixing", mix_fp, ("wav",), mix_tracks, audio_outputs=("wav",)
        )
        if self._cancelled(job_id):
            return

        output_dir = Path(self.settings["runtime_dir"]) / "artifacts"
        file_path = output_dir / f"{job_id}.wav"
        await audio.run_blocking(_atomic_copy, mixed["wav"], file_path)
        await audio.run_blocking(audio.validate_wav, file_path)
        self.store.record_artifact(job_id, "output", target["target_fingerprint"], {"wav": str(file_path)})
        self.store.update(job_id, status="ready", stage="ready", progress="Completed", error_code="", error_message="")
