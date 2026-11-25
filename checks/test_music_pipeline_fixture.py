from __future__ import annotations

import asyncio
import shutil
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from music_runtime.audio import validate_wav  # noqa: E402
from music_runtime.job_store import JobStore  # noqa: E402
from music_runtime.pipeline import MusicPipeline  # noqa: E402


def _write_wav(path: Path, amplitude: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = 4410
    payload = struct.pack("<h", amplitude) * frames
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(payload)
    return path


class FixtureProvider:
    provider_fingerprint = "fixture-provider-v1"

    def __init__(self):
        self.download_calls = 0
        self.resolve_calls = 0

    async def resolve(self, query: str) -> dict:
        self.resolve_calls += 1
        return {"id": "12345", "title": "Fixture Song", "artist": "Fixture Artist", "duration_ms": 200}

    async def download(self, song: dict, destination: Path) -> Path:
        self.download_calls += 1
        return _write_wav(destination)

    async def close(self) -> None:
        return None


class FixtureRuntime:
    def __init__(self):
        self.model_fingerprints = {
            "separation": {"vocals_model": "v1", "backing_model": "v2", "clean_model": "v3"},
            "rift": {"lead": "r1", "backing": "r1"},
        }
        self.separate_calls: list[str] = []
        self.convert_calls: list[str] = []

    def target_model_fingerprints(self) -> dict:
        return self.model_fingerprints

    async def check_ready(self) -> dict:
        return {"ready": True, "components": {}}

    async def separate(self, source: Path, output_dir: Path, kind: str) -> dict[str, Path]:
        self.separate_calls.append(kind)
        output_dir.mkdir(parents=True, exist_ok=True)
        if kind == "vocals":
            return {
                "vocals": _write_wav(output_dir / "vocals.wav", 7000),
                "instrumental": _write_wav(output_dir / "instrumental.wav", 6000),
            }
        if kind == "backing":
            return {
                "lead": _write_wav(output_dir / "lead.wav", 6500),
                "backing": _write_wav(output_dir / "backing.wav", 3000),
            }
        dry = _write_wav(output_dir / "dry.wav", 5000)
        return {"dry": dry}

    async def ensure_started(self) -> None:
        return None

    async def convert(self, source: Path, destination: Path, role: str) -> Path:
        self.convert_calls.append(role)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    async def release_workers(self) -> None:
        return None

    async def close(self) -> None:
        return None


class AutomaticDecisionRuntime(FixtureRuntime):
    def __init__(self):
        super().__init__()
        self.song_shift_calls: list[tuple[str, int]] = []

    async def convert(self, source: Path, destination: Path, role: str, *, song_octave_shift: int = 0) -> Path:
        self.convert_calls.append(role)
        self.song_shift_calls.append((role, song_octave_shift))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.with_suffix(".qa.json").write_text("{}", encoding="utf-8")
        if role == "lead":
            destination.with_suffix(".song-octave.json").write_text(
                '{"song_octave_shift":12,"song_octave_source":"automatic"}', encoding="utf-8",
            )
        return destination


def _settings(root: Path) -> dict:
    return {
        "plugin_dir": root,
        "runtime_dir": root,
        "pipeline": {
            "enabled": True,
            "max_queue": 5,
            "max_duration_seconds": 600,
            "max_download_bytes": 1024 * 1024,
            "stage_timeout_seconds": 30,
        },
        "netease": {"base_url": "http://127.0.0.1:13306", "profile": "fixture"},
        "separation": {"vocals_model": "vocals", "backing_model": "backing", "clean_model": "clean"},
        "rift": {"checkpoint": "", "speaker": "fixture", "backing_checkpoint": "", "backing_speaker": "fixture"},
        "mix": {"lead_gain": 1.0, "backing_gain": 0.7},
        "runtime": {"device": "cpu"},
        "ffmpeg_path": None,
    }


async def _wait_for_terminal(store: JobStore, job_id: str) -> dict:
    for _ in range(500):
        current = store.get_job(job_id)
        if current and current["status"] in {"ready", "failed", "selection_required"}:
            return current
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


class MusicPipelineFixtureTests(unittest.IsolatedAsyncioTestCase):
    async def test_automatic_lead_octave_is_handed_to_backing_without_formant_alignment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="music-runtime-pipeline-octave-") as folder:
            root = Path(folder)
            settings = _settings(root)
            settings["rift"].update({
                "adaptive_quality": True,
                "timbre_slicing": True,
                "domain_alignment": True,
                "domain_formants": False,
                "allow_octave_jump": True,
                "song_octave_shift": 0,
                "key_shift": 0,
                "backing_key_shift": 0,
            })
            store = JobStore(root / "jobs.sqlite")
            runtime = AutomaticDecisionRuntime()
            pipeline = MusicPipeline(
                settings, store=store, provider=FixtureProvider(), runtime=runtime,
            )
            accepted = await pipeline.submit("b" * 32, "Fixture Song")

            async def copy_master(source: Path, destination: Path, **_kwargs) -> None:
                shutil.copyfile(source, destination)

            with patch("music_runtime.pipeline.audio.limit_master_true_peak", side_effect=copy_master):
                completed = await _wait_for_terminal(store, accepted["id"])
            self.assertEqual(completed["status"], "ready", completed)
            self.assertEqual(runtime.song_shift_calls, [("lead", 0), ("backing", 12)])
            await pipeline.close()

    async def test_produces_durable_wav_and_resumes_from_saved_stage_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="music-runtime-pipeline-") as folder:
            root = Path(folder)
            settings = _settings(root)
            store = JobStore(root / "jobs.sqlite")
            provider = FixtureProvider()
            runtime = FixtureRuntime()
            first = MusicPipeline(settings, store=store, provider=provider, runtime=runtime)

            accepted = await first.submit("a" * 32, "Fixture Song")
            completed = await _wait_for_terminal(store, accepted["id"])
            self.assertEqual(completed["status"], "ready", completed)
            output = Path(completed["artifacts"]["output"]["wav"])
            self.assertEqual(output.parent, (root / "artifacts").resolve())
            self.assertTrue(output.is_file())
            self.assertGreater(validate_wav(output)["rms"], 0.5)
            self.assertGreater(len(runtime.separate_calls), 0)
            self.assertGreater(len(runtime.convert_calls), 0)
            self.assertFalse((root / "music_library.json").exists())

            # Model a service crash in a previously completed stage. Startup
            # must requeue it and validate/reuse the durable stage checkpoints.
            store.update(accepted["id"], status="mixing", stage="mixing", progress="Interrupted during mix")
            await first.close()

            resumed_provider = FixtureProvider()
            resumed_runtime = FixtureRuntime()
            resumed = MusicPipeline(settings, store=store, provider=resumed_provider, runtime=resumed_runtime)
            await resumed.ensure_started()
            final = await _wait_for_terminal(store, accepted["id"])
            self.assertEqual(final["status"], "ready", final)
            self.assertEqual(final["id"], accepted["id"])
            self.assertEqual(resumed_provider.download_calls, 0)
            self.assertEqual(resumed_runtime.separate_calls, [])
            self.assertEqual(resumed_runtime.convert_calls, [])
            self.assertTrue(Path(final["artifacts"]["output"]["wav"]).is_file())
            await resumed.close()


if __name__ == "__main__":
    unittest.main()
