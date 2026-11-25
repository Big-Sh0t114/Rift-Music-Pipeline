"""Bounded FFmpeg rubberband pitch retuning for RIFT domain alignment."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetuneResult:
    samples: np.ndarray
    input_samples: int
    output_samples: int
    drift_samples: int


class RetuneError(RuntimeError):
    """The requested FFmpeg pitch retune could not be trusted."""


def retune_pitch(samples, semitones, ffmpeg, *, sample_rate=44100, timeout_seconds=120):
    """Retune mono float32 audio without changing its published sample count.

    Rubber Band can return a few milliseconds of length drift. Small drift is
    trimmed or zero padded at the tail; larger drift fails closed so callers
    can rerender without a virtual lift.
    """
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(audio) or not np.isfinite(audio).all():
        raise RetuneError("Retune input must contain finite mono samples.")
    if not isinstance(ffmpeg, str) or not ffmpeg.strip():
        raise RetuneError("No FFmpeg executable was configured for virtual pitch retuning.")
    if type(sample_rate) is not int or sample_rate < 8000 or sample_rate > 192000:
        raise RetuneError("Retune sample rate is invalid.")
    if type(semitones) not in (int, float) or not np.isfinite(semitones) or abs(semitones) > 24:
        raise RetuneError("Retune semitone amount is invalid.")
    if type(timeout_seconds) not in (int, float) or not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise RetuneError("Retune timeout is invalid.")

    pitch_ratio = 2.0 ** (float(semitones) / 12.0)
    audio_filter = f"rubberband=pitch={pitch_ratio:.12g}:formant=preserved:pitchq=quality:transients=smooth"
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-af", audio_filter,
        "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(sample_rate), "pipe:1",
    ]
    try:
        result = subprocess.run(command, input=audio.tobytes(), capture_output=True,
                                timeout=float(timeout_seconds), check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RetuneError(f"FFmpeg pitch retuning failed: {exc}") from exc
    if result.returncode:
        error = result.stderr.decode("utf-8", errors="replace")[-1200:]
        raise RetuneError(f"FFmpeg pitch retuning exited with status {result.returncode}: {error}")
    if len(result.stdout) % np.dtype(np.float32).itemsize:
        raise RetuneError("FFmpeg returned a partial float32 sample.")
    retuned = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
    if not len(retuned) or not np.isfinite(retuned).all():
        raise RetuneError("FFmpeg returned empty or invalid samples.")

    drift = len(retuned) - len(audio)
    allowed_drift = max(sample_rate // 50, int(round(len(audio) * 0.01)))
    if abs(drift) > allowed_drift:
        raise RetuneError(f"FFmpeg output length drifted by {drift} samples.")
    if drift > 0:
        retuned = retuned[:len(audio)]
    elif drift < 0:
        retuned = np.pad(retuned, (0, -drift))
    return RetuneResult(retuned, len(audio), len(retuned) + drift, drift)
