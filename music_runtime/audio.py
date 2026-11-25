"""Bounded FFmpeg conversion and PCM WAV validation/mixing helpers."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import struct
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any


class AudioError(ValueError):
    """Audio is malformed, silent, misaligned, or cannot be converted safely."""


async def run_blocking(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run file/PCM work outside the event loop and finish safely on cancel.

    Cancelling ``asyncio.to_thread`` does not stop its worker thread. Waiting for
    the worker before propagating cancellation prevents temporary files from
    being removed while a copy or validation is still using them.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            await worker
        except Exception:
            pass
        raise


def resolve_ffmpeg(explicit: Path | str | None = None) -> str:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise AudioError("FFmpeg is unavailable; configure a static or managed FFmpeg executable")


def wav_info(path: Path | str) -> dict[str, int | float]:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.getnframes()
            if wav.getcomptype() != "NONE":
                raise AudioError("WAV must use uncompressed PCM")
            if channels not in {1, 2} or width not in {1, 2, 3, 4} or rate < 8000 or frames <= 0:
                raise AudioError("WAV has unsupported format or empty audio")
            return {
                "channels": channels,
                "sample_width": width,
                "sample_rate": rate,
                "frames": frames,
                "duration": frames / rate,
            }
    except AudioError:
        raise
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioError("File is not a readable PCM WAV") from exc


def validate_wav(
    path: Path | str,
    *,
    min_duration: float = 0.08,
    max_duration: float | None = None,
    reject_silence: bool = True,
    minimum_rms: float = 0.5,
) -> dict[str, int | float]:
    """Validate PCM header and inspect bounded chunks for audibility.

    PCM has no floating-point samples, so non-finite values cannot appear in a
    valid WAV; malformed headers and truncated frames are rejected by `wave`.
    """
    path = Path(path)
    info = wav_info(path)
    duration = float(info["duration"])
    if duration < min_duration:
        raise AudioError("Audio duration is too short")
    if max_duration is not None and duration > max_duration:
        raise AudioError("Audio duration exceeds the configured limit")
    width = int(info["sample_width"])
    total_sq = 0.0
    sample_count = 0
    actual_bytes = 0
    try:
        with wave.open(str(path), "rb") as wav:
            while True:
                raw = wav.readframes(32768)
                if not raw:
                    break
                actual_bytes += len(raw)
                samples = _decode_samples(raw, width)
                total_sq += sum(float(sample) * float(sample) for sample in samples)
                sample_count += len(samples)
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioError("WAV ended before all declared frames were readable") from exc
    if sample_count == 0:
        raise AudioError("Audio has no sample data")
    expected_bytes = int(info["frames"]) * int(info["channels"]) * width
    if actual_bytes != expected_bytes or actual_bytes % (int(info["channels"]) * width):
        raise AudioError("WAV sample data is truncated or does not match its frame count")
    rms = math.sqrt(total_sq / sample_count)
    if reject_silence and rms < minimum_rms:
        raise AudioError("Audio is silent")
    return {**info, "rms": rms}


def _decode_samples(raw: bytes, width: int) -> list[int]:
    if width == 1:
        return [sample - 128 for sample in raw]
    if width == 2:
        count = len(raw) // 2
        return list(struct.unpack(f"<{count}h", raw[: count * 2])) if count else []
    if width == 3:
        data = raw[: len(raw) - len(raw) % 3]
        result = []
        for i in range(0, len(data), 3):
            value = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16)
            result.append(value - (1 << 24) if value & (1 << 23) else value)
        return result
    if width == 4:
        count = len(raw) // 4
        values = struct.unpack(f"<{count}i", raw[: count * 4]) if count else ()
        return list(values)
    raise AudioError("Unsupported PCM sample width")


async def convert_to_wav(
    source: Path | str,
    destination: Path | str,
    *,
    ffmpeg: Path | str | None = None,
    timeout_seconds: float = 900,
) -> Path:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".partial")
    try:
        try:
            await run_blocking(wav_info, source)
            if source != destination:
                await run_blocking(shutil.copyfile, source, tmp)
                await run_blocking(validate_wav, tmp)
                os.replace(tmp, destination)
                return destination
        except AudioError:
            pass

        executable = resolve_ffmpeg(ffmpeg)
        process = await asyncio.create_subprocess_exec(
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(tmp),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=max(1, timeout_seconds))
        except (asyncio.TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.wait()
            raise
        if process.returncode != 0:
            raise AudioError("FFmpeg could not decode the selected audio")
        await run_blocking(validate_wav, tmp)
        os.replace(tmp, destination)
        return destination
    except asyncio.CancelledError:
        raise
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


async def limit_master_true_peak(
    source: Path, output: Path, *, ffmpeg: Path | str | None, target_dbtp: float,
) -> None:
    """Measure intersample peaks and apply one global gain without reshaping dynamics."""
    executable = resolve_ffmpeg(ffmpeg)
    pattern = re.compile(rb"True peak:\s*Peak:\s*(-?(?:\d+(?:\.\d+)?|inf))\s+dBFS")

    async def run(path, *args):
        process = await asyncio.create_subprocess_exec(
            executable, "-hide_banner", "-nostats", "-nostdin", "-i", str(path), *args,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await process.communicate()
        except asyncio.CancelledError:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            raise
        if process.returncode:
            raise AudioError("FFmpeg could not master the mixed WAV")
        return stderr

    async def measure(path):
        summary = await run(path, "-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "-")
        match = pattern.search(summary)
        if not match:
            raise AudioError("FFmpeg did not return a true-peak measurement")
        return float(match[1])

    input_peak = await measure(source)
    # FFmpeg reports to one decimal place. Leave margin for rounding and PCM16 quantization.
    if input_peak < target_dbtp:
        os.replace(source, output)
        return
    partial = output.with_suffix(".partial.wav")
    try:
        attenuation = target_dbtp - 0.2 - input_peak
        await run(source, "-af", f"volume={attenuation:.4f}dB", "-c:a", "pcm_s16le", "-y", str(partial))
        if await measure(partial) >= target_dbtp:
            raise AudioError("The mastered WAV exceeds the true-peak ceiling")
        original, mastered = await run_blocking(lambda: (wav_info(source), wav_info(partial)))
        if mastered != original:
            raise AudioError("Mastering changed the mix format or timeline")
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)


def content_fingerprint(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def settings_fingerprint(value: Any) -> str:
    """Canonical settings hash that omits likely credentials and tokens."""

    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                str(key): clean(val)
                for key, val in sorted(item.items(), key=lambda pair: str(pair[0]))
                if not any(secret in str(key).casefold() for secret in ("password", "token", "secret", "cookie", "credential", "api_key"))
            }
        if isinstance(item, (tuple, list)):
            return [clean(part) for part in item]
        if isinstance(item, Path):
            return str(item)
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        return str(item)

    payload = json.dumps(clean(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mix_wavs(
    instrument: Path | str,
    lead: Path | str,
    backing: Path | str,
    output: Path | str,
    *,
    peak_limit: float = 0.98,
    max_alignment_drift_seconds: float = 0.05,
    gains: tuple[float, float] = (1.0, 1.0),
) -> Path:
    """Mix aligned PCM tracks with a single global peak gain.

    Tracks must share a sample rate. Width/channel conversions are permitted,
    while frame-count drift is only repaired by trailing zero padding. No
    audio is trimmed or time-shifted.
    """
    paths = [Path(instrument), Path(lead), Path(backing)]
    infos = [validate_wav(path, reject_silence=False) for path in paths]
    if float(infos[0].get("rms", 0.0)) < 0.5 or float(infos[1].get("rms", 0.0)) < 0.5:
        raise AudioError("Instrumental and lead tracks must contain audible audio")
    sample_rate = int(infos[0]["sample_rate"])
    channel_count = int(infos[0]["channels"])
    converted: list[bytes] = []
    frame_counts: list[int] = []
    try:
        import audioop  # Python 3.11/3.12; output is PCM16 for transport.
    except ImportError as exc:
        raise AudioError("PCM16 mixing support is unavailable") from exc

    for path, info in zip(paths, infos, strict=True):
        if int(info["sample_rate"]) != sample_rate:
            raise AudioError("Separated vocal tracks do not share a sample rate")
        with wave.open(str(path), "rb") as wav:
            pcm = wav.readframes(wav.getnframes())
        width = int(info["sample_width"])
        if width != 2:
            pcm = audioop.lin2lin(pcm, width, 2)
        channels = int(info["channels"])
        if channels != channel_count:
            if channels == 1 and channel_count == 2:
                pcm = audioop.tostereo(pcm, 2, 1.0, 1.0)
            elif channels == 2 and channel_count == 1:
                pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
            else:
                raise AudioError("Unsupported channel layout")
        converted.append(pcm)
        frame_counts.append(int(info["frames"]))

    target_frames = max(frame_counts)
    allowed = max(1, round(max_alignment_drift_seconds * sample_rate))
    if target_frames - min(frame_counts) > allowed:
        raise AudioError("Vocal conversion changed the song timeline too much")
    frame_bytes = 2 * channel_count
    normalized = [
        pcm + b"\x00" * ((target_frames - frames) * frame_bytes)
        for pcm, frames in zip(converted, frame_counts, strict=True)
    ]
    if len(gains) != 2:
        raise AudioError("Mix gain configuration is invalid")
    mix_gains = (1.0, max(0.0, min(float(gains[0]), 4.0)), max(0.0, min(float(gains[1]), 4.0)))
    # Allow the maximum sum of nine full-scale inputs while accumulating in
    # PCM32. Restore headroom after measuring the whole mix, before quantization.
    headroom = 16.0
    scaled_gains = tuple(gain / headroom for gain in mix_gains)
    chunk_frames = 32768

    def sum_pcm(start: int) -> bytes:
        start_byte = start * frame_bytes
        end_byte = min(target_frames, start + chunk_frames) * frame_bytes
        tracks = [
            audioop.mul(audioop.lin2lin(pcm[start_byte:end_byte], 2, 4), 4, gain)
            for pcm, gain in zip(normalized, scaled_gains, strict=True)
        ]
        return audioop.add(audioop.add(tracks[0], tracks[1], 4), tracks[2], 4)

    peak = 0
    for start in range(0, target_frames, chunk_frames):
        peak = max(peak, audioop.max(sum_pcm(start), 4))

    limit = min(max(float(peak_limit), 0.1), 0.999)
    gain = min(headroom, limit * (32767 << 16) / peak) if peak else headroom
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    try:
        with wave.open(str(partial), "wb") as wav_out:
            wav_out.setnchannels(channel_count)
            wav_out.setsampwidth(2)
            wav_out.setframerate(sample_rate)
            for start in range(0, target_frames, chunk_frames):
                scaled = audioop.mul(sum_pcm(start), 4, gain)
                rounded = audioop.bias(scaled, 4, 1 << 15)
                wav_out.writeframesraw(audioop.lin2lin(rounded, 4, 2))
        validate_wav(partial)
        os.replace(partial, output)
        return output
    finally:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
