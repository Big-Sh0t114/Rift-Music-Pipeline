"""Versioned speaker profiles and shared RIFT feature extraction for profiling."""

import hashlib
import json
import math
import re
from pathlib import Path


BANDS = 18
RATE = 44100
PITCH_HOP = 441
MAX_PROFILE_BYTES = 48 * 1024


def validate_profile(profile, speaker):
    fields_v1 = {"version", "speaker", "encoder_sha256", "pitch_range", "ltas_db", "cvec_mean", "cvec_std"}
    fields_v2 = fields_v1 | {"pitch_tails", "formant_anchor_hz", "formant_frames", "formant_log_mad"}
    if not isinstance(profile, dict):
        raise ValueError("Invalid RIFT speaker profile schema")
    version = profile.get("version")
    expected = fields_v1 if type(version) is int and version == 1 else fields_v2 if type(version) is int and version == 2 else None
    if expected is None or set(profile) != expected:
        raise ValueError("Invalid RIFT speaker profile schema")
    if not speaker or profile["speaker"] != speaker:
        raise ValueError("The RIFT speaker profile does not match the configured speaker or version")
    digest = profile["encoder_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid RIFT speaker profile encoder fingerprint")

    def vector(name, minimum, maximum, low, high):
        values = profile[name]
        if not isinstance(values, list) or not minimum <= len(values) <= maximum or any(
            type(value) not in (float, int) or not low <= value <= high or not math.isfinite(value) for value in values
        ):
            raise ValueError(f"Invalid {name} in the RIFT speaker profile")
        return values

    pitches = vector("pitch_range", 3, 3, 0, 127)
    if pitches != sorted(pitches) or pitches[0] == pitches[-1]:
        raise ValueError("Invalid pitch percentiles in the RIFT speaker profile")
    vector("ltas_db", BANDS, BANDS, -120, 120)
    mean = vector("cvec_mean", 1, 1024, -100, 100)
    vector("cvec_std", len(mean), len(mean), 0, 100)

    if version == 2:
        tails = vector("pitch_tails", 2, 2, 0, 127)
        if tails[0] > pitches[0] or tails[1] < pitches[2]:
            raise ValueError("Invalid pitch tails in the RIFT speaker profile")
        frames = profile["formant_frames"]
        if type(frames) is not int or not 0 <= frames <= 1_000_000_000:
            raise ValueError("Invalid formant frame count in the RIFT speaker profile")
        anchor = profile["formant_anchor_hz"]
        mad = profile["formant_log_mad"]
        if anchor is not None and (
            type(anchor) not in (int, float) or not math.isfinite(anchor) or not 250 <= anchor <= 3500
        ):
            raise ValueError("Invalid formant anchor in the RIFT speaker profile")
        if mad is not None and (
            type(mad) not in (int, float) or not math.isfinite(mad) or not 0 <= mad <= 2
        ):
            raise ValueError("Invalid formant dispersion in the RIFT speaker profile")
        if frames == 0 and (anchor is not None or mad is not None):
            raise ValueError("A profile without formant frames cannot contain formant statistics")
        if frames > 0 and mad is None:
            raise ValueError("Formant frames require a dispersion estimate")


def read_profile(path, speaker):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_PROFILE_BYTES + 1)
    if len(raw) > MAX_PROFILE_BYTES:
        raise ValueError("RIFT speaker profile exceeds its byte limit")
    profile = json.loads(raw)
    validate_profile(profile, speaker)
    return profile, hashlib.sha256(raw).hexdigest()


def encoder_fingerprint(source):
    folder = Path(source) / "pretrained" / "content-vec-best"
    digest = hashlib.sha256()
    weights = "model.safetensors" if (folder / "model.safetensors").is_file() else "pytorch_model.bin"
    for name in ("config.json", weights):
        digest.update(name.encode("ascii"))
        with (folder / name).open("rb") as handle:
            digest.update(hashlib.file_digest(handle, "sha256").digest())
    return digest.hexdigest()


def rmvpe_pitch(samples, model, torch, device):
    import numpy as np
    from scipy.signal import resample_poly

    resampled = resample_poly(samples, 160, 441)
    # Centered RMVPE STFT requires enough samples for reflection padding.
    resampled = np.pad(resampled, (0, max(0, 4096 - len(resampled))))
    audio = torch.as_tensor(resampled, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        hidden = model.mel2hidden(model.mel_extractor(audio, center=True))
        f0 = model.decode(hidden, thred=0.3).reshape(-1)
        confidence = hidden.amax(dim=2).cpu().numpy().reshape(-1)
    count = len(samples) // PITCH_HOP + 1
    return f0[:count], confidence[:count]
