"""Pinned optional models for the shadow-only singer-register probe."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen


OCTAVE_PROBE_SCHEMA_VERSION = 1
OCTAVE_PROBE_IMPLEMENTATION = "rift_octave_probe_shadow_v1"
MODELS = {
    "audioset-yamnet-1.onnx": (
        "https://essentia.upf.edu/models/audio-event-recognition/yamnet/audioset-yamnet-1.onnx",
        14934724,
        "9bc15ac91426e431527196ee6663de78dffcc7db53ac002d5afbda61429b456f",
    ),
    "gender-audioset-yamnet-1.onnx": (
        "https://essentia.upf.edu/models/classification-heads/gender/gender-audioset-yamnet-1.onnx",
        411690,
        "56bbb0d21c37f9db67ffece8360ef8137b4d9f33802783a07aa8ec3563542016",
    ),
    "voice_instrumental-audioset-yamnet-1.onnx": (
        "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-audioset-yamnet-1.onnx",
        411715,
        "f619b35a95a44d8c3ec27a205d4c93ec988401fca443ff552ed007b3f1c018a0",
    ),
}


def octave_probe_assets_state(folder: Path, *, verify: bool = False) -> dict[str, object]:
    """Return a compact, content-addressed state without raising on optional assets."""
    folder = Path(folder)
    try:
        hashes: list[str] = []
        for name, (_, size, digest) in MODELS.items():
            path = folder / name
            stat = path.stat()
            if not path.is_file() or stat.st_size != size:
                return {"state": "corrupt", "fingerprint": _state_fingerprint("corrupt", name)}
            actual = None
            if verify:
                with path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != digest:
                    return {"state": "corrupt", "fingerprint": _state_fingerprint("corrupt", name)}
            hashes.append(actual or digest)
    except FileNotFoundError:
        return {"state": "missing", "fingerprint": _state_fingerprint("missing")}
    except OSError:
        return {"state": "unavailable", "fingerprint": _state_fingerprint("unavailable")}
    digest = hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()
    return {"state": "prepared", "fingerprint": digest}


def _state_fingerprint(state: str, detail: str = "") -> str:
    value = f"{OCTAVE_PROBE_IMPLEMENTATION}|{OCTAVE_PROBE_SCHEMA_VERSION}|{state}|{detail}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def octave_probe_assets_ready(folder: Path, *, verify: bool = False) -> bool:
    return octave_probe_assets_state(folder, verify=verify)["state"] == "prepared"


def prepare_octave_probe_assets(folder: Path) -> Path:
    """Download the optional pinned ONNX bundle through a staged directory."""
    folder = Path(folder)
    state = octave_probe_assets_state(folder, verify=True)
    if state["state"] == "prepared":
        return folder
    if folder.exists():
        raise RuntimeError(
            "The existing octave probe model directory failed validation. Move it aside before preparing again."
        )
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".octave-probe-prep-", dir=folder.parent) as temporary:
        stage = Path(temporary) / "models"
        stage.mkdir()
        for name, (url, size, expected_digest) in MODELS.items():
            target = stage / name
            digest = hashlib.sha256()
            remaining = size
            with urlopen(url, timeout=40) as source, target.open("xb") as output:
                while chunk := source.read(min(1024 * 1024, remaining + 1)):
                    if len(chunk) > remaining:
                        raise RuntimeError("An octave probe model exceeded its pinned byte limit.")
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if remaining or digest.hexdigest() != expected_digest:
                raise RuntimeError("An octave probe model failed its pinned size or SHA-256 check.")
        if not octave_probe_assets_ready(stage, verify=True):
            raise RuntimeError("The octave probe model installation failed validation.")
        try:
            os.rename(stage, folder)
        except OSError:
            if not octave_probe_assets_ready(folder, verify=True):
                raise
    return folder


__all__ = [
    "MODELS", "OCTAVE_PROBE_IMPLEMENTATION", "OCTAVE_PROBE_SCHEMA_VERSION",
    "octave_probe_assets_ready", "octave_probe_assets_state", "prepare_octave_probe_assets",
]
