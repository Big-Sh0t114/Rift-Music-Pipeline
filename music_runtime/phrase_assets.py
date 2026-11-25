"""Pinned Silero and WeSpeaker models for the isolated RIFT worker."""

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen


MODELS = {
    "silero_vad.onnx": (
        "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx",
        2327524, "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f",
    ),
    "wespeaker.onnx": (
        "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34/resolve/ff1ac5bca8ef11e90662b879aa923979e0bd277b/voxceleb_resnet34.onnx?download=true",
        26534127, "9fea6516d7ad6bf0a76c7689f5a49b65d330fad6dde96c91bb4435ffbfe056a1",
    ),
}


def phrase_assets_ready(folder, *, verify=False):
    try:
        if not (folder / "packages/onnxruntime/__init__.py").is_file():
            return False
        for name, (_, size, digest) in MODELS.items():
            path = folder / name
            if path.stat().st_size != size:
                return False
            if verify:
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                        return False
        return True
    except OSError:
        return False


def prepare_phrase_assets(folder, python):
    if phrase_assets_ready(folder, verify=True):
        return folder
    if folder.exists():
        raise RuntimeError("The existing phrase model directory failed validation. Move it aside before preparing again.")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to prepare the phrase models.")
    lock = Path(__file__).parent / "runtime_specs/phrases/requirements.lock.txt"
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phrase-prep-", dir=folder.parent) as temp:
        stage = Path(temp) / "install"
        stage.mkdir()
        for name, (url, size, _) in MODELS.items():
            with urlopen(url, timeout=40) as source, (stage / name).open("xb") as target:
                remaining = size
                while chunk := source.read(min(1024 * 1024, remaining + 1)):
                    if len(chunk) > remaining:
                        raise RuntimeError("A phrase model exceeded its pinned byte limit.")
                    target.write(chunk)
                    remaining -= len(chunk)
        command = [uv, "pip", "install", "--python", str(python), "--target", str(stage / "packages"),
                   "--no-deps", "--require-hashes", "-r", str(lock)]
        if subprocess.run(command, check=False, timeout=240).returncode or not phrase_assets_ready(stage, verify=True):
            raise RuntimeError("The phrase model installation failed validation.")
        try:
            os.rename(stage, folder)
        except OSError:
            if not phrase_assets_ready(folder, verify=True):
                raise
    return folder
