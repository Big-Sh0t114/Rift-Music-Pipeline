"""Persistent audio-separator process with semantic output mapping."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from importlib import resources
from pathlib import Path
from typing import Any


MAX_REQUEST_BYTES = 64 * 1024
MAX_AUDIO_BYTES = 512 * 1024 * 1024
BECRUILY_VOCALS_MODEL = "mel_band_roformer_vocals_becruily.ckpt"
BECRUILY_VOCALS_SHA256 = "a05961310cc55fbb901290c2e8be02682942f73522b6ac76bf2ec11e347ed95a"
DEECHO_DEREVERB_MODEL = "UVR-DeEcho-DeReverb.pth"
DEECHO_DEREVERB_SHA256 = "e644028ec82865dc0fe082bc6fea85a43f7c71cfe375caee2da2d154aa661ee7"
VOCAL_STEMS = {"vocal", "vocals", "lead vocal", "lead vocals"}
INSTRUMENTAL_STEMS = {"instrumental", "karaoke", "no vocals", "no vocal", "accompaniment"}
DRY_STEMS = {"dry", "clean", "no reverb", "no echo", "no echo reverb", "dereverb", "deecho"}
_MODEL_LOADERS = {
    BECRUILY_VOCALS_MODEL: (
        "MDXC",
        "Roformer Model: MelBand Roformer | Vocals by becruily",
        "config_mel_band_roformer_vocals_becruily.yaml",
    ),
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt": (
        "MDXC",
        "Roformer Model: Mel-Roformer-Karaoke-Aufr33-Viperx",
        "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml",
    ),
    DEECHO_DEREVERB_MODEL: (
        "VR",
        "VR Arch Single Model v5: UVR-DeEcho-DeReverb by FoxJoy",
        None,
    ),
}
_MODEL_DATA_LIMIT = 8 * 1024 * 1024


class _StemMappingError(ValueError):
    """The prepared model's declared stems do not fit the requested stage."""


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("The requested audio path is outside the job directory.") from None
    return resolved


class _Audio:
    def __init__(self, runtime_dir: Path):
        self.runtime_dir = runtime_dir.resolve(strict=True)
        self.output_dir: Path | None = None
        self.model_cache: Path | None = None
        self.separator = None
        self.model = ""
        self.model_identity: tuple[tuple[str, int, int, str], ...] | None = None
        self.model_sha256 = ""
        self._verified_model_assets: dict[str, Path] = {}
        self.device = "auto"
        self.ffmpeg_path: Path | None = None
        self.protocol_stdout = sys.stdout
        sys.stdout = sys.stderr

    def start(self, settings: dict[str, Any]) -> None:
        if set(settings) != {"model_cache", "device", "ffmpeg_path"} or settings["device"] not in {"auto", "cpu", "cuda"}:
            raise ValueError("Invalid separator startup settings.")
        self.device = str(settings["device"])
        if self.device == "cpu":
            # The child has not imported Torch yet; hiding CUDA here keeps
            # both Torch and ONNX Runtime on CPU for this process lifetime.
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        cache = Path(str(settings["model_cache"])).expanduser().resolve(strict=False)
        cache.mkdir(parents=True, exist_ok=True)
        try:
            cache.relative_to(self.runtime_dir)
        except ValueError:
            raise ValueError("The separator model cache is outside the managed runtime.") from None
        self.model_cache = cache
        ffmpeg = Path(str(settings["ffmpeg_path"])).expanduser().resolve(strict=True)
        if not ffmpeg.is_file():
            raise ValueError("The managed FFmpeg executable is missing.")
        self.ffmpeg_path = ffmpeg
        # audio-separator resolves FFmpeg by executable name. Scope PATH to
        # this worker process so Core and other plugins are unaffected.
        os.environ["PATH"] = str(ffmpeg.parent) + os.pathsep + os.environ.get("PATH", "")

    def _prepared_model(self, model: str) -> tuple[Path, tuple[tuple[str, int, int, str], ...], str]:
        if self.model_cache is None:
            raise RuntimeError("Separator model storage is not initialized.")
        manifest_path = self.model_cache / "manifest.json"
        try:
            if manifest_path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("Prepared model manifest is too large.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("The requested model was not prepared.") from None
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("audio_separator_version") != "0.47.1"
        ):
            raise ValueError("The requested model was not prepared.")
        models = manifest.get("models")
        entry = models.get(model) if isinstance(models, dict) else None
        files = entry.get("files") if isinstance(entry, dict) else None
        if not isinstance(files, list) or not files or len(files) > 256:
            raise ValueError("The requested model was not prepared.")
        model_path: Path | None = None
        model_digest = ""
        identity_parts: list[tuple[str, int, int, str]] = []
        verified_assets: dict[str, Path] = {}
        for item in files:
            if not isinstance(item, dict) or item.get("verified") is not True:
                raise ValueError("The requested model was not prepared.")
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
                raise ValueError("The requested model was not prepared.")
            path = (self.model_cache / name).resolve(strict=False)
            try:
                path.relative_to(self.model_cache.resolve(strict=True))
                stat = path.stat()
            except (OSError, ValueError):
                raise ValueError("The requested model was not prepared.") from None
            if not path.is_file() or stat.st_size != size or stat.st_mtime_ns != mtime_ns:
                raise ValueError("The requested model was not prepared.")
            identity_parts.append((name, size, mtime_ns, digest.lower()))
            verified_assets[name] = path
            if path.name.casefold() == model.casefold():
                model_path = path
                model_digest = digest.lower()
        if model_path is None:
            raise ValueError("The requested model was not prepared.")
        if not {"vr_model_data.json", "mdx_model_data.json"}.issubset(verified_assets):
            raise ValueError("The pinned local model metadata is missing.")
        loader = _MODEL_LOADERS.get(model)
        if loader is None:
            raise ValueError("The requested model has no offline loader configuration.")
        if loader[2] is not None and loader[2] not in verified_assets:
            raise ValueError("The pinned local model configuration is missing.")
        self._verified_model_assets = verified_assets
        return model_path, tuple(sorted(identity_parts)), model_digest

    def _offline_model_files(self, model_filename: str):
        """Resolve a prepared model without audio-separator's remote discovery path."""
        loader = _MODEL_LOADERS.get(model_filename)
        model_path = self._verified_model_assets.get(model_filename)
        if loader is None or model_path is None or not model_path.is_file():
            raise ValueError("The requested model has no verified offline asset set.")
        model_type, friendly_name, yaml_name = loader
        self.separator.model_is_uvr_vip = False
        self.separator.model_friendly_name = friendly_name
        return model_filename, model_type, friendly_name, str(model_path), yaml_name

    def _offline_model_data_using_hash(self, model_path: str) -> dict[str, Any]:
        """Load pinned UVR model metadata from the runtime cache; never fetch it."""
        path = Path(model_path).resolve(strict=True)
        try:
            path.relative_to(self.model_cache.resolve(strict=True))
        except (OSError, ValueError):
            raise ValueError("The requested model is outside the prepared cache.") from None
        # UVR metadata uses Audio Separator's specific tail hash, not the
        # full file digest that the installer uses to verify downloaded assets.
        model_digest = self.separator.get_model_hash(str(path))

        vr_data = self._read_local_model_json("vr_model_data.json")
        mdx_data = self._read_local_model_json("mdx_model_data.json")
        try:
            with resources.open_text("audio_separator", "model-data.json", encoding="utf-8") as source:
                package_data = json.load(source)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise ValueError("The installed audio-separator model metadata is unavailable.") from None
        if not isinstance(package_data, dict):
            raise ValueError("The installed audio-separator model metadata is invalid.")
        vr_additions = package_data.get("vr_model_data", {})
        mdx_additions = package_data.get("mdx_model_data", {})
        if not isinstance(vr_additions, dict) or not isinstance(mdx_additions, dict):
            raise ValueError("The installed audio-separator model metadata is invalid.")
        vr_data.update(vr_additions)
        mdx_data.update(mdx_additions)
        result = mdx_data.get(model_digest) or vr_data.get(model_digest)
        if not isinstance(result, dict):
            raise ValueError("The prepared model hash has no pinned inference metadata.")
        return result

    def _read_local_model_json(self, name: str) -> dict[str, Any]:
        path = self._verified_model_assets.get(name)
        if path is None:
            raise ValueError("The pinned local model metadata is missing.")
        try:
            if path.stat().st_size > _MODEL_DATA_LIMIT:
                raise ValueError("The pinned local model metadata is too large.")
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("The pinned local model metadata is invalid.") from None
        if not isinstance(value, dict):
            raise ValueError("The pinned local model metadata is invalid.")
        return value

    @staticmethod
    def _network_discovery_disabled(*_args, **_kwargs):
        raise RuntimeError("Remote audio model discovery is disabled in this worker.")

    def _load_model(self, model: str, output_dir: Path) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_. -]{1,128}", model) or ".." in model:
            raise ValueError("Invalid separator model filename.")
        model_path, identity, model_digest = self._prepared_model(model)
        if self.model == model and self.separator is not None and self.model_identity == identity:
            self.separator.output_dir = str(output_dir)
            return
        self.separator = None
        self.model = ""
        self.model_identity = None
        self.model_sha256 = ""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        from audio_separator.separator import Separator

        self.separator = Separator(
            output_dir=str(output_dir),
            model_file_dir=str(self.model_cache),
            output_format="WAV",
            output_single_stem=None,
            use_soundfile=True,
            normalization_threshold=0.9,
            sample_rate=44100,
            mdx_params={"segment_size": 256, "batch_size": 1, "overlap": 8},
        )
        # audio-separator 0.47.1's default model loader downloads its supported
        # model index, model weights/configs, and VR/MDX metadata on demand.
        # This worker only uses installer-verified local assets, so replace
        # those discovery hooks for the lifetime of this separator instance.
        self.separator.download_model_files = self._offline_model_files
        self.separator.load_model_data_using_hash = self._offline_model_data_using_hash
        self.separator.list_supported_model_files = self._network_discovery_disabled
        self.separator.download_file_if_not_exists = self._network_discovery_disabled
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in the selected audio environment.")
        elif self.device == "cpu":
            self.separator.torch_device = self.separator.torch_device_cpu
            self.separator.onnx_execution_provider = ["CPUExecutionProvider"]
        self.separator.load_model(model_filename=model)
        self.model = model
        self.model_identity = identity
        self.model_sha256 = model_digest

    def _stem_plan(self, kind: str) -> tuple[dict[str, str], list[str]]:
        model_instance = getattr(self.separator, "model_instance", None)
        if model_instance is None:
            raise _StemMappingError("The selected model has no loaded stem metadata.")
        primary = str(getattr(model_instance, "primary_stem_name", "") or "")
        secondary = str(getattr(model_instance, "secondary_stem_name", "") or "")
        if not primary or not secondary or primary.casefold() == secondary.casefold():
            raise _StemMappingError("The model does not declare two unique output stems.")
        stem_map = getattr(self.separator, "STEM_NAME_MAP", {})
        if not isinstance(stem_map, dict) or not stem_map:
            try:
                from audio_separator.separator.separator import STEM_NAME_MAP

                stem_map = STEM_NAME_MAP
            except ImportError:
                stem_map = {}

        def canonical(value: str) -> str:
            known = stem_map.get(value.casefold(), value)
            return " ".join(str(known).casefold().replace("_", " ").replace("-", " ").split())

        labels = {canonical(primary): primary, canonical(secondary): secondary}
        if len(labels) != 2:
            raise _StemMappingError("The model stem metadata is ambiguous.")
        if kind in {"vocals", "backing"}:
            vocal_stem = next((stem for label, stem in labels.items() if label in VOCAL_STEMS), None)
            instrumental_stem = next((stem for label, stem in labels.items() if label in INSTRUMENTAL_STEMS), None)
            # This exact vocals model calls its accompaniment stem "other".
            if kind == "vocals" and self.model == BECRUILY_VOCALS_MODEL and self.model_sha256 == BECRUILY_VOCALS_SHA256:
                instrumental_stem = instrumental_stem or labels.get("other")
            if not vocal_stem or not instrumental_stem:
                raise _StemMappingError("The model does not declare vocal and instrumental output stems.")
            mapping = {vocal_stem: "vocals", instrumental_stem: "instrumental"} if kind == "vocals" else {vocal_stem: "lead", instrumental_stem: "backing"}
        else:
            clean_labels = set(DRY_STEMS)
            if self.model.casefold() == DEECHO_DEREVERB_MODEL.casefold() and self.model_sha256 == DEECHO_DEREVERB_SHA256:
                clean_labels.add("no other")
            dry_stem = next((stem for label, stem in labels.items() if label in clean_labels), None)
            if not dry_stem:
                raise _StemMappingError("The clean model does not declare a dry output stem.")
            mapping = {dry_stem: "dry"}
        return mapping, list(mapping.values())

    def separate(self, request: dict[str, Any]) -> dict[str, Any]:
        if set(request) != {"cmd", "input", "output_dir", "kind", "model"}:
            raise ValueError("Invalid separator request fields.")
        kind = request["kind"]
        if kind not in {"vocals", "backing", "clean"}:
            raise ValueError("Invalid separator output kind.")
        source = _inside(Path(str(request["input"])), self.runtime_dir)
        if not source.is_file() or source.stat().st_size > MAX_AUDIO_BYTES:
            raise ValueError("Input audio is missing or too large.")
        raw_output = Path(str(request["output_dir"])).expanduser().resolve(strict=False)
        try:
            raw_output.relative_to(self.runtime_dir)
        except ValueError:
            raise ValueError("The output directory is outside the managed runtime.") from None
        raw_output.mkdir(parents=True, exist_ok=True)
        output = raw_output.resolve(strict=True)
        if self.model_cache is None:
            raise RuntimeError("Separator is not initialized.")
        self._load_model(str(request["model"]), output)
        # Avoid output collisions across repeated stages while keeping every
        # resulting file inside the caller's output folder.
        self.separator.output_dir = str(output)
        self.separator.output_format = "WAV"
        if getattr(self.separator, "model_instance", None) is not None:
            self.separator.model_instance.output_dir = str(output)
            self.separator.model_instance.output_format = "WAV"
        stem_names, expected_names = self._stem_plan(kind)
        if any((output / f"{name}.wav").exists() for name in expected_names):
            raise ValueError("A named stem already exists in this job directory.")
        produced = self.separator.separate(str(source), custom_output_names=stem_names)
        if not isinstance(produced, (list, tuple)) or not produced or len(produced) > 32:
            raise ValueError("The separator returned an invalid output list.")
        paths: list[Path] = []
        for item in produced:
            raw_path = Path(str(item))
            if not raw_path.is_absolute():
                raw_path = output / raw_path
            path = _inside(raw_path, output)
            if not path.is_file() or path.stat().st_size > MAX_AUDIO_BYTES:
                raise ValueError("A separator output is missing or too large.")
            paths.append(path)
        if len(set(paths)) != len(paths):
            raise ValueError("The separator returned duplicate output paths.")
        result: dict[str, str] = {}
        fixed_names = {name: output / f"{name}.wav" for name in expected_names}
        for name, path in fixed_names.items():
            actual = next((candidate for candidate in paths if candidate.name.casefold() == path.name.casefold()), path)
            actual = _inside(actual, output)
            if not actual.is_file() or actual.stat().st_size > MAX_AUDIO_BYTES:
                raise ValueError("A requested named stem is missing or too large.")
            path = actual
            canonical_name = next(semantic for source_stem, semantic in stem_names.items() if semantic == name)
            name = canonical_name
            fixed = output / f"{name}.wav"
            result[name] = str(fixed)
        # Audio Separator may leave ancillary files in the job folder. Only
        # remove files this invocation returned, and only after stems map.
        for path in paths:
            if path.exists() and path.name.casefold() not in {f"{name}.wav" for name in expected_names}:
                path.unlink(missing_ok=True)
        return {"stems": result}


def _response(protocol_stdout, payload: dict[str, Any]) -> None:
    protocol_stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    protocol_stdout.flush()


def serve(runtime_dir: Path) -> int:
    protocol_stdout = sys.stdout
    try:
        root = runtime_dir.resolve(strict=True)
        startup_line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if len(startup_line) > MAX_REQUEST_BYTES:
            return 2
        startup = json.loads(startup_line)
        if startup.get("cmd") != "start" or set(startup) != {"cmd", "settings"}:
            raise ValueError("Invalid separator startup request.")
        service = _Audio(root)
        service.start(startup["settings"])
        _response(protocol_stdout, {"ok": True, "ready": True})
        while True:
            raw = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
            if not raw:
                break
            if len(raw) > MAX_REQUEST_BYTES:
                _response(protocol_stdout, {"ok": False, "code": "request_too_large", "message": "Worker request too large."})
                break
            try:
                request = json.loads(raw)
                if not isinstance(request, dict):
                    raise ValueError("Invalid separator request.")
                if request.get("cmd") == "close" and set(request) == {"cmd"}:
                    _response(protocol_stdout, {"ok": True, "closed": True})
                    break
                if request.get("cmd") != "separate":
                    raise ValueError("Unsupported separator command.")
                result = service.separate(request)
                _response(protocol_stdout, {"ok": True, **result})
            except _StemMappingError:
                _response(protocol_stdout, {"ok": False, "code": "stem_map_invalid", "message": "UVR 模型声明的音轨与当前分离阶段不匹配。"})
            except Exception as exc:
                code = "invalid_request" if isinstance(exc, (ValueError, TypeError, KeyError)) else "separation_failed"
                _response(protocol_stdout, {"ok": False, "code": code, "message": "Audio separation failed."})
    except Exception:
        _response(protocol_stdout, {"ok": False, "code": "startup_failed", "message": "Separator worker could not start."})
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-dir", required=True)
    args = parser.parse_args()
    return serve(Path(args.runtime_dir))


if __name__ == "__main__":
    raise SystemExit(main())
