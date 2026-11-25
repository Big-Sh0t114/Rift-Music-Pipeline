"""Validated settings for the independent music runtime."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
import tomllib
import tomlkit
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


RUNTIME_ROOT = Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    configured = os.environ.get("MUSIC_RUNTIME_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local).expanduser() if local else Path.home() / "AppData" / "Local"
        return (base / "NachoBot" / "MusicRuntime").resolve()
    base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))).expanduser()
    return (base / "nachobot-music-runtime").resolve()


DATA_DIR = default_data_dir()
PLUGIN_DIR = DATA_DIR  # Retained key for relative checkpoints in the copied engine.
RUNTIME_DIR = DATA_DIR
SETTINGS_PATH = DATA_DIR / "config.toml"
_SECTIONS = ("pipeline", "netease", "separation", "rift", "mix", "runtime")

DEFAULTS: dict[str, dict[str, Any]] = {
    "pipeline": {
        "enabled": True,
        "max_queue": 5,
        "max_duration_seconds": 600,
        "max_download_bytes": 100 * 1024 * 1024,
        "stage_timeout_seconds": 900,
    },
    "netease": {
        "base_url": "http://127.0.0.1:13306",
        "profile": "operator",
    },
    "separation": {
        "vocals_model": "mel_band_roformer_vocals_becruily.ckpt",
        "backing_model": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "clean_model": "UVR-DeEcho-DeReverb.pth",
    },
    "rift": {
        "checkpoint": "",
        "speaker": "",
        "backing_checkpoint": "",
        "backing_speaker": "",
        "key_shift": 0,
        "backing_key_shift": 0,
        "song_octave_shift": 0,
        "infer_steps": 38,
        "ds_cfg_strength": 0.1,
        "spk_cfg_strength": 0.55,
        "cfg_rescale": 0.7,
        "batch_size": 1,
        "robust_f0": 1,
        "adaptive_quality": True,
        "timbre_slicing": True,
        "domain_alignment": True,
        "domain_formants": True,
        "allow_octave_jump": False,
        "octave_probe": False,
        "octave_probe_target_voice": "unknown",
        "lead_pitch_range": [],
        "backing_pitch_range": [],
        "lead_voice_profile": "",
        "backing_voice_profile": "",
    },
    "mix": {"lead_gain": 1.0, "backing_gain": 0.7},
    "runtime": {
        "device": "auto",
        "rift_source": "",
        "rift_python": "",
    },
}


class PipelineError(RuntimeError):
    """A safe user-facing music pipeline error."""

    def __init__(self, code: str, message: str):
        self.code = str(code)[:64]
        self.message = str(message)[:500]
        super().__init__(self.message)


class SelectionRequired(PipelineError):
    """Raised when a provider query is ambiguous and needs a user choice."""

    def __init__(self, candidates: list[dict[str, Any]]):
        self.candidates = list(candidates[:8])
        super().__init__("selection_required", "More than one song matches; choose a result.")


def _default_plugin_dir(plugin_dir: Path | None) -> Path:
    return Path(plugin_dir or DATA_DIR).expanduser().resolve()


def _canonical_loopback_url(value: Any) -> str | None:
    """Normalize a credential-free local HTTP base URL for service matching."""
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    host = (parts.hostname or "").casefold()
    if (
        parts.scheme != "http"
        or host not in {"127.0.0.1", "localhost", "::1"}
        or parts.username
        or parts.password
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
        or port is None
        or not 1 <= port <= 65535
    ):
        return None
    return f"http://127.0.0.1:{port}"


def netease_source_fingerprint(settings: Mapping[str, Any]) -> str:
    """Return the canonical, credential-free identity used for source caching."""
    netease = settings.get("netease", {})
    if not isinstance(netease, Mapping):
        netease = {}
    base_url = str(netease.get("base_url", "http://127.0.0.1:13306")).rstrip("/")
    endpoint = _canonical_loopback_url(base_url) or base_url
    return f"netease:{netease.get('profile', 'operator')}:{endpoint}"


def _settings_path(plugin_dir: Path | None) -> Path:
    return _default_plugin_dir(plugin_dir) / "config.toml"


def _copy_sections(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {section: dict(value.get(section, {})) for section in _SECTIONS}


def _merge(raw: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULTS)
    for section in _SECTIONS:
        values = raw.get(section, {})
        if not isinstance(values, Mapping):
            raise PipelineError("invalid_settings", f"The [{section}] settings section must be a table.")
        unknown = set(values) - set(merged[section])
        if unknown:
            raise PipelineError("invalid_settings", f"The [{section}] settings contain an unsupported key.")
        merged[section].update(values)
    return merged


def _finite_number(section: str, key: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError("invalid_settings", f"[{section}].{key} must be a number.")
    number = float(value)
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise PipelineError("invalid_settings", f"[{section}].{key} must be between {minimum:g} and {maximum:g}.")
    return number


def _integer(section: str, key: str, value: Any, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PipelineError("invalid_settings", f"[{section}].{key} must be an integer between {minimum} and {maximum}.")
    return value


def _text(section: str, key: str, value: Any, maximum: int = 512, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise PipelineError("invalid_settings", f"[{section}].{key} must be text.")
    value = value.strip()
    if len(value) > maximum or (not empty and not value):
        raise PipelineError("invalid_settings", f"[{section}].{key} must contain at most {maximum} characters.")
    if "\x00" in value:
        raise PipelineError("invalid_settings", f"[{section}].{key} contains an invalid character.")
    return value


def _validate_checkpoint(
    value: str,
    plugin_dir: Path,
    key: str = "checkpoint",
    *,
    require_exists: bool = True,
) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    absolute = path.is_absolute()
    if not absolute:
        path = plugin_dir / path
    try:
        path = path.resolve(strict=require_exists)
    except OSError:
        if require_exists:
            raise PipelineError("checkpoint_missing", f"RIFT {key.replace('_', ' ')} file was not found.") from None
        path = Path(value).expanduser()
        if not absolute:
            path = plugin_dir / path
        path = path.resolve(strict=False)
    if path.suffix.casefold() not in {".ckpt", ".pth", ".pt"}:
        raise PipelineError("invalid_checkpoint", f"RIFT {key.replace('_', ' ')} must be a .ckpt, .pth, or .pt file.")
    if require_exists and not path.is_file():
        raise PipelineError("invalid_checkpoint", f"RIFT {key.replace('_', ' ')} must be a .ckpt, .pth, or .pt file.")
    if absolute:
        return str(path)
    try:
        return path.relative_to(plugin_dir).as_posix()
    except ValueError:
        raise PipelineError("invalid_checkpoint", f"Relative RIFT {key.replace('_', ' ')} must stay under the plugin folder.") from None


def _validate_source_override(
    value: str,
    plugin_dir: Path,
    key: str,
    *,
    require_exists: bool = True,
) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = plugin_dir / path
    try:
        resolved = path.resolve(strict=require_exists)
    except OSError:
        if require_exists:
            raise PipelineError("runtime_not_found", f"Configured {key.replace('_', ' ')} was not found.") from None
        resolved = path.resolve(strict=False)
    if require_exists and key == "rift_source" and not (resolved / "infer.py").is_file():
        raise PipelineError("runtime_not_found", "Configured RIFT source does not contain infer.py.")
    if require_exists and key == "rift_python" and not resolved.is_file():
        raise PipelineError("runtime_not_found", "Configured RIFT Python interpreter was not found.")
    return str(resolved)


def validate_settings(
    value: Mapping[str, Any],
    plugin_dir: Path | None = None,
    *,
    require_paths: bool = True,
) -> dict[str, Any]:
    """Validate known public fields and inject resolved plugin/runtime paths."""
    root = _default_plugin_dir(plugin_dir)
    merged = _merge(value)
    p = merged["pipeline"]
    if type(p["enabled"]) is not bool:
        raise PipelineError("invalid_settings", "[pipeline].enabled must be true or false.")
    p["max_queue"] = _integer("pipeline", "max_queue", p["max_queue"], 1, 100)
    p["max_duration_seconds"] = _integer("pipeline", "max_duration_seconds", p["max_duration_seconds"], 1, 7200)
    p["max_download_bytes"] = _integer("pipeline", "max_download_bytes", p["max_download_bytes"], 1024, 2 * 1024 * 1024 * 1024)
    p["stage_timeout_seconds"] = _integer("pipeline", "stage_timeout_seconds", p["stage_timeout_seconds"], 10, 7200)

    n = merged["netease"]
    base_url = _text("netease", "base_url", n["base_url"], 256)
    parts = urlsplit(base_url)
    if parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"} or parts.username or parts.password or parts.path not in {"", "/"} or parts.query or parts.fragment:
        raise PipelineError("invalid_settings", "[netease].base_url must be an HTTP loopback URL without a path or credentials.")
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None or not 1 <= port <= 65535:
        raise PipelineError("invalid_settings", "[netease].base_url must include a valid loopback port.")
    n["base_url"] = base_url.rstrip("/")
    n["profile"] = _text("netease", "profile", n["profile"], 64)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", n["profile"]):
        raise PipelineError("invalid_settings", "[netease].profile contains unsupported characters.")

    sep = merged["separation"]
    for key, model in tuple(sep.items()):
        model = _text("separation", key, model, 128)
        if not re.fullmatch(r"[A-Za-z0-9_. -]{1,128}", model) or ".." in model:
            raise PipelineError("invalid_settings", f"[separation].{key} must be a model filename.")
        sep[key] = model

    r = merged["rift"]
    r["checkpoint"] = _validate_checkpoint(
        _text("rift", "checkpoint", r["checkpoint"], 1024, empty=True),
        root,
        require_exists=require_paths,
    )
    r["backing_checkpoint"] = _validate_checkpoint(
        _text("rift", "backing_checkpoint", r["backing_checkpoint"], 1024, empty=True),
        root,
        "backing_checkpoint",
        require_exists=require_paths,
    )
    r["speaker"] = _text("rift", "speaker", r["speaker"], 128, empty=True)
    r["backing_speaker"] = _text("rift", "backing_speaker", r["backing_speaker"], 128, empty=True)
    r["key_shift"] = _integer("rift", "key_shift", r["key_shift"], -24, 24)
    r["backing_key_shift"] = _integer("rift", "backing_key_shift", r["backing_key_shift"], -24, 24)
    song_octave_shift = r["song_octave_shift"]
    if type(song_octave_shift) is not int or song_octave_shift not in {-12, 0, 12}:
        raise PipelineError("invalid_settings", "[rift].song_octave_shift must be -12, 0, or 12.")
    if song_octave_shift and (r["key_shift"] or r["backing_key_shift"]):
        raise PipelineError(
            "invalid_settings",
            "[rift].song_octave_shift cannot be combined with nonzero [rift].key_shift or [rift].backing_key_shift.",
        )
    r["infer_steps"] = _integer("rift", "infer_steps", r["infer_steps"], 1, 200)
    r["ds_cfg_strength"] = _finite_number("rift", "ds_cfg_strength", r["ds_cfg_strength"], 0.0, 5.0)
    r["spk_cfg_strength"] = _finite_number("rift", "spk_cfg_strength", r["spk_cfg_strength"], 0.0, 5.0)
    r["cfg_rescale"] = _finite_number("rift", "cfg_rescale", r["cfg_rescale"], 0.0, 1.0)
    r["batch_size"] = _integer("rift", "batch_size", r["batch_size"], 1, 16)
    r["robust_f0"] = _integer("rift", "robust_f0", r["robust_f0"], 0, 2)
    if type(r["adaptive_quality"]) is not bool:
        raise PipelineError("invalid_settings", "[rift].adaptive_quality must be true or false.")
    if type(r["timbre_slicing"]) is not bool:
        raise PipelineError("invalid_settings", "[rift].timbre_slicing must be true or false.")
    for key in ("domain_alignment", "domain_formants", "allow_octave_jump"):
        if type(r[key]) is not bool:
            raise PipelineError("invalid_settings", f"[rift].{key} must be true or false.")
    if type(r["octave_probe"]) is not bool:
        raise PipelineError("invalid_settings", "[rift].octave_probe must be true or false.")
    target_voice = _text("rift", "octave_probe_target_voice", r["octave_probe_target_voice"], 16)
    if target_voice not in {"unknown", "female", "male"}:
        raise PipelineError("invalid_settings", "[rift].octave_probe_target_voice must be unknown, female, or male.")
    r["octave_probe_target_voice"] = target_voice
    for key in ("lead_pitch_range", "backing_pitch_range"):
        raw_pitch_range = r[key]
        if not isinstance(raw_pitch_range, list) or len(raw_pitch_range) not in (0, 3):
            raise PipelineError("invalid_settings", f"[rift].{key} must be empty or contain three MIDI notes.")
        pitch_range = [_finite_number("rift", key, note, 0.0, 127.0) for note in raw_pitch_range]
        if pitch_range and (pitch_range != sorted(pitch_range) or pitch_range[0] == pitch_range[-1]):
            raise PipelineError("invalid_settings", f"[rift].{key} must contain ordered P10, median and P90 notes.")
        r[key] = pitch_range
    for key in ("lead_voice_profile", "backing_voice_profile"):
        value = _text("rift", key, r[key], 1024, empty=True)
        if value:
            path = Path(value).expanduser()
            path = (root / path).resolve()
            if path.suffix.casefold() != ".json" or (require_paths and not path.is_file()):
                raise PipelineError("invalid_settings", f"[rift].{key} must name a speaker profile JSON file.")
            if not r["adaptive_quality"]:
                raise PipelineError("invalid_settings", f"[rift].{key} requires adaptive_quality.")
            value = str(path)
        r[key] = value

    m = merged["mix"]
    m["lead_gain"] = _finite_number("mix", "lead_gain", m["lead_gain"], 0.0, 4.0)
    m["backing_gain"] = _finite_number("mix", "backing_gain", m["backing_gain"], 0.0, 4.0)

    runtime = merged["runtime"]
    if runtime["device"] not in {"auto", "cpu", "cuda"}:
        raise PipelineError("invalid_settings", "[runtime].device must be auto, cpu, or cuda.")
    runtime["rift_source"] = _validate_source_override(
        _text("runtime", "rift_source", runtime["rift_source"], 1024, empty=True),
        root,
        "rift_source",
        require_exists=require_paths,
    )
    runtime["rift_python"] = _validate_source_override(
        _text("runtime", "rift_python", runtime["rift_python"], 1024, empty=True),
        root,
        "rift_python",
        require_exists=require_paths,
    )

    merged["plugin_dir"] = root
    merged["runtime_dir"] = root
    managed_ffmpeg = root / "models" / "tools" / "ffmpeg.exe"
    merged["ffmpeg_path"] = managed_ffmpeg if managed_ffmpeg.is_file() else None
    return merged


def load_settings(plugin_dir: Path | None = None) -> dict[str, Any]:
    """Load config.toml, merge defaults, validate, and inject plugin paths."""
    path = _settings_path(plugin_dir)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, tomllib.TOMLDecodeError):
        raise PipelineError("invalid_settings", "The music pipeline settings file could not be read.") from None
    return validate_settings(raw, plugin_dir, require_paths=False)


def public_settings(settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return only UI settings; never include injected directories or credentials."""
    source = settings or load_settings()
    result = _copy_sections(source)
    # Override paths with their configured public value, not an internal resolved path.
    original = source.get("rift", {})
    for key in ("checkpoint", "backing_checkpoint"):
        path = str(original.get(key, "") or "")
        result["rift"][key] = path
    return result


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise PipelineError("invalid_settings", "Only scalar settings values are supported.")


def _dump_toml(document: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for name, values in document.items():
        if not isinstance(values, Mapping):
            continue
        lines.append(f"[{name}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")
    return "\n".join(lines)


def save_settings(values: Mapping[str, Any], plugin_dir: Path | None = None) -> dict[str, Any]:
    """Merge a partial approved section map and atomically persist config.toml."""
    if not isinstance(values, Mapping):
        raise PipelineError("invalid_settings", "Settings must be an object.")
    unknown_sections = set(values) - set(_SECTIONS)
    if unknown_sections:
        raise PipelineError("invalid_settings", "The settings object contains an unsupported section.")
    root = _default_plugin_dir(plugin_dir)
    path = root / "config.toml"
    try:
        source = path.read_text(encoding="utf-8") if path.is_file() else ""
        document = tomlkit.parse(source)
    except (OSError, tomllib.TOMLDecodeError, tomlkit.exceptions.ParseError):
        raise PipelineError("invalid_settings", "The existing settings file could not be read.") from None
    original_paths: dict[tuple[str, str], str] = {}
    for section, key in (
        ("rift", "checkpoint"),
        ("rift", "backing_checkpoint"),
        ("runtime", "rift_source"),
        ("runtime", "rift_python"),
    ):
        table = document.get(section)
        value = table.get(key, "") if isinstance(table, Mapping) else ""
        original_paths[(section, key)] = str(value or "")

    for section, partial in values.items():
        if not isinstance(partial, Mapping):
            raise PipelineError("invalid_settings", f"[{section}] must be an object.")
        allowed = set(DEFAULTS[section])
        if set(partial) - allowed:
            raise PipelineError("invalid_settings", f"[{section}] contains an unsupported key.")
        target = document.get(section)
        if target is None:
            target = tomlkit.table()
            document[section] = target
        if not hasattr(target, "__setitem__"):
            raise PipelineError("invalid_settings", f"[{section}] must be a table.")
        for key, value in partial.items():
            target[key] = value
    # Stored paths can become unavailable after a drive disconnect or file
    # removal. Keep them readable so the operator can correct them in the UI.
    validated = validate_settings(document.unwrap(), root, require_paths=False)
    rift_updates = values.get("rift", {})
    runtime_updates = values.get("runtime", {})
    for key in ("checkpoint", "backing_checkpoint"):
        if key in rift_updates and str(rift_updates[key] or "") != original_paths[("rift", key)]:
            _validate_checkpoint(validated["rift"][key], root, key, require_exists=True)
    for key in ("rift_source", "rift_python"):
        if key in runtime_updates and str(runtime_updates[key] or "") != original_paths[("runtime", key)]:
            _validate_source_override(validated["runtime"][key], root, key, require_exists=True)
    public = public_settings(validated)
    # Preserve existing comments and unrelated sections; only normalize the
    # owned fields after validation.
    for section in _SECTIONS:
        target = document.get(section)
        if target is None:
            target = tomlkit.table()
            document[section] = target
        for key, value in public[section].items():
            target[key] = value
    root.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", prefix=".config.", suffix=".tmp", dir=root, delete=False) as output:
            temporary = output.name
            output.write(tomlkit.dumps(document))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError:
        raise PipelineError("settings_write_failed", "The music settings could not be saved.") from None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass
    return public


__all__ = [
    "DATA_DIR", "DEFAULTS", "PipelineError", "SelectionRequired", "default_data_dir", "load_settings", "public_settings",
    "save_settings", "validate_settings",
]
