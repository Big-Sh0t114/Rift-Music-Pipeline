"""Prepare and run the standalone NachoBot music runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from music_runtime.installer import RuntimeInstaller  # noqa: E402
from music_runtime.octave_probe_assets import prepare_octave_probe_assets  # noqa: E402
from music_runtime.phrase_assets import prepare_phrase_assets  # noqa: E402
from music_runtime.runtime import MusicRuntime, _worker_env, discover_speakers  # noqa: E402
from music_runtime.settings import PipelineError, load_settings  # noqa: E402
from music_runtime.service import serve  # noqa: E402


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and run the standalone music production service.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Check installed runtime components and models.")
    prepare = commands.add_parser("prepare", help="Prepare isolated UVR, RIFT and Netease runtimes.")
    prepare.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    prepare.add_argument(
        "--component", action="append",
        choices=("all", "control", "audio", "rift", "login", "netease"),
        help="May be repeated; defaults to all components.",
    )
    commands.add_parser("login", help="Create a QR PNG on this host and complete Netease login.")
    commands.add_parser("speakers", help="Read speaker names from the configured RIFT checkpoint.")
    commands.add_parser("prepare-phrases", help="Prepare isolated Silero and WeSpeaker assets for RIFT phrase clustering.")
    commands.add_parser("prepare-octave-probe", help="Prepare optional pinned Essentia models for shadow octave QA.")
    profile = commands.add_parser("profile-speaker", help="Build a RIFT voice profile from one speaker's training audio.")
    profile.add_argument("--input", type=Path, required=True, help="Training WAV/FLAC directory for one speaker.")
    profile.add_argument("--output", type=Path, required=True, help="New JSON profile path.")
    profile.add_argument("--speaker", help="Speaker name from the checkpoint. Defaults to [rift].speaker.")
    profile.add_argument("--max-files", type=int, default=1000, help="Maximum number of evenly sampled training files.")
    run = commands.add_parser("serve", help="Run the authenticated versioned HTTP API.")
    run.add_argument("--host", help="Overrides MUSIC_RUNTIME_HOST; default is 127.0.0.1.")
    run.add_argument("--port", type=int, help="Overrides MUSIC_RUNTIME_PORT; default is 8765.")
    return parser


async def _login() -> int:
    from music_runtime.providers.login import NeteaseLoginManager

    settings = load_settings()
    installer = RuntimeInstaller(settings)
    try:
        report = await installer.prepare(components=("control",))
        states = report.get("components", {})
        if not all(isinstance(states.get(name), dict) and states[name].get("ready") for name in ("login", "netease")):
            _print({"ready": False, "status": "runtime_not_ready"})
            return 1
    finally:
        await installer.close()

    manager = NeteaseLoginManager(settings)
    await manager.startup()
    shown = False
    try:
        started = await manager.start()
        job_id = str(started.get("job_id") or "")
        if not job_id:
            _print({"ready": False, "status": started.get("status", "error"), "message": started.get("message", "Login could not start.")})
            return 1
        while True:
            current = await manager.status(job_id)
            if current is None:
                return 1
            if current.get("qr_ready") and not shown:
                qr_path = settings["runtime_dir"] / "login" / job_id / "qr.png"
                _print({"ready": True, "status": current.get("status"), "qr_png": str(qr_path)})
                if os.name == "nt" and qr_path.is_file():
                    try:
                        os.startfile(str(qr_path))  # type: ignore[attr-defined]
                    except OSError:
                        pass
                shown = True
            if current.get("terminal"):
                success = current.get("status") == "success"
                _print({"ready": success, "status": current.get("status"), "message": current.get("message")})
                return 0 if success else 1
            await asyncio.sleep(0.4)
    except KeyboardInterrupt:
        return 130
    finally:
        await manager.shutdown()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check":
            report = RuntimeInstaller().check()
            _print(report)
            return 0 if report.get("ready") else 1
        if args.command == "prepare":
            report = asyncio.run(RuntimeInstaller().prepare(args.device, components=args.component))
            _print(report)
            if args.component is None or "all" in args.component:
                selected = ("audio", "rift", "login", "netease")
            else:
                selected_items: list[str] = []
                for component in args.component:
                    if component == "control":
                        selected_items.extend(("netease", "login"))
                    elif component in {"login", "netease", "audio", "rift"}:
                        selected_items.append(component)
                if "login" in selected_items and "netease" not in selected_items:
                    selected_items.append("netease")
                selected = tuple(dict.fromkeys(selected_items))
            states = report.get("components", {})
            return 0 if all(
                isinstance(states.get(name), dict) and states[name].get("ready") for name in selected
            ) else 1
        if args.command == "login":
            return asyncio.run(_login())
        if args.command == "speakers":
            settings = load_settings()
            speakers = asyncio.run(discover_speakers(settings))
            _print({"ready": bool(speakers), "speakers": speakers})
            return 0 if speakers else 1
        if args.command == "prepare-phrases":
            runtime = MusicRuntime(load_settings())
            _, python = runtime._rift_paths()
            if not python.is_file():
                raise PipelineError("rift_not_ready", "Prepare the RIFT environment before installing phrase models.")
            path = prepare_phrase_assets(runtime.runtime_dir / "models/phrases", python)
            _print({"ready": True, "path": str(path)})
            return 0
        if args.command == "prepare-octave-probe":
            runtime = MusicRuntime(load_settings())
            if not (runtime.runtime_dir / "models/phrases/packages/onnxruntime/__init__.py").is_file():
                raise PipelineError(
                    "rift_phrase_models_missing",
                    "Run prepare-phrases before preparing the octave probe models.",
                )
            path = prepare_octave_probe_assets(runtime.runtime_dir / "models/octave_probe")
            _print({"ready": True, "path": str(path)})
            return 0
        if args.command == "profile-speaker":
            settings = load_settings()
            runtime = MusicRuntime(settings)
            source, python = runtime._rift_paths()
            speaker = args.speaker or settings["rift"]["speaker"]
            if not speaker or not args.input.is_dir():
                raise PipelineError("invalid_profile_input", "Provide a speaker name and its training audio directory.")
            if not python.is_file() or not (source / "infer.py").is_file() or not runtime._rift_pretrained_ready(source):
                raise PipelineError("rift_not_ready", "Prepare the RIFT environment and pretrained assets before profiling.")
            return subprocess.run([
                str(python), str(runtime.worker_dir / "build_voice_profile.py"), "--source", str(source),
                "--input", str(args.input.resolve()), "--output", str(args.output.resolve()), "--speaker", speaker,
                "--device", settings["runtime"]["device"], "--max-files", str(args.max_files),
            ], cwd=source, env=_worker_env(), check=False).returncode
        if args.command == "serve":
            serve(args.host, args.port)
            return 0
    except PipelineError as exc:
        _print({"ready": False, "error_code": exc.code, "error": exc.message})
        return 1
    except (RuntimeError, ValueError) as exc:
        _print({"ready": False, "error_code": "configuration_error", "error": str(exc)[:300]})
        return 1
    except KeyboardInterrupt:
        return 130
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
