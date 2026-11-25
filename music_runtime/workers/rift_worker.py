"""Persistent RIFT-SVC child protocol; heavy imports stay inside this file."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


MAX_REQUEST_BYTES = 64 * 1024
MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_OUTPUT_BYTES = 512 * 1024 * 1024
DEFAULT_SLICER = {
    "threshold": -30.0,
    "min_length": 3000,
    "min_interval": 100,
    "hop_size": 10,
    "max_sil_kept": 200,
}


def _inside(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("The requested file is outside the job directory.") from None
    return resolved


class _Rift:
    def __init__(self, source: Path, runtime_dir: Path):
        self.source = source.resolve(strict=True)
        self.runtime_dir = runtime_dir.resolve(strict=True)
        sys.path.insert(0, str(self.source))
        self._protocol_stdout = sys.stdout
        # Upstream infer.py uses click/tqdm on stdout. Keep the JSON-lines
        # protocol on the original stream and redirect model chatter to stderr.
        sys.stdout = sys.stderr
        import torch
        import soundfile as sf
        import librosa
        import numpy as np
        import infer

        self.torch = torch
        self.sf = sf
        self.librosa = librosa
        self.np = np
        self.infer = infer
        self.bundle = None
        self.spk2idx: dict[str, int] = {}
        self.loaded_path: Path | None = None
        self.loaded_key: tuple[str, int, int, str] | None = None
        self.profile_encoder_sha256: str | None = None
        self.phrase_models = None
        self._octave_probe_classifier = None

    def load(self, checkpoint: str, device_name: str) -> None:
        path = Path(checkpoint).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.casefold() not in {".ckpt", ".pt", ".pth"}:
            raise ValueError("The configured RIFT checkpoint is unavailable.")
        if device_name not in {"auto", "cpu", "cuda"}:
            raise ValueError("The configured RIFT device is invalid.")
        stat = path.stat()
        device = "cuda" if device_name == "cuda" or (device_name == "auto" and self.torch.cuda.is_available()) else "cpu"
        load_key = (str(path), stat.st_size, stat.st_mtime_ns, device)
        if self.loaded_key == load_key and self.bundle is not None:
            return
        self._unload()
        if device_name == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected RIFT environment.")
        # These are explicitly operator-supplied training checkpoints. The
        # pinned upstream infer.load_models() omits weights_only; Torch 2.9's
        # safe-default loader rejects Lightning checkpoints containing their
        # optimizer/config objects. This worker therefore permits pickle only
        # for the explicitly selected local checkpoint.
        original_load = self.torch.load

        def load_trusted_checkpoint(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        self.infer.torch.load = load_trusted_checkpoint
        self.bundle = self.infer.load_models(str(path), self.torch.device(device), use_fp16=device != "cpu")
        # Upstream NsfHifiGAN lazily constructs its inner Generator in
        # forward(). Its constructor chooses CUDA when CUDA is available even
        # when load_models later moves the wrapper to CPU, so align the lazy
        # factory's device with the already-selected inference device.
        vocoder = self.bundle[1]
        vocoder.device = self.torch.device(device)
        if getattr(vocoder, "model", None) is not None:
            vocoder.model.to(device)
        self.spk2idx = dict(self.bundle[5])
        if not self.spk2idx or len(self.spk2idx) > 4096:
            self._unload()
            raise ValueError("The selected checkpoint has no usable speaker map.")
        self.loaded_path = path
        self.loaded_key = load_key

    def _unload(self) -> None:
        self.bundle = None
        self.spk2idx = {}
        self.loaded_path = None
        self.loaded_key = None
        gc.collect()
        if getattr(self, "torch", None) is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def speakers(self) -> list[dict[str, str]]:
        return [
            {"id": str(name), "name": str(name)}
            for name, _ in sorted(self.spk2idx.items(), key=lambda pair: (int(pair[1]), str(pair[0])))[:2048]
        ]

    def convert(self, request: dict[str, Any]) -> dict[str, Any]:
        role = request.get("role")
        if role not in {"lead", "backing"}:
            raise ValueError("The requested voice conversion role is invalid.")
        rift = request.get("rift")
        if not isinstance(rift, dict):
            raise ValueError("The voice conversion settings are invalid.")
        allowed = {
            "checkpoint", "speaker", "key_shift", "infer_steps", "ds_cfg_strength",
            "spk_cfg_strength", "batch_size", "robust_f0", "device",
        }
        optional = {
            "adaptive_quality", "timbre_slicing", "pitch_range", "cfg_rescale", "voice_profile",
            "domain_alignment", "domain_formants", "allow_octave_jump", "domain_ffmpeg", "song_octave_shift",
            "song_detection_backing_manual_shift", "octave_probe", "octave_probe_target_voice",
            "octave_probe_asset_state", "octave_probe_asset_fingerprint",
        }
        if not allowed <= set(rift) or set(rift) - allowed - optional:
            raise ValueError("The voice conversion settings are incomplete.")
        song_octave_shift = rift.get("song_octave_shift", 0)
        if type(song_octave_shift) is not int or song_octave_shift not in {-12, 0, 12}:
            raise ValueError("The song octave shift must be -12, 0, or 12.")
        backing_manual_shift = rift.get("song_detection_backing_manual_shift", 0)
        if type(backing_manual_shift) is not int or not -24 <= backing_manual_shift <= 24:
            raise ValueError("The backing manual key shift is invalid.")
        if song_octave_shift and int(rift["key_shift"]):
            raise ValueError("A song octave shift cannot be combined with a nonzero manual key shift.")
        for flag in ("adaptive_quality", "timbre_slicing", "domain_alignment", "domain_formants", "allow_octave_jump"):
            if flag in rift and type(rift[flag]) is not bool:
                raise ValueError(f"The voice conversion setting {flag} must be true or false.")
        if "octave_probe" in rift and type(rift["octave_probe"]) is not bool:
            raise ValueError("The voice conversion setting octave_probe must be true or false.")
        if rift.get("octave_probe"):
            if type(rift.get("octave_probe_target_voice")) is not str or rift["octave_probe_target_voice"] not in {"unknown", "female", "male"}:
                raise ValueError("The octave probe target voice is invalid.")
            if type(rift.get("octave_probe_asset_state")) is not str or rift["octave_probe_asset_state"] not in {"prepared", "missing", "corrupt", "unavailable"}:
                raise ValueError("The octave probe asset state is invalid.")
            fingerprint = rift.get("octave_probe_asset_fingerprint")
            if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
                char not in "0123456789abcdef" for char in fingerprint.casefold()
            ):
                raise ValueError("The octave probe asset fingerprint is invalid.")
        if "domain_ffmpeg" in rift and (
            not isinstance(rift["domain_ffmpeg"], str) or len(rift["domain_ffmpeg"]) > 4096
        ):
            raise ValueError("The domain FFmpeg path is invalid.")
        self.load(str(rift["checkpoint"]), str(rift["device"]))
        speaker = str(rift["speaker"])
        if speaker not in self.spk2idx:
            raise ValueError("The configured speaker is not present in the selected checkpoint.")
        source = _inside(Path(str(request.get("input", ""))), self.runtime_dir)
        if source.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("The source audio exceeds the worker limit.")
        destination_raw = Path(str(request.get("output", "")))
        if destination_raw.suffix.casefold() != ".wav":
            raise ValueError("The requested output must be a WAV file.")
        try:
            resolved_destination = destination_raw.resolve(strict=False)
            resolved_destination.relative_to(self.runtime_dir)
        except ValueError:
            raise ValueError("The requested output is outside the job directory.") from None
        resolved_destination.parent.mkdir(parents=True, exist_ok=True)
        destination = resolved_destination
        if destination.exists():
            raise ValueError("The requested output already exists.")
        song_decision_path = destination.with_suffix(".song-octave.json")
        song_decision_path.unlink(missing_ok=True)

        audio, sample_rate = self.sf.read(str(source), dtype="float32", always_2d=True)
        if audio.ndim != 2 or not len(audio) or sample_rate < 8000 or sample_rate > 192000:
            raise ValueError("The source WAV format is not supported.")
        if not self.np.isfinite(audio).all():
            raise ValueError("The source audio contains invalid samples.")
        adaptive = rift.get("adaptive_quality", False)
        timbre_slicing = rift.get("timbre_slicing", True)
        profile = rift.get("voice_profile")
        formants_enabled = bool(rift.get("domain_formants", False))
        if adaptive:
            if __package__:
                from . import rift_quality as quality
                from .rift_profile import encoder_fingerprint, rmvpe_pitch, validate_profile
            else:
                import rift_quality as quality
                from rift_profile import encoder_fingerprint, rmvpe_pitch, validate_profile

        # RIFT-SVC V3 is monophonic. Downmix explicitly and resample to the
        # upstream 44.1 kHz inference rate, retaining its original sample map.
        frames = len(audio)
        mono = quality.downmix(audio) if adaptive else self.np.mean(audio, axis=1, dtype=self.np.float32)
        del audio
        if sample_rate != 44100:
            mono = self.librosa.resample(mono, orig_sr=int(sample_rate), target_sr=44100)
        mono = self.np.asarray(mono, dtype=self.np.float32)
        if not self.np.isfinite(mono).all():
            raise ValueError("The source audio contains invalid samples.")
        device = next(self.bundle[0].parameters()).device
        encoder = self.bundle[3]
        if profile is not None:
            if not adaptive:
                raise ValueError("A voice profile requires adaptive RIFT inference.")
            validate_profile(profile, speaker)
            if self.profile_encoder_sha256 is None:
                self.profile_encoder_sha256 = encoder_fingerprint(self.source)
            if self.profile_encoder_sha256 != profile["encoder_sha256"]:
                raise ValueError("The voice profile was built with different ContentVec assets.")
            mean = self.torch.tensor(profile["cvec_mean"], dtype=self.torch.float32, device=device)
            std = self.torch.tensor(profile["cvec_std"], dtype=self.torch.float32, device=device).clamp_min(1e-4)

        def process(segments, *, shift, speaker_cfg, detail_cfg, rescale, robust_f0, cvec_alpha=0.0,
                    target_loudness=-18.0, cvec_source_stats=None, formant_ratio=1.0):
            stats = tuple(self.torch.as_tensor(a, device=device) for a in cvec_source_stats) if cvec_source_stats else None

            def aligned_encoder(audio16):
                encoder_input = audio16
                if formants_enabled and formant_ratio != 1.0:
                    encoder_input = self._warp_contentvec_audio(audio16, formant_ratio)
                result = encoder(encoder_input)
                if not cvec_alpha:
                    return result
                return {"last_hidden_state": quality.timbre.align_cvec(result["last_hidden_state"], mean, std,
                                                                      cvec_alpha, self.torch, stats)}

            model_bundle = (
                *self.bundle[:3],
                aligned_encoder if (cvec_alpha != 0 or (formants_enabled and formant_ratio != 1.0)) else encoder,
                self.bundle[4],
            )
            return self.infer.batch_process_segments(
                segments, *model_bundle, int(self.spk2idx[speaker]), 44100, 512, device,
                key_shift=shift, infer_steps=int(rift["infer_steps"]), ds_cfg_strength=detail_cfg,
                spk_cfg_strength=speaker_cfg, cfg_rescale=rescale, restore_loudness=not adaptive,
                robust_f0=robust_f0, use_fp16=device.type != "cpu", batch_size=int(rift["batch_size"]),
                target_loudness=target_loudness,
            )

        report = None
        if adaptive:
            if timbre_slicing and self.phrase_models is None:
                if __package__:
                    from .rift_sections import PhraseModels
                else:
                    from rift_sections import PhraseModels
                self.phrase_models = PhraseModels(self.runtime_dir / "models/phrases", self.torch)

            def slice_vocals(samples, threshold):
                slicer = self.infer.Slicer(sr=44100, **{**DEFAULT_SLICER, "threshold": threshold, "min_length": 5000})
                return slicer.slice(samples)

            def pitch(samples):
                return rmvpe_pitch(samples, self.bundle[2], self.torch, device)

            def run_inference(segments, guidance, shift, **kwargs):
                return process(segments, shift=shift, speaker_cfg=guidance.speaker, detail_cfg=guidance.detail,
                               rescale=guidance.rescale, robust_f0=guidance.robust_f0, **kwargs)

            def content_features(samples, *, formant_ratio=1.0):
                audio16 = self.librosa.resample(samples, orig_sr=44100, target_sr=16000)
                if formants_enabled and formant_ratio != 1.0:
                    audio16 = self._warp_contentvec_audio(audio16, formant_ratio)
                dtype = self.torch.float16 if device.type == "cuda" else self.torch.float32
                tensor = self.torch.as_tensor(audio16, dtype=dtype, device=device).unsqueeze(0)
                with self.torch.inference_mode(), self.torch.autocast(device.type, enabled=device.type == "cuda"):
                    features = encoder(tensor)["last_hidden_state"][0].float().cpu().numpy()
                receptive, stride = 1, 1
                for kernel, hop in zip(encoder.config.conv_kernel, encoder.config.conv_stride, strict=True):
                    receptive += (kernel - 1) * stride
                    stride *= hop
                times = (self.np.arange(len(features)) * stride + (receptive - 1) / 2) / 16000
                return features, times

            phrase_mode = bool(timbre_slicing)
            adaptive_settings = {"pitch_range": [], "cfg_rescale": 0.7, "timbre_slicing": timbre_slicing, **rift}
            if not phrase_mode:
                # The legacy adaptive route has no phrase-level octave policy.
                # Apply a deliberate whole-song lock through its existing shift.
                adaptive_settings["key_shift"] = int(rift["key_shift"]) + song_octave_shift
                adaptive_settings["song_octave_shift"] = 0
            else:
                # This internal flag is derived from the job role and both
                # manual overrides; callers cannot enable it on backing or
                # create a lead-only shift when backing has its own override.
                adaptive_settings["allow_song_octave_detection"] = (
                    role == "lead"
                    and song_octave_shift == 0
                    and int(rift["key_shift"]) == 0
                    and backing_manual_shift == 0
                    and bool(rift.get("allow_octave_jump", False))
                    and bool(rift.get("domain_alignment", True))
                )
                adaptive_settings.pop("song_detection_backing_manual_shift", None)
            if rift.get("octave_probe") is True and adaptive_settings.get("allow_song_octave_detection") is True:
                adaptive_settings["octave_probe_target_voice"] = str(rift["octave_probe_target_voice"])
                adaptive_settings["octave_probe_asset_state"] = str(rift["octave_probe_asset_state"])
                adaptive_settings["octave_probe_asset_fingerprint"] = str(rift["octave_probe_asset_fingerprint"])
                adaptive_settings["_octave_probe_classify"] = self._classify_octave_probe
                adaptive_settings["_octave_probe_preserve_rng"] = self._preserve_probe_rng
            output_audio, report = quality.convert_adaptive(
                mono, slice_vocals, pitch, run_inference,
                adaptive_settings,
                phrase_models=self.phrase_models, content_features=content_features if profile else None,
            )
        else:
            segments = self.infer.Slicer(sr=44100, **DEFAULT_SLICER).slice(mono)
            if not segments:
                raise ValueError("The source contains no usable vocal segments.")
            values = process(
                segments, shift=int(rift["key_shift"]) + song_octave_shift, speaker_cfg=float(rift["spk_cfg_strength"]),
                detail_cfg=float(rift["ds_cfg_strength"]), rescale=float(rift.get("cfg_rescale", 0.7)),
                robust_f0=int(rift["robust_f0"]),
            )
            output_audio = self._stitch_fixed(mono, values)

        if sample_rate != 44100:
            output_audio = self.librosa.resample(output_audio, orig_sr=44100, target_sr=int(sample_rate))[:frames]
            output_audio = self.np.pad(output_audio, (0, max(0, frames - len(output_audio))))
            if adaptive:
                output_audio = quality.limit_peak(output_audio)
        if not self.np.isfinite(output_audio).all():
            raise ValueError("The converted audio contains invalid samples.")
        if frames * 2 + 44 > MAX_OUTPUT_BYTES:
            raise ValueError("The converted audio exceeds the worker limit.")
        temp = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.wav")
        report_path = destination.with_suffix(".qa.json")
        temp_report = temp.with_suffix(".json")
        temp_song_decision = temp.with_suffix(".song-octave.json")
        song_decision = None
        if role == "lead" and isinstance(report, dict):
            report_shift = report.get("song_octave_shift")
            if (
                report.get("song_octave_source") == "automatic"
                and type(report_shift) is int
                and report_shift in {-12, 12}
            ):
                song_decision = {
                    "song_octave_shift": report_shift,
                    "song_octave_source": "automatic",
                }
        try:
            self.sf.write(str(temp), output_audio, sample_rate, subtype="PCM_16", format="WAV")
            if report is not None:
                temp_report.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
                os.replace(temp_report, report_path)
            if song_decision is not None:
                temp_song_decision.write_text(
                    json.dumps(song_decision, ensure_ascii=False, allow_nan=False), encoding="utf-8",
                )
                os.replace(temp_song_decision, song_decision_path)
            os.replace(temp, destination)
        finally:
            temp.unlink(missing_ok=True)
            temp_report.unlink(missing_ok=True)
            temp_song_decision.unlink(missing_ok=True)
        return {"path": str(destination), "sample_rate": sample_rate, "frames": frames}

    def _classify_octave_probe(self, excerpts):
        if self._octave_probe_classifier is None:
            if __package__:
                from .rift_octave_probe import OctaveProbeClassifier
            else:
                from rift_octave_probe import OctaveProbeClassifier
            self._octave_probe_classifier = OctaveProbeClassifier(
                self.runtime_dir / "models/octave_probe",
                self.runtime_dir / "models/phrases/packages",
            )
        prepared = []
        for excerpt in excerpts:
            converted = dict(excerpt)
            try:
                converted["audio16"] = self.np.asarray(
                    self.librosa.resample(
                        self.np.asarray(excerpt["audio"], dtype=self.np.float32),
                        orig_sr=44100,
                        target_sr=16000,
                    ),
                    dtype=self.np.float32,
                )
            except Exception:
                converted["audio16"] = self.np.empty(0, dtype=self.np.float32)
            prepared.append(converted)
        return self._octave_probe_classifier.classify(prepared)

    @contextmanager
    def _preserve_probe_rng(self):
        """Restore process RNG streams after shadow-only candidate inference."""
        python_state = random.getstate()
        numpy_state = self.np.random.get_state()
        torch_cpu_state = self.torch.get_rng_state()
        active_device = next(self.bundle[0].parameters()).device
        torch_cuda_state = None
        if active_device.type == "cuda":
            torch_cuda_state = self.torch.cuda.get_rng_state(active_device)
        try:
            yield
        finally:
            random.setstate(python_state)
            self.np.random.set_state(numpy_state)
            self.torch.set_rng_state(torch_cpu_state)
            if torch_cuda_state is not None:
                self.torch.cuda.set_rng_state(torch_cuda_state, active_device)

    def _warp_contentvec_audio(self, audio16, ratio: float):
        """Warp only the waveform passed to ContentVec, never RIFT's pitch path."""
        try:
            ratio = float(ratio)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("The requested ContentVec formant ratio is invalid.") from None
        if not self.np.isfinite(ratio) or not 0.82 <= ratio <= 1.22:
            raise ValueError("The requested ContentVec formant ratio is outside the supported range.")
        if ratio == 1.0:
            return audio16
        if __package__:
            from .rift_domain import warp_formants
        else:
            from rift_domain import warp_formants

        is_tensor = bool(self.torch.is_tensor(audio16))
        if is_tensor:
            tensor = audio16
            try:
                source = tensor.detach().float().cpu().numpy()
            except Exception as exc:
                raise ValueError("The ContentVec waveform could not be prepared for formant alignment.") from exc
        else:
            tensor = None
            source = self.np.asarray(audio16)
        if source.ndim == 1:
            rows, layout = source[None, :], "mono"
        elif source.ndim == 2:
            rows, layout = source, "batch"
        elif source.ndim == 3 and source.shape[1] == 1:
            rows, layout = source[:, 0, :], "channel"
        else:
            raise ValueError("The ContentVec waveform has an unsupported shape for formant alignment.")
        if not self.np.isfinite(source).all():
            raise ValueError("The ContentVec waveform contains invalid samples.")

        warped_rows = []
        for row in rows:
            original = self.np.asarray(row, dtype=self.np.float32).reshape(-1)
            try:
                warped = self.np.asarray(warp_formants(original, ratio, 16000), dtype=self.np.float32).reshape(-1)
            except Exception as exc:
                raise ValueError("The ContentVec formant transform failed.") from exc
            if len(warped) != len(original) or not self.np.isfinite(warped).all():
                raise ValueError("The ContentVec formant transform returned invalid samples.")
            rms = float(self.np.sqrt(self.np.mean(original.astype(self.np.float64) ** 2))) if len(original) else 0.0
            if rms > 1e-8:
                difference = float(self.np.sqrt(self.np.mean((warped.astype(self.np.float64) - original) ** 2)))
                if difference <= max(1e-9, rms * 1e-6):
                    raise ValueError("The ContentVec formant transform left non-silent audio unchanged.")
            warped_rows.append(warped)
        stacked = self.np.stack(warped_rows) if warped_rows else rows.astype(self.np.float32, copy=True)
        if layout == "mono":
            result = stacked[0]
        elif layout == "channel":
            result = stacked[:, None, :]
        else:
            result = stacked
        if is_tensor:
            return self.torch.as_tensor(result, dtype=tensor.dtype, device=tensor.device)
        return result.astype(source.dtype, copy=False)

    def _stitch_fixed(self, mono, values):
        output_audio = self.np.zeros(len(mono), dtype=self.np.float32)
        fade_samples = int(20.0 * 44100 / 1000)
        for index, (start_sample, audio_out, expected_length) in enumerate(values):
            audio_out = self.np.asarray(audio_out, dtype=self.np.float32)
            audio_out = audio_out[:expected_length]
            if len(audio_out) < expected_length:
                audio_out = self.np.pad(audio_out, (0, expected_length - len(audio_out)))
            if index > 0:
                count = min(fade_samples, len(audio_out), len(output_audio) - start_sample)
                if count > 0:
                    audio_out = self.infer.apply_fade(audio_out.copy(), count, fade_in=True)
                    output_audio[start_sample : start_sample + count] *= self.np.linspace(1, 0, count)
            if index < len(values) - 1 and len(audio_out):
                count = min(fade_samples, len(audio_out))
                audio_out[-count:] *= self.np.linspace(1, 0, count)
            end = min(len(output_audio), start_sample + len(audio_out))
            output_audio[start_sample:end] += audio_out[: end - start_sample]
        return output_audio


def _response(stdout, payload: dict[str, Any]) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    stdout.flush()


def serve(source: Path, runtime_dir: Path) -> int:
    protocol_stdout = sys.stdout
    try:
        runtime_root = runtime_dir.resolve(strict=True)
        service = _Rift(source, runtime_root)
        startup_line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if len(startup_line) > MAX_REQUEST_BYTES:
            _response(protocol_stdout, {"ok": False, "code": "request_too_large", "message": "Worker request too large."})
            return 2
        startup = json.loads(startup_line)
        if startup.get("cmd") != "start" or set(startup) != {"cmd", "settings"}:
            raise ValueError("Invalid worker startup request.")
        settings = startup["settings"]
        if set(settings) != {"checkpoint", "device"}:
            raise ValueError("Invalid worker startup settings.")
        service.load(str(settings["checkpoint"]), str(settings["device"]))
        _response(protocol_stdout, {"ok": True, "ready": True, "speakers": service.speakers()})
        while True:
            raw = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
            if not raw:
                break
            if len(raw) > MAX_REQUEST_BYTES:
                _response(protocol_stdout, {"ok": False, "code": "request_too_large", "message": "Worker request too large."})
                break
            try:
                request = json.loads(raw)
                if not isinstance(request, dict) or set(request) - {"cmd", "role", "rift", "input", "output"}:
                    raise ValueError("Invalid worker request.")
                command = request.get("cmd")
                if command == "speakers" and set(request) == {"cmd"}:
                    result = {"speakers": service.speakers()}
                elif command == "convert" and set(request) == {"cmd", "role", "rift", "input", "output"}:
                    result = service.convert(request)
                elif command == "close" and set(request) == {"cmd"}:
                    _response(protocol_stdout, {"ok": True, "closed": True})
                    break
                else:
                    raise ValueError("Invalid worker command.")
                _response(protocol_stdout, {"ok": True, **result})
            except Exception as exc:
                code = "invalid_request" if isinstance(exc, (ValueError, TypeError, KeyError)) else "inference_failed"
                # Never return exception text, checkpoint paths, model data, or
                # upstream diagnostics to the host protocol.
                _response(protocol_stdout, {"ok": False, "code": code, "message": "RIFT worker operation failed."})
    except Exception:
        _response(protocol_stdout, {"ok": False, "code": "startup_failed", "message": "RIFT worker could not start."})
        return 1
    finally:
        try:
            sys.stdout = protocol_stdout
        except Exception:
            pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", required=True)
    parser.add_argument("--runtime-dir", required=True)
    args = parser.parse_args()
    try:
        return serve(Path(args.source), Path(args.runtime_dir))
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
