"""Build an offline profile from one speaker's clean training audio."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from rift_profile import BANDS, MAX_PROFILE_BYTES, RATE, encoder_fingerprint, rmvpe_pitch, validate_profile
from rift_domain import formant_frames, summarize_formants


def build_profile(source, dataset, speaker, device_name, max_files):
    import librosa
    import numpy as np
    import pyloudnorm
    import soundfile
    import torch

    from rift_quality import MAX_CHUNK, downmix, limit_peak
    from rift_timbre import extract_spectrum, voiced_at

    sys.path.insert(0, str(source))
    from infer import HubertModelWithFinalProj, RMVPE

    files = sorted(path for path in dataset.rglob("*") if path.suffix.casefold() in {".wav", ".flac"} and path.is_file())
    if len(files) > max_files:
        files = [files[int(index)] for index in np.linspace(0, len(files) - 1, max_files)]
    if not files:
        raise ValueError("No WAV or FLAC training audio was found")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu"
                          if device_name == "auto" else device_name)
    use_half = device.type == "cuda"
    model_dir = source / "pretrained"
    rmvpe = RMVPE(model_path=str(model_dir / "rmvpe/model.pt"), hop_length=160, device=device)
    encoder = HubertModelWithFinalProj.from_pretrained(str(model_dir / "content-vec-best")).to(device).eval()
    if use_half:
        encoder.half()
    step = int(np.prod(encoder.config.conv_stride))
    receptive = 1
    stride = 1
    for kernel, hop in zip(encoder.config.conv_kernel, encoder.config.conv_stride, strict=True):
        receptive += (kernel - 1) * stride
        stride *= hop

    meter = pyloudnorm.Meter(RATE, block_size=0.1)
    pitches = []
    formants = []
    weighted_ltas = np.zeros(BANDS, dtype=np.float64)
    spectrum_frames = content_frames = used_files = 0
    channel_sum = channel_squares = 0
    for path in files:
        used = False
        with soundfile.SoundFile(path) as handle:
            rate = handle.samplerate
            for audio in handle.blocks(blocksize=round(MAX_CHUNK * rate / RATE), dtype="float32", always_2d=True):
                mono = downmix(audio)
                if rate != RATE:
                    mono = librosa.resample(mono, orig_sr=rate, target_sr=RATE)
                if len(mono) < RATE // 4 or not np.isfinite(mono).all():
                    continue
                pitch = rmvpe_pitch(mono, rmvpe, torch, device)
                anchors = formant_frames(mono, pitch)
                if len(anchors):
                    formants.append(anchors)
                spectrum = extract_spectrum(mono, pitch)
                loudness = meter.integrated_loudness(mono)
                if spectrum is None or not np.isfinite(loudness):
                    continue
                normalized = limit_peak(mono * 10 ** (np.clip(-18.0 - loudness, -18, 18) / 20))
                audio16 = librosa.resample(normalized, orig_sr=RATE, target_sr=16000)
                tensor = torch.as_tensor(audio16, dtype=torch.float16 if use_half else torch.float32, device=device).unsqueeze(0)
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=use_half):
                    features = encoder(tensor)["last_hidden_state"].squeeze(0).float().cpu().numpy()
                frame_times = (np.arange(len(features)) * step + (receptive - 1) / 2) / 16000
                features = features[voiced_at(frame_times, pitch)].astype(np.float64)
                if not len(features):
                    continue
                channel_sum += features.sum(axis=0)
                channel_squares += (features ** 2).sum(axis=0)
                content_frames += len(features)
                weighted_ltas += spectrum.relative_db * spectrum.active_frames
                spectrum_frames += spectrum.active_frames
                f0 = pitch[0][(pitch[0] > 0) & (pitch[1] >= 0.3)]
                pitches.append(69 + 12 * np.log2(f0 / 440))
                used = True
        used_files += used

    if content_frames < 100 or used_files < 2:
        raise ValueError("The training audio must contain at least 100 voiced ContentVec frames across two files")
    mean = channel_sum / content_frames
    std = np.sqrt(np.maximum(0, channel_squares / content_frames - mean ** 2))
    pitch_values = np.concatenate(pitches)
    formant_anchor, formant_count, formant_mad = summarize_formants(
        np.concatenate(formants) if formants else np.empty(0, dtype=np.float64)
    )
    profile = {
        "version": 2, "speaker": speaker, "encoder_sha256": encoder_fingerprint(source),
        "pitch_range": np.round(np.percentile(pitch_values, [10, 50, 90]), 4).tolist(),
        "pitch_tails": np.round(np.percentile(pitch_values, [5, 95]), 4).tolist(),
        "formant_anchor_hz": round(formant_anchor, 3) if formant_anchor is not None else None,
        "formant_frames": formant_count,
        "formant_log_mad": round(formant_mad, 6) if formant_mad is not None else None,
        "ltas_db": np.round(weighted_ltas / spectrum_frames, 5).tolist(),
        "cvec_mean": np.round(mean, 6).tolist(), "cvec_std": np.round(std, 6).tolist(),
    }
    validate_profile(profile, speaker)
    return profile, used_files, content_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-files", type=int, default=1000)
    args = parser.parse_args()
    if args.output.suffix.casefold() != ".json" or args.output.exists() or args.max_files < 2:
        parser.error("Choose a new JSON destination and a limit of at least two files")
    profile, count, frames = build_profile(args.source, args.input, args.speaker, args.device, args.max_files)
    encoded = json.dumps(profile, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_PROFILE_BYTES:
        raise ValueError("Generated speaker profile exceeds its byte limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encoded)
            stream.close()
            # Publish the snapshot atomically and protect an existing profile.
            os.link(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"speaker": args.speaker, "files": count, "voiced_content_frames": frames,
                      "output": str(args.output), "pitch_range": profile["pitch_range"],
                      "pitch_tails": profile["pitch_tails"], "formant_anchor_hz": profile["formant_anchor_hz"],
                      "formant_frames": profile["formant_frames"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
