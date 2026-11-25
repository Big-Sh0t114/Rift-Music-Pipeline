"""Signal preparation, adaptive RIFT inference and per-stem quality checks."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pyloudnorm
from scipy import ndimage, signal

if __package__:
    from . import rift_timbre as timbre
else:
    import rift_timbre as timbre


RATE = 44100
MAX_CHUNK = 15 * RATE
OVERLAP = RATE // 2
MIN_MODEL_SAMPLES = RATE // 4
LOSS_LIMIT = 0.15
FLOOR = 10 ** (-60 / 20)
PEAK_LIMIT = 10 ** (-1 / 20)


@dataclass(frozen=True)
class Guidance:
    robust_f0: int
    speaker: float
    detail: float
    rescale: float


@dataclass(frozen=True)
class InferencePlan:
    guidance: Guidance
    shift: int
    alpha: float = 0.0
    target_loudness: float = -18.0


def rms(samples):
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))


def downmix(channels):
    mono = channels.mean(axis=1, dtype=np.float32)
    if channels.shape[1] > 1:
        strongest = max(channels.T, key=rms)
        if rms(mono) < 0.25 * rms(strongest):
            return strongest.copy()
    return mono


def limit_peak(samples):
    peak = float(np.max(np.abs(samples)))
    return samples * min(1.0, PEAK_LIMIT / max(peak, FLOOR))


def condition_vocal(samples):
    highpass = signal.butter(2, 80, fs=RATE, btype="highpass", output="sos").astype(np.float32)
    filtered = signal.sosfilt(highpass, samples)
    envelope = np.sqrt(np.maximum(0, ndimage.uniform_filter1d(filtered ** 2, size=RATE // 50)))
    # Compress above -20 dBFS. Smooth gain across syllables and preserve timing.
    gain = np.maximum(1.0, envelope / 0.1) ** (1.0 / 2.5 - 1.0)
    filtered *= ndimage.uniform_filter1d(gain, size=RATE // 12)
    meter = pyloudnorm.Meter(RATE, block_size=0.1)
    loudness = meter.integrated_loudness(np.pad(filtered, (0, max(0, MIN_MODEL_SAMPLES - len(filtered)))))
    if np.isfinite(loudness):
        filtered *= 10 ** (np.clip(-18.0 - loudness, -18.0, 18.0) / 20)
    return limit_peak(filtered), meter


def chunk_bounds(segments):
    """Keep slicer's gaps and split long vocal runs with 500 ms of overlap."""
    for start, chunk in segments:
        end = start + len(chunk)
        if end <= start:
            continue
        stride = MAX_CHUNK - OVERLAP
        count = max(1, (len(chunk) - OVERLAP + stride - 1) // stride)
        for index in range(count):
            left = start + index * (len(chunk) - OVERLAP) // count if count > 1 else start
            right = start + (index + 1) * (len(chunk) - OVERLAP) // count + OVERLAP if count > 1 else end
            yield left, right


def choose_shift(f0_tracks, pitch_range, manual_shift):
    if not pitch_range:
        return manual_shift, None
    voiced = np.concatenate([f0[(confidence >= 0.3) & (f0 > 0)] for f0, confidence in f0_tracks])
    if len(voiced) < 10:
        return manual_shift, None
    midi = 69 + 12 * np.log2(voiced / 440)
    low, median, high = pitch_range
    source_median = float(np.median(midi))

    def score(shift):
        coverage = float(np.mean((low <= midi + shift) & (midi + shift <= high)))
        return coverage, shift == 0, -abs(source_median + shift - median)

    shift = manual_shift or max((0, -12, 12), key=score)
    return shift, 1.0 - score(shift)[0]


def octave_plan(bounds, f0_tracks, centroids, pitch_range, manual_shift):
    labels = np.zeros(len(bounds), dtype=np.intp)
    if pitch_range and not manual_shift:
        midi_tracks = [69 + 12 * np.log2(f0[(confidence >= 0.3) & (f0 > 0)] / 440) for f0, confidence in f0_tracks]
        labels = timbre.pitch_clusters(bounds, midi_tracks, centroids)
    shifts = {}
    outside = []
    for label in np.unique(labels):
        members = [f0 for f0, group in zip(f0_tracks, labels, strict=True) if group == label]
        shift, ratio = choose_shift(members, pitch_range, manual_shift)
        shifts[int(label)] = shift
        if ratio is not None:
            count = sum(np.count_nonzero((f0 > 0) & (confidence >= 0.3)) for f0, confidence in members)
            outside.append((ratio, count))
    ratio = float(sum(value * count for value, count in outside) / sum(count for _, count in outside)) if outside else None
    return labels, shifts, ratio


def route(f0, confidence, level, settings):
    voiced = f0 > 0
    adjacent = voiced[:-1] & voiced[1:]
    jumps = np.abs(12 * np.log2(f0[1:][adjacent] / f0[:-1][adjacent]))
    jitter = float(np.mean(jumps > 1.5)) if len(jumps) else 0.0
    unstable = np.count_nonzero(voiced) >= 10 and (jitter > 0.08 or float(np.mean(confidence[voiced])) < 0.68)
    kind = "harsh" if unstable else "soft" if level < -24 else "clean"
    robust, speaker, detail, rescale = {
        "harsh": (2, 0.45, 0.0, 0.75),
        "soft": (1, 0.50, 0.05, 0.70),
        "clean": (1, 0.55, 0.10, 0.70),
    }[kind]
    return kind, Guidance(
        max(settings["robust_f0"], robust),
        min(settings["spk_cfg_strength"], speaker),
        min(settings["ds_cfg_strength"], detail),
        max(settings["cfg_rescale"], rescale),
    )


def band_fraction(samples):
    freqs, density = signal.welch(samples, fs=RATE, nperseg=min(2048, len(samples)))
    return float(density[(freqs >= 4000) & (freqs <= 8000)].sum() / max(1e-12, density.sum()))


def high_band_excess(source, output):
    return float(10 * np.log10((band_fraction(output) + 1e-5) / (band_fraction(source) + 1e-5)))


def tame_high_band(source, output, excess):
    if excess <= 6 or len(output) < RATE // 12 or band_fraction(output) < 0.025:
        return output, excess, 0.0
    cut_db = min(6.0, excess - 6)
    bandpass = signal.butter(2, (4000, 8000), fs=RATE, btype="bandpass", output="sos")
    upper_band = signal.sosfiltfilt(bandpass, output).astype(np.float32)
    adjusted = output - (1 - 10 ** (-cut_db / 20)) * upper_band
    return adjusted, high_band_excess(source, adjusted), cut_db


def measure(source, output, reference_f0, pitch):
    if len(output) != len(source) or not np.isfinite(output).all() or rms(output) < FLOOR / 10:
        return 1.0, None, False
    reference_voiced = reference_f0 > 0
    loss = None
    if np.count_nonzero(reference_voiced) >= 10:
        f0, _ = pitch(output)
        voiced = np.zeros_like(reference_voiced)
        count = min(len(voiced), len(f0))
        voiced[:count] = f0[:count] > 0
        # Allow 30 ms of natural model/vocoder timing variation.
        nearby = ndimage.maximum_filter1d(voiced, size=7, mode="constant")
        loss = float(np.mean(~nearby[reference_voiced]))
    return loss, high_band_excess(source, output), True


def stitch(total, bounds, outputs):
    canvas = np.zeros(total, dtype=np.float32)
    for index, ((start, end), output) in enumerate(zip(bounds, outputs, strict=True)):
        left = max(0, bounds[index - 1][1] - start) if index else 0
        right = max(0, end - bounds[index + 1][0]) if index + 1 < len(bounds) else 0
        # Complementary Hann ramps sum to one. Short edge fades in gaps do
        # not touch a contiguous neighbouring slice.
        window = np.ones(end - start, dtype=np.float32)
        if left:
            window[:left] = np.sin(np.linspace(0, np.pi / 2, left)) ** 2
        elif not index or bounds[index - 1][1] < start:
            fade = min(RATE // 200, len(window) // 2)
            window[:fade] = np.linspace(0, 1, fade)
        if right:
            window[-right:] = np.cos(np.linspace(0, np.pi / 2, right)) ** 2
        elif index + 1 == len(bounds) or end < bounds[index + 1][0]:
            fade = min(RATE // 200, len(window) // 2)
            if fade:
                window[-fade:] = np.linspace(1, 0, fade)
        canvas[start:end] += window * output
    return canvas


def restore_dynamics(source, output):
    def envelope(samples):
        return np.sqrt(np.maximum(0, ndimage.uniform_filter1d(samples ** 2, size=round(RATE * 0.4))))

    dry_rms, converted_rms = envelope(source), envelope(output)
    ratio = (dry_rms + FLOOR) / (converted_rms + FLOOR)
    gain = np.clip(ratio, 10 ** (-6 / 20), 10 ** (6 / 20))
    gain[dry_rms < FLOOR] = np.minimum(1, gain[dry_rms < FLOOR])
    return limit_peak(output * ndimage.uniform_filter1d(gain, size=round(RATE * 0.08)))


def polish_vocal(samples):
    # Subtract a small contribution around 350 Hz to soften boxiness.
    b, a = signal.iirpeak(350, Q=1.2, fs=RATE)
    midrange = signal.lfilter(b.astype(np.float32), a.astype(np.float32), samples)
    return samples - (1 - 10 ** (-1.5 / 20)) * midrange


def profile_chunk(source, spectrum, pitch, profile, meter, guidance, settings):
    if spectrum is None:
        return source, guidance, 0.0, -18.0, {}
    matched, drift, maximum_eq = timbre.match_spectrum(source, spectrum, profile["ltas_db"])
    loudness = timbre.voiced_loudness(matched, pitch, meter)
    if loudness is not None:
        matched *= 10 ** (np.clip(-18.0 - loudness, -18, 18) / 20)
        matched = limit_peak(matched)
    padded = np.pad(matched, (0, max(0, MIN_MODEL_SAMPLES - len(matched))))
    integrated = meter.integrated_loudness(padded)
    if not np.isfinite(integrated):
        return source, guidance, 0.0, -18.0, {}
    # Quantize to 0.5 dB so compatible chunks can still share an inference batch.
    # This prevents upstream's own loudness normalization from undoing voiced gain.
    target = round(integrated * 2) / 2 if loudness is not None else -18.0
    alpha = round(float(np.interp(drift, [5, 17], [0.30, 0.50])) * 20) / 20
    if drift >= timbre.DRIFT_THRESHOLD_DB:
        severity = min((drift - timbre.DRIFT_THRESHOLD_DB) / 10, 1.0)
        speaker = max(settings["spk_cfg_strength"], 0.70 + 0.15 * severity)
        guidance = Guidance(guidance.robust_f0, round(min(speaker, 0.85), 2),
                            round(min(guidance.detail, 0.10 * (1 - severity)), 2),
                            min(0.75, max(guidance.rescale, 0.70)))
    return matched, guidance, alpha, target, {
        "ltas_drift_db": round(drift, 3), "ltas_max_eq_db": round(maximum_eq, 3),
        "voiced_loudness_lufs": timbre.voiced_loudness(matched, pitch, meter),
        "timbre_route": "high_drift" if drift >= timbre.DRIFT_THRESHOLD_DB else "regular",
    }


def convert_adaptive(mono, slice_vocals, pitch, run_inference, settings, *, phrase_models=None, content_features=None):
    if settings.get("timbre_slicing", False):
        if __package__:
            from .rift_phrase_quality import convert_phrases
        else:
            from rift_phrase_quality import convert_phrases
        return convert_phrases(mono, pitch, run_inference, settings, phrase_models, content_features)

    conditioned, meter = condition_vocal(mono)
    # Relative threshold keeps quiet vocals while still detecting long rests.
    frames = conditioned[:len(conditioned) // 441 * 441].reshape(-1, 441)
    levels = np.sqrt(np.mean(frames ** 2, axis=1))
    audible = levels[levels > FLOOR]
    threshold = -58.0
    if len(audible):
        reference_db = 20 * np.log10(np.percentile(audible, 75))
        threshold = float(np.clip(reference_db - 30, -58, -48))
    profile = settings.get("voice_profile")
    pitch_range = settings["pitch_range"] or (profile["pitch_range"] if profile else [])
    sections = slice_vocals(conditioned, threshold)
    bounds, sources, f0_tracks, centroids = [], [], [], []
    routes, options, timbre_stats = [], [], []
    for start, end in chunk_bounds(sections):
        source = conditioned[start:end]
        padded = np.pad(source, (0, max(0, MIN_MODEL_SAMPLES - len(source))))
        if not np.isfinite(meter.integrated_loudness(padded)):
            continue
        f0, confidence = pitch(source)
        level = 20 * np.log10(max(rms(mono[start:end]), 1e-8))
        kind, guidance = route(f0, confidence, level, settings)
        spectrum = timbre.extract_spectrum(source, (f0, confidence)) if profile or (pitch_range and not settings["key_shift"]) else None
        alpha, loudness, stats = 0.0, -18.0, {}
        if profile:
            matched, guidance, alpha, loudness, stats = profile_chunk(source, spectrum, (f0, confidence), profile, meter, guidance, settings)
        else:
            matched = source
        options.append((guidance, alpha, loudness))
        centroids.append(spectrum.centroid_hz if spectrum else None)
        timbre_stats.append(stats)
        bounds.append((start, end))
        sources.append(np.pad(matched, (0, max(0, MIN_MODEL_SAMPLES - len(matched)))))
        f0_tracks.append((f0, confidence))
        routes.append(kind)
    if not bounds:
        raise RuntimeError("The source contains no audible vocal segments.")

    labels, shifts, outside_range = octave_plan(bounds, f0_tracks, centroids, pitch_range, settings["key_shift"])
    groups = defaultdict(list)
    for i, (guidance, alpha, loudness) in enumerate(options):
        plan = InferencePlan(guidance, shifts[int(labels[i])], alpha, loudness)
        groups[plan].append(i)

    def infer(segments, plan):
        kwargs = {"cvec_alpha": plan.alpha, "target_loudness": plan.target_loudness} if profile else {}
        return run_inference(segments, plan.guidance, plan.shift, **kwargs)

    outputs = [None] * len(bounds)
    checks = [None] * len(bounds)
    for plan, indices in groups.items():
        guidance = plan.guidance
        requests = [(bounds[i][0], sources[i]) for i in indices]
        batch = infer(requests, plan)
        for i, (_, result, _) in zip(indices, batch, strict=True):
            start, end = bounds[i]
            output = np.asarray(result[:end - start], dtype=np.float32)
            source = conditioned[start:end]
            loss, excess, valid = measure(source, output, f0_tracks[i][0], pitch)
            retried = not valid or (loss is not None and loss > LOSS_LIMIT)
            accepted = False
            if retried:
                retry_guidance = Guidance(2, max(0.0, guidance.speaker - 0.15), 0.0, max(0.75, guidance.rescale))
                retry_plan = InferencePlan(retry_guidance, plan.shift, min(0.30, plan.alpha), plan.target_loudness)
                (_, retry_result, _), = infer([(start, sources[i])], retry_plan)
                retry = np.asarray(retry_result[:end - start], dtype=np.float32)
                new_loss, new_excess, new_valid = measure(source, retry, f0_tracks[i][0], pitch)
                if new_valid and (not valid or (new_loss is not None and new_loss < loss)):
                    output, loss, excess, valid = retry, new_loss, new_excess, new_valid
                    accepted = True
            high_band_cut = 0.0
            if valid:
                output, excess, high_band_cut = tame_high_band(source, output, excess)
            else:
                output = np.zeros(end - start, dtype=np.float32)
            outputs[i] = output
            checks[i] = {
                "start_seconds": round(start / RATE, 4), "end_seconds": round(end / RATE, 4),
                "route": routes[i], "voiced_frame_loss": loss, "high_band_excess_db": excess,
                "high_band_cut_db": high_band_cut,
                "retried": retried, "accepted_retry": accepted, "valid": valid,
                "pitch_cluster": int(labels[i]), "key_shift": plan.shift,
                **timbre_stats[i],
            }
            if profile:
                applied = retry_plan if accepted else plan
                checks[i].update(cvec_alpha=applied.alpha, speaker_cfg=applied.guidance.speaker,
                                 detail_cfg=applied.guidance.detail, cfg_rescale=applied.guidance.rescale)

    if not any(check["valid"] for check in checks):
        raise RuntimeError("No audible output remained after the RIFT retry.")
    combined = stitch(len(mono), bounds, outputs)
    audio = restore_dynamics(mono, polish_vocal(combined))
    durations = np.array([end - start for start, end in bounds], dtype=np.float64)
    overlaps = [max(0, end - upcoming[0]) / 2 for (_, end), upcoming in zip(bounds, bounds[1:])]
    durations[:-1] -= overlaps
    durations[1:] -= overlaps
    harsh_ratio = float(durations[np.array(routes) == "harsh"].sum() / durations.sum())
    return audio, {
        "key_shift": next(iter(shifts.values())) if len(set(shifts.values())) == 1 else None,
        "pitch_clusters": [{"cluster": label, "key_shift": shift} for label, shift in shifts.items()],
        "voice_profile": profile["speaker"] if profile else None,
        "timbre_slicing": False,
        "pitch_outside_range_ratio": outside_range, "harsh_ratio": harsh_ratio,
        "needs_review": bool(harsh_ratio > 0.4 or (outside_range is not None and outside_range > 0.25) or any(
            not check["valid"] or (check["voiced_frame_loss"] or 0) > LOSS_LIMIT
            or (check["high_band_excess_db"] or 0) > 6 for check in checks
        )),
        "chunks": checks,
    }
