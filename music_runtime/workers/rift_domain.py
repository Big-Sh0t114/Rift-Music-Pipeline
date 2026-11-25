"""Conservative pitch-domain and formant planning for RIFT.

This module deliberately contains no gender or speaker-category heuristics. It
uses measured note distributions and explicit formant anchors only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    from .rift_profile import PITCH_HOP, RATE
except ImportError:  # direct execution by the isolated worker
    from rift_profile import PITCH_HOP, RATE


MIN_CONFIDENCE = 0.3
FORMANT_DEAD_ZONE = 0.07
FORMANT_RATIO_MIN = 0.82
FORMANT_RATIO_MAX = 1.22
AUTO_SHIFT_LIMIT = 5
LOW_REGISTER_GAP = 3.0
LOW_REGISTER_MIN_SHARE = 0.5
LOW_REGISTER_MIN_OCCUPANCY = 0.5
LOW_REGISTER_MIN_FRAMES = 100
TAIL_LOWER_WEIGHT = 1.3
TAIL_DEAD_ZONE_SEMITONES = 0.5
MIN_AUTO_IMPROVEMENT = 0.3
BEST_SCORE_TOLERANCE = 0.12
MIN_TRUSTED_FORMANT_FRAMES = 30
MAX_FORMANT_LOG_MAD = 0.25
SONG_OCTAVE_MIN_RELIABLE_PHRASES = 4
SONG_OCTAVE_MIN_PHRASE_FRAMES = 100
SONG_OCTAVE_MIN_PHRASE_OCCUPANCY = 0.40
SONG_OCTAVE_MIN_POOLED_SIDE_SHARE = 0.65
SONG_OCTAVE_MIN_PHRASE_MAJORITY_SHARE = 0.75
SONG_OCTAVE_MIN_SHIFTED_IN_RANGE_SHARE = 0.65
SONG_OCTAVE_MIN_TAIL_IMPROVEMENT = 0.8


@dataclass(frozen=True)
class DomainPlan:
    """A deterministic pitch/formant plan in semitones and a spectral ratio."""

    octave_shift: int
    virtual_lift: int
    formant_ratio: float
    reason: str
    predicted_outside_ratio: float | None = None


@dataclass(frozen=True)
class SongOctavePlan:
    """Evidence for a single final octave shift across a whole song."""

    octave_shift: int
    reason: str
    reliable_phrases: int = 0
    pooled_notes: int = 0
    target_bounds: tuple[float, float] | None = None
    pooled_outside_ratio: float | None = None
    phrase_majority_ratio: float | None = None
    shifted_in_range_ratio: float | None = None
    tail_improvement: float | None = None


def _valid_track_notes(track) -> tuple[np.ndarray, int]:
    if not isinstance(track, (tuple, list)) or len(track) != 2:
        return np.empty(0, dtype=np.float64), 0
    try:
        f0 = np.asarray(track[0], dtype=np.float64).reshape(-1)
        confidence = np.asarray(track[1], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float64), 0
    frame_count = min(len(f0), len(confidence))
    f0, confidence = f0[:frame_count], confidence[:frame_count]
    valid = np.isfinite(f0) & np.isfinite(confidence) & (f0 > 0) & (confidence >= MIN_CONFIDENCE)
    notes = 69.0 + 12.0 * np.log2(f0[valid] / 440.0) if np.any(valid) else np.empty(0, dtype=np.float64)
    return notes, frame_count


def _song_target_bounds(pitch_range, pitch_tails) -> tuple[float, float] | None:
    try:
        tails = np.asarray(pitch_tails, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        tails = np.empty(0, dtype=np.float64)
    if (len(tails) == 2 and np.isfinite(tails).all() and 0 <= tails[0] < tails[1] <= 127):
        return float(tails[0]), float(tails[1])
    bounds = _usable_range(pitch_range)
    if bounds is None:
        return None
    # V1 profiles contain P10/P50/P90 but no explicit P05/P95 tails.
    return max(0.0, bounds[0] - 2.0), min(127.0, bounds[2] + 2.0)


def plan_song_octave(tracks, pitch_range, pitch_tails=None) -> SongOctavePlan:
    """Conservatively detect a consistent whole-song octave mismatch.

    Every confidence-valid note contributes to the pooled direction, shifted
    target coverage, and tail-loss comparison. Reliable natural phrases also
    need to agree, so a short or sparse phrase cannot create an octave lock.
    """
    bounds = _song_target_bounds(pitch_range, pitch_tails)
    if bounds is None:
        return SongOctavePlan(0, "song_octave_target_range_missing")
    low, high = bounds

    pooled: list[np.ndarray] = []
    reliable: list[np.ndarray] = []
    for track in tracks if tracks is not None else ():
        notes, frame_count = _valid_track_notes(track)
        if len(notes):
            pooled.append(notes)
            if (len(notes) >= SONG_OCTAVE_MIN_PHRASE_FRAMES
                    and len(notes) / max(1, frame_count) >= SONG_OCTAVE_MIN_PHRASE_OCCUPANCY):
                reliable.append(notes)

    if len(reliable) < SONG_OCTAVE_MIN_RELIABLE_PHRASES:
        return SongOctavePlan(0, "song_octave_insufficient_reliable_phrases",
                              reliable_phrases=len(reliable),
                              pooled_notes=sum(len(notes) for notes in pooled), target_bounds=bounds)
    if not pooled:
        return SongOctavePlan(0, "song_octave_no_confidence_valid_pitch",
                              reliable_phrases=len(reliable), target_bounds=bounds)

    notes = np.concatenate(pooled)
    below = notes < low
    above = notes > high
    low_share = float(np.mean(below))
    high_share = float(np.mean(above))
    low_phrase_share = float(np.mean([np.mean(phrase < low) > 0.5 for phrase in reliable]))
    high_phrase_share = float(np.mean([np.mean(phrase > high) > 0.5 for phrase in reliable]))

    if (low_share >= SONG_OCTAVE_MIN_POOLED_SIDE_SHARE
            and low_phrase_share >= SONG_OCTAVE_MIN_PHRASE_MAJORITY_SHARE):
        shift, outside_share, phrase_share = 12, low_share, low_phrase_share
    elif (high_share >= SONG_OCTAVE_MIN_POOLED_SIDE_SHARE
          and high_phrase_share >= SONG_OCTAVE_MIN_PHRASE_MAJORITY_SHARE):
        shift, outside_share, phrase_share = -12, high_share, high_phrase_share
    else:
        return SongOctavePlan(0, "song_octave_mixed_register", len(reliable), len(notes), bounds,
                              max(low_share, high_share), max(low_phrase_share, high_phrase_share))

    shifted = notes + shift
    in_range_share = float(np.mean((shifted >= low) & (shifted <= high)))
    baseline_tail = _tail_score(notes, low, high, 0)
    shifted_tail = _tail_score(notes, low, high, shift)
    tail_improvement = baseline_tail - shifted_tail
    if in_range_share < SONG_OCTAVE_MIN_SHIFTED_IN_RANGE_SHARE:
        return SongOctavePlan(0, "song_octave_shift_fails_target_coverage", len(reliable), len(notes), bounds,
                              outside_share, phrase_share, in_range_share, tail_improvement)
    if tail_improvement < SONG_OCTAVE_MIN_TAIL_IMPROVEMENT:
        return SongOctavePlan(0, "song_octave_tail_improvement_below_floor", len(reliable), len(notes), bounds,
                              outside_share, phrase_share, in_range_share, tail_improvement)
    return SongOctavePlan(shift, "song_octave_automatic", len(reliable), len(notes), bounds,
                          outside_share, phrase_share, in_range_share, tail_improvement)


def _valid_pitch_frames(tracks: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, int]:
    values: list[np.ndarray] = []
    total_frames = 0
    for track in tracks:
        notes, count = _valid_track_notes(track)
        total_frames += count
        if len(notes):
            values.append(notes)
    return (np.concatenate(values) if values else np.empty(0, dtype=np.float64)), total_frames


def _usable_range(pitch_range) -> tuple[float, float, float] | None:
    try:
        values = np.asarray(pitch_range, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(values) != 3 or not np.isfinite(values).all():
        return None
    if values[0] >= values[2] or values[0] > values[1] or values[1] > values[2]:
        return None
    if values[0] < 0 or values[2] > 127:
        return None
    return float(values[0]), float(values[1]), float(values[2])


def _tail_score(notes: np.ndarray, low: float, high: float, shift: int) -> float:
    shifted = notes + shift
    lower = np.maximum(0.0, low - TAIL_DEAD_ZONE_SEMITONES - shifted)
    upper = np.maximum(0.0, shifted - high - TAIL_DEAD_ZONE_SEMITONES)
    return float(np.mean(TAIL_LOWER_WEIGHT * lower + upper))


def _outside_ratio(notes: np.ndarray, low: float, high: float, shift: int) -> float:
    shifted = notes + shift
    return float(np.mean((shifted < low) | (shifted > high)))


def _supports_low_register_octave(notes, low, high, pitch_tails, occupancy):
    if occupancy < LOW_REGISTER_MIN_OCCUPANCY or len(notes) < LOW_REGISTER_MIN_FRAMES:
        return False
    lower, center, upper = np.percentile(notes, [10, 50, 90])
    if lower > low - LOW_REGISTER_GAP or np.mean(notes < low) < LOW_REGISTER_MIN_SHARE:
        return False

    # The raised phrase must clear the lower range with room for its median
    # and upper notes. V1 and explicit pitch ranges use a P95 estimate.
    upper_tail = high + 2.0 if pitch_tails is None else pitch_tails[1]
    return lower + 12 >= low and center + 12 <= upper_tail + 0.5 and upper + 12 <= upper_tail + 5.0


def supports_cluster_octave(track, pitch_range, shift, occupancy):
    """Check the proposed cluster octave against the individual phrase."""
    bounds = _usable_range(pitch_range)
    if shift not in (-12, 12) or occupancy < 0.2 or bounds is None:
        return False
    notes, _ = _valid_pitch_frames([track])
    if len(notes) < 30:
        return False
    low, _, high = bounds
    outside = notes < low if shift > 0 else notes > high
    return (np.mean(outside) >= 0.5
            and _tail_score(notes, low, high, 0) - _tail_score(notes, low, high, shift) >= MIN_AUTO_IMPROVEMENT)


def _formant_ratio(source_anchor, target_anchor, protect_formants: bool) -> float:
    try:
        source = float(source_anchor)
        target = float(target_anchor)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    # A scalar outside the plausible voiced F1-F3 geometric-mean range is not
    # sufficiently trustworthy to drive a timbre transform.
    if not (math.isfinite(source) and math.isfinite(target)) or not (250.0 <= source <= 3500.0 and 250.0 <= target <= 3500.0):
        return 1.0
    delta = target / source - 1.0
    if abs(delta) <= FORMANT_DEAD_ZONE:
        return 1.0
    damping = 0.35 if protect_formants else 0.7
    adjusted = math.copysign(abs(delta) - FORMANT_DEAD_ZONE, delta) * damping
    return float(np.clip(1.0 + adjusted, FORMANT_RATIO_MIN, FORMANT_RATIO_MAX))


def summarize_formants(frames) -> tuple[float | None, int, float | None]:
    """Summarize geometric anchors with the shared robust log-domain estimator."""
    try:
        values = np.asarray(frames, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None, 0, None
    values = values[np.isfinite(values) & (values >= 250.0) & (values <= 3500.0)]
    count = int(len(values))
    if not count:
        return None, 0, None
    log_values = np.log(values)
    median_log = float(np.median(log_values))
    log_mad = float(np.median(np.abs(log_values - median_log)))
    anchor = math.exp(median_log) if count >= MIN_TRUSTED_FORMANT_FRAMES and log_mad <= MAX_FORMANT_LOG_MAD else None
    return anchor, count, log_mad


def plan_domain(
    tracks: list[tuple[np.ndarray, np.ndarray]],
    pitch_range,
    manual_shift: int,
    *,
    virtual_available: bool,
    allow_octave: bool,
    voiced_share: float,
    source_anchor,
    target_anchor,
    protect_formants: bool,
    pitch_tails=None,
    allow_phrase_octave: bool = False,
) -> DomainPlan:
    """Plan conservative pitch alignment and formant scaling.

    ``octave_shift`` is the intentional final output shift: a nonzero explicit
    manual shift, or a severe and explicitly permitted octave correction. A
    automatic ``virtual_lift`` is a reversible model-domain support
    transform; the worker compensates it so the requested musical pitch remains.
    """
    if type(manual_shift) is not int:
        raise TypeError("manual_shift must be an integer")
    formant_ratio = _formant_ratio(source_anchor, target_anchor, protect_formants)
    notes, _ = _valid_pitch_frames(tracks)
    try:
        occupancy = float(voiced_share)
    except (TypeError, ValueError, OverflowError):
        occupancy = 0.0
    if not math.isfinite(occupancy):
        occupancy = 0.0
    occupancy = min(1.0, max(0.0, occupancy))

    pitch_bounds = _usable_range(pitch_range)
    if manual_shift == 0 and allow_phrase_octave and pitch_bounds is not None:
        low, _, high = pitch_bounds
        if _supports_low_register_octave(notes, low, high, pitch_tails, occupancy):
            return DomainPlan(12, 0, formant_ratio, "sustained_low_register_octave", _outside_ratio(notes, low, high, 12))
    enough_notes = len(notes) >= 10
    auto_allowed = occupancy >= 0.20 and enough_notes and pitch_bounds is not None
    final_shift = manual_shift if manual_shift != 0 else 0
    virtual_lift = 0
    predicted_outside = None
    baseline_score = None
    best_virtual_score = None
    reason = "no_pitch_domain"
    if enough_notes and pitch_bounds is not None:
        baseline_score = _tail_score(notes, pitch_bounds[0], pitch_bounds[2], 0)
        best_virtual_score = baseline_score

    if manual_shift != 0:
        if enough_notes and pitch_bounds is not None:
            low, _, high = pitch_bounds
            predicted_outside = _outside_ratio(notes, low, high, manual_shift)
        reason = "manual_shift"
    elif auto_allowed:
        low, _, high = pitch_bounds
        if virtual_available:
            candidate_scores = {
                lift: _tail_score(notes, low, high, lift)
                for lift in range(-AUTO_SHIFT_LIMIT, AUTO_SHIFT_LIMIT + 1)
            }
            best_virtual_score = min(candidate_scores.values())
            if baseline_score - best_virtual_score >= MIN_AUTO_IMPROVEMENT:
                useful = [lift for lift, score in candidate_scores.items()
                          if score <= best_virtual_score + BEST_SCORE_TOLERANCE
                          and baseline_score - score >= MIN_AUTO_IMPROVEMENT]
                if useful:
                    virtual_lift = min(useful, key=lambda lift: (abs(lift), candidate_scores[lift], lift))
                    reason = "pitch_tails_improved"
            if not virtual_lift:
                reason = "pitch_improvement_below_floor"
        else:
            best_virtual_score = baseline_score
            reason = "virtual_lift_unavailable"
        predicted_outside = _outside_ratio(notes, low, high, virtual_lift)

    # Final audible octave movement is strictly opt-in and must materially beat
    # the best reversible support lift to avoid an unneeded octave jump.
    if (manual_shift == 0 and enough_notes and pitch_bounds is not None
            and allow_octave and occupancy >= 0.18):
        low, _, high = pitch_bounds
        source_median = float(np.median(notes))
        gap = low - source_median if source_median < low else source_median - high if source_median > high else 0.0
        if gap > 5.5:
            octave = 12 if source_median < low else -12
            octave_score = _tail_score(notes, low, high, octave)
            comparison_score = best_virtual_score
            if comparison_score is not None and comparison_score - octave_score >= MIN_AUTO_IMPROVEMENT:
                final_shift = octave
                virtual_lift = 0
                predicted_outside = _outside_ratio(notes, low, high, octave)
                reason = "structural_octave_shift"

    return DomainPlan(final_shift, virtual_lift, formant_ratio, reason, predicted_outside)


def formant_frames(samples, track) -> np.ndarray:
    """Return valid voiced F1-F3 geometric-mean anchors from Praat Burg.

    ``samples`` are mono float audio at 44.1 kHz. ``track`` is the RMVPE
    ``(f0_hz, confidence)`` pair on the shared 10 ms RIFT hop. Praat is imported
    lazily so ordinary runtime imports do not load or require it.
    """
    try:
        audio = np.asarray(samples, dtype=np.float64).reshape(-1)
        f0 = np.asarray(track[0], dtype=np.float64).reshape(-1)
        confidence = np.asarray(track[1], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, IndexError):
        return np.empty(0, dtype=np.float64)
    if not len(audio) or not np.isfinite(audio).all() or not len(f0) or not len(confidence):
        return np.empty(0, dtype=np.float64)
    count = min(len(f0), len(confidence), len(audio) // PITCH_HOP + 1)
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    try:
        import parselmouth

        sound = parselmouth.Sound(audio, sampling_frequency=RATE)
        formants = sound.to_formant_burg(
            time_step=PITCH_HOP / RATE,
            max_number_of_formants=5,
            maximum_formant=5500.0,
            window_length=0.025,
            pre_emphasis_from=50.0,
        )
        anchors = []
        for index in range(count):
            pitch_value = f0[index]
            confidence_value = confidence[index]
            if not (math.isfinite(pitch_value) and pitch_value > 0
                    and math.isfinite(confidence_value) and confidence_value >= MIN_CONFIDENCE):
                continue
            time = index * PITCH_HOP / RATE
            values = [float(formants.get_value_at_time(number, time)) for number in (1, 2, 3)]
            if not all(math.isfinite(value) for value in values):
                continue
            f1, f2, f3 = values
            if not (200.0 <= f1 <= 1200.0 and 500.0 <= f2 <= 3500.0 and 1200.0 <= f3 <= 5000.0):
                continue
            if not (f1 < f2 < f3):
                continue
            anchors.append(math.exp(sum(math.log(value) for value in values) / 3.0))
        return np.asarray(anchors, dtype=np.float64)
    except Exception:
        # Praat may fail for a short, silent, or otherwise unusable segment.
        # Formant alignment is optional, so expose this as no usable evidence.
        return np.empty(0, dtype=np.float64)


def warp_formants(samples, ratio: float, sample_rate: int) -> np.ndarray:
    """Move the spectral envelope while preserving pitch, duration and shape.

    Used on 16 kHz ContentVec input only. Any unreliable transform returns an
    unchanged float32 copy, allowing the inference caller to proceed safely.
    """
    original = np.asarray(samples, dtype=np.float32).reshape(-1)
    fallback = original.copy()
    try:
        ratio = float(ratio)
        sample_rate = int(sample_rate)
        if (not math.isfinite(ratio) or not FORMANT_RATIO_MIN <= ratio <= FORMANT_RATIO_MAX
                or sample_rate != 16000 or len(original) < 256 or not np.isfinite(original).all()):
            return fallback
        if abs(ratio - 1.0) < 1e-6:
            return fallback
        from scipy import ndimage, signal

        nperseg = min(1024, 2 ** int(math.floor(math.log2(len(original)))))
        nperseg = max(256, nperseg)
        noverlap = nperseg * 3 // 4
        nfft = max(2048, nperseg * 2)
        frequencies, times, spectrum = signal.stft(
            original,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary="zeros",
            padded=True,
        )
        magnitude = np.abs(spectrum)
        # A broad log-frequency envelope captures resonances while leaving the
        # harmonic fine structure and phase in place.
        envelope = ndimage.gaussian_filter1d(magnitude, sigma=max(2.0, nfft * 0.018), axis=0, mode="nearest")
        source_frequency = frequencies / ratio
        warped_envelope = np.empty_like(envelope)
        for frame in range(envelope.shape[1]):
            warped_envelope[:, frame] = np.interp(
                source_frequency, frequencies, envelope[:, frame],
                left=envelope[0, frame], right=envelope[-1, frame],
            )
        residual = magnitude / np.maximum(envelope, 1e-9)
        warped = spectrum * (warped_envelope * residual / np.maximum(magnitude, 1e-9))
        _, result = signal.istft(
            warped,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            input_onesided=True,
            boundary=True,
        )
        result = np.asarray(result[:len(original)], dtype=np.float32)
        if len(result) != len(original) or not np.isfinite(result).all():
            return fallback
        # Match block RMS to avoid introducing an unrelated loudness shift.
        before = float(np.sqrt(np.mean(original.astype(np.float64) ** 2)))
        after = float(np.sqrt(np.mean(result.astype(np.float64) ** 2)))
        if before > 1e-8 and after > 1e-8:
            result *= np.float32(np.clip(before / after, 0.5, 2.0))
        return result.astype(np.float32, copy=False)
    except Exception:
        return fallback


__all__ = ["DomainPlan", "SongOctavePlan", "formant_frames", "plan_domain", "plan_song_octave", "warp_formants"]
