"""Profile-based spectral matching, feature alignment, and octave routing."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

if __package__:
    from .rift_profile import BANDS, PITCH_HOP, RATE
else:
    from rift_profile import BANDS, PITCH_HOP, RATE


EDGES = np.geomspace(100, 10000, BANDS + 1)
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])
FFT_SIZE = 4096
FFT_OVERLAP = FFT_SIZE * 3 // 4
DRIFT_THRESHOLD_DB = 8.0


@dataclass
class Spectrum:
    frequencies: np.ndarray
    transform: np.ndarray
    relative_db: np.ndarray
    centroid_hz: float
    active_frames: int
    band_power: np.ndarray


def voiced_at(times, pitch):
    f0, confidence = pitch
    if not len(f0):
        return np.zeros(len(times), dtype=bool)
    indices = np.clip(np.rint(times * RATE / PITCH_HOP).astype(np.intp), 0, len(f0) - 1)
    return (f0[indices] > 0) & (confidence[indices] >= 0.3)


def extract_spectrum(audio, pitch):
    padded = np.pad(audio, (0, max(0, FFT_SIZE - len(audio))))
    freqs, time, transform = signal.stft(padded, fs=RATE, nperseg=FFT_SIZE, noverlap=FFT_OVERLAP)
    magnitude = np.abs(transform)
    frame_energy = np.sum(magnitude ** 2, axis=0)
    active = voiced_at(time, pitch) & (time < len(audio) / RATE)
    active &= frame_energy > max(1e-12, float(frame_energy.max()) * 10 ** (-32 / 10))
    count = int(np.count_nonzero(active))
    if count < 3:
        return None
    mean_power = np.mean(magnitude[:, active] ** 2, axis=1)
    shaping = np.array([mean_power[(freqs >= lower) & (freqs < upper)].mean() for lower, upper in zip(EDGES, EDGES[1:])])
    # A relative floor limits the influence of missing frequency bands and noise.
    levels = 10 * np.log10(np.maximum(shaping, max(1e-14, float(shaping.max()) * 1e-6)))
    centroid = float(np.dot(freqs, mean_power) / max(float(mean_power.sum()), 1e-14))
    return Spectrum(freqs, transform, levels - levels.mean(), centroid, count, shaping)


def match_spectrum(audio, spectrum, reference):
    gains, drift = spectral_gains(spectrum.relative_db, reference)
    return apply_spectrum(audio, spectrum.frequencies, spectrum.transform, gains), drift, float(np.max(np.abs(gains)))


def spectral_gains(source_db, reference):
    delta = ndimage.gaussian_filter1d(np.asarray(reference) - source_db, sigma=1.2)
    drift = float(np.sqrt(np.mean(delta ** 2)))
    return np.clip(delta * 0.75, -6.0, 6.0), drift


def apply_spectrum(audio, freqs, transform, gains):
    band_gain = np.interp(freqs, np.r_[0, 100, CENTERS, 10000, RATE / 2], np.r_[0, 0, gains, 0, 0])
    _, matched = signal.istft(transform * (10 ** (band_gain / 20))[:, None], fs=RATE,
                             nperseg=FFT_SIZE, noverlap=FFT_OVERLAP)
    return matched[:len(audio)].astype(np.float32)


def equalize(audio, gains):
    if gains is None:
        return audio
    padded = np.pad(audio, (0, max(0, FFT_SIZE - len(audio))))
    freqs, _, transform = signal.stft(padded, fs=RATE, nperseg=FFT_SIZE, noverlap=FFT_OVERLAP)
    return apply_spectrum(audio, freqs, transform, gains)


def voiced_loudness(samples, pitch, meter):
    voiced = voiced_at(np.arange(0, len(samples), PITCH_HOP) / RATE, pitch)
    if np.count_nonzero(voiced) < 10:
        return None
    # Measure the K-weighted signal on voiced samples, without inserting
    # discontinuities into either the meter's filters or the inference audio.
    filtered = samples.astype(np.float64)
    for weighting in meter._filters.values():
        filtered = weighting.apply_filter(filtered)
    active = np.repeat(voiced, PITCH_HOP)[:len(samples)]
    power = float(np.mean(filtered[active] ** 2))
    return -0.691 + 10 * np.log10(power) if power > 1e-10 else None


def align_cvec(features, target_mean, target_std, alpha, torch, source_stats=None):
    if features.ndim != 3 or features.shape[-1] != target_mean.numel():
        raise ValueError("RIFT speaker profile and ContentVec dimensions do not match")
    if features.shape[1] < 3:
        return features
    working = features.float()
    if source_stats is None:
        source_std, source_mean = torch.std_mean(working, dim=1, unbiased=False, keepdim=True)
    else:
        source_mean, source_std = source_stats
    scale = (target_std / source_std.clamp_min(1e-4)).clamp(0.5, 2.0)
    correction = (target_mean - source_mean).clamp(-2 * target_std, 2 * target_std)
    aligned = working + alpha * ((working - source_mean) * (scale - 1) + correction)
    return aligned.to(features.dtype)


def pitch_clusters(bounds, midi_tracks, centroids):
    """Return up to two clusters without assigning differing octaves inside an overlap."""
    units = []
    for index, (start, end) in enumerate(bounds):
        if index and start < bounds[index - 1][1]:
            units[-1].append(index)
        else:
            units.append([index])

    voiced = []
    for index, unit in enumerate(units):
        midi = np.concatenate([midi_tracks[i] for i in unit])
        if len(midi) >= 10:
            color = [centroids[i] for i in unit if centroids[i] is not None]
            voiced.append((index, midi, float(np.median(color)) if color else None))
    labels = np.zeros(len(bounds), dtype=np.intp)
    if len(voiced) < 2:
        return labels

    medians = np.array([np.median(midi) for _, midi, _ in voiced])
    colors = [value for _, _, value in voiced if value is not None]
    neutral = float(np.median(colors)) if colors else 1000.0
    features = np.column_stack((medians / 7, np.log2([max(100, value or neutral) for _, _, value in voiced]) / 2))
    weights = np.array([len(midi) for _, midi, _ in voiced])
    if np.ptp(medians) <= 7:
        return labels
    centers = features[[int(medians.argmin()), int(medians.argmax())]].copy()
    for _ in range(16):
        groups = np.argmin(np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
        if min(weights[groups == 0].sum(), weights[groups == 1].sum()) < 0.15 * weights.sum():
            return labels
        updated = np.array([np.average(features[groups == i], weights=weights[groups == i], axis=0) for i in (0, 1)])
        if np.allclose(updated, centers):
            break
        centers = updated

    medians = [np.median(np.concatenate([midi for (_, midi, _), group in zip(voiced, groups) if group == i])) for i in (0, 1)]
    if abs(medians[0] - medians[1]) <= 7:
        return labels
    # Breath-only units borrow the nearest voiced section's group.
    known = list(zip([index for index, _, _ in voiced], groups))
    for index, unit in enumerate(units):
        group = min(known, key=lambda entry: abs(entry[0] - index))[1]
        labels[unit] = group if medians[0] < medians[1] else 1 - group
    return labels
