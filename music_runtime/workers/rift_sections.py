"""Detect vocal phrases, cluster their style, and merge compatible neighbours."""

import sys
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal
from scipy.cluster.hierarchy import cut_tree, fcluster, linkage
from scipy.spatial.distance import pdist

if __package__:
    from .rift_profile import RATE
else:
    from rift_profile import RATE

VAD_RATE = 16000
VAD_FRAME = 512
MIN_PHRASE = round(1.5 * RATE)
MAX_PHRASE = 12 * RATE
MAX_SECTION = 14 * RATE
MAX_GAP = round(1.5 * RATE)
ANALYSIS_CONTEXT = round(0.3 * RATE)
EDGE_FADE = round(0.035 * RATE)


@dataclass(frozen=True)
class Phrase:
    start: int
    end: int
    boundary: str


@dataclass(frozen=True)
class Section:
    start: int
    end: int
    cluster: int
    phrases: tuple[int, ...]


class PhraseModels:
    def __init__(self, directory, torch):
        sys.path.insert(0, str(directory / "packages"))
        try:
            import onnxruntime as ort
        finally:
            sys.path.pop(0)
        from torchaudio.compliance.kaldi import fbank

        self.torch, self.fbank = torch, fbank
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.vad = ort.InferenceSession(str(directory / "silero_vad.onnx"), options, providers=["CPUExecutionProvider"])
        self.speaker = ort.InferenceSession(str(directory / "wespeaker.onnx"), options, providers=["CPUExecutionProvider"])

    def probabilities(self, audio16):
        blocks = np.pad(audio16, (0, -len(audio16) % VAD_FRAME)).reshape(-1, VAD_FRAME)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(64, dtype=np.float32)
        rate = np.asarray(VAD_RATE, dtype=np.int64)
        result = np.empty(len(blocks), dtype=np.float32)
        for i, block in enumerate(blocks):
            window = np.concatenate((context, block))[None, :]
            output, state = self.vad.run(None, {"input": window, "state": state, "sr": rate})
            result[i] = output.item()
            context = block[-64:]
        return result

    def embedding(self, audio16):
        samples = self.torch.from_numpy(audio16).reshape(1, -1) * 32768
        with self.torch.inference_mode():
            features = self.fbank(samples, sample_frequency=VAD_RATE, num_mel_bins=80,
                                  window_type="hamming", dither=0.0, use_energy=False)
            features = (features - features.mean(dim=0)).unsqueeze(0).numpy()
        embedding = self.speaker.run(["embs"], {"feats": features})[0].reshape(-1)
        return embedding / max(float(np.linalg.norm(embedding)), 1e-8)


def analysis_audio(samples):
    return signal.resample_poly(samples, 160, 441).astype(np.float32, copy=False)


def phrase_boundaries(samples, probabilities):
    """Find VAD pauses with an energy guard for unrecognised sung vowels."""
    starts = np.rint(np.arange(len(probabilities)) * RATE * VAD_FRAME / VAD_RATE).astype(np.intp)
    starts = starts[starts < len(samples)]
    if not len(starts):
        return []
    probabilities = probabilities[:len(starts)]
    stops = np.r_[starts[1:], len(samples)]
    levels = np.sqrt(np.add.reduceat(samples ** 2, starts) / (stops - starts))
    reference = float(np.percentile(levels, 85))
    silence = (probabilities < 0.25) & (levels < max(2e-4, min(0.0015, reference * 0.025)))
    audible = np.flatnonzero(~silence)
    if not len(audible):
        return []
    margin = round(0.06 * RATE)
    start = max(0, int(starts[audible[0]]) - margin)
    end = min(len(samples), int(stops[audible[-1]]) + margin)
    nearby = ndimage.maximum_filter1d(levels, size=25, mode="nearest")
    before = nearby[np.maximum(0, np.arange(len(levels)) - 12)]
    after = nearby[np.minimum(len(levels) - 1, np.arange(len(levels)) + 12)]
    floor = np.maximum(0.001, np.minimum(0.02, np.minimum(before, after) * 0.15))
    quiet = (probabilities < 0.40) & (levels < floor)
    transitions = np.diff(np.r_[False, quiet, False].astype(np.int8))
    phrases = []
    cursor, reason = start, "vocal_onset"
    for i, j in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1), strict=True):
        if j - i < 4:
            continue
        left, right = int(starts[i]), int(stops[j - 1])
        cut = (left + right) // 2
        if cut <= cursor or cut >= end:
            continue
        if right - left >= MAX_GAP and silence[i:j].mean() >= 0.90:
            if left + margin > cursor and right - margin < end:
                phrases.append(Phrase(cursor, left + margin, reason))
                cursor, reason = right - margin, "pause"
        elif cut - cursor >= MIN_PHRASE and end - cut >= MIN_PHRASE:
            phrases.append(Phrase(cursor, cut, reason))
            cursor, reason = cut, "breath"
    if cursor < end:
        phrases.append(Phrase(cursor, end, reason))

    # Only overlong continuous phrases may be cut without a detected pause.
    envelope = ndimage.uniform_filter1d(levels, size=3)
    bounded = []
    for phrase in phrases:
        cursor, reason = phrase.start, phrase.boundary
        while phrase.end - cursor > MAX_PHRASE:
            latest = min(cursor + MAX_PHRASE, phrase.end - MIN_PHRASE)
            earliest = max(cursor + MIN_PHRASE, latest - 3 * RATE)
            candidates = np.flatnonzero((starts >= earliest) & (starts <= latest))
            cut = int(starts[candidates[np.argmin(envelope[candidates])]]) if len(candidates) else latest
            bounded.append(Phrase(cursor, cut, reason))
            cursor, reason = cut, "energy_minimum"
        bounded.append(Phrase(cursor, phrase.end, reason))
    return bounded


def style_features(samples, pitch):
    f0, confidence = pitch
    voiced = (f0 > 20) & (confidence >= 0.3)
    midi = 69 + 12 * np.log2(f0[voiced] / 440)
    pairs = voiced[1:] & voiced[:-1]
    jitter = float(np.mean(np.abs(12 * np.log2(f0[1:][pairs] / f0[:-1][pairs])) > 1.5)) if pairs.any() else 0.0
    freqs, power = signal.welch(samples, fs=RATE, nperseg=min(2048, len(samples)))
    low = power[(freqs >= 120) & (freqs < 500)].sum()
    high = power[(freqs >= 2000) & (freqs < 6000)].sum()
    tilt = 10 * np.log10((high + 1e-14) / (low + 1e-14))
    mask = np.repeat(voiced, RATE // 100)[:len(samples)]
    active = samples[:len(mask)][mask] if mask.any() else samples
    level = 10 * np.log10(max(float(np.mean(active ** 2)), 1e-12))
    return np.array([(np.median(midi) - 60) / 12 if len(midi) else 0.0, tilt / 15.0,
                     jitter * 5, (level + 20) / 10], dtype=np.float64)


def cluster_phrases(phrases, embeddings, styles):
    if len(phrases) <= 1:
        return np.zeros(len(phrases), dtype=np.intp)
    speaker = np.clip(np.nan_to_num(pdist(embeddings, metric="cosine")), 0, 2)
    acoustic = np.clip(pdist(styles) / 2, 0, 2)
    tree = linkage(0.65 * speaker + 0.35 * acoustic, method="average")
    labels = fcluster(tree, t=0.38, criterion="distance")
    if len(np.unique(labels)) > 4:
        labels = cut_tree(tree, n_clusters=4).reshape(-1)
    original = labels.copy()
    for i in range(1, len(phrases) - 1):
        short = phrases[i].end - phrases[i].start < round(2.5 * RATE)
        nearby = (phrases[i].start - phrases[i - 1].end <= MAX_GAP
                  and phrases[i + 1].start - phrases[i].end <= MAX_GAP)
        if short and nearby and original[i - 1] == original[i + 1]:
            labels[i] = original[i - 1]
    by_order = {int(label): i for i, label in enumerate(dict.fromkeys(labels))}
    return np.array([by_order[int(label)] for label in labels], dtype=np.intp)


def merge_phrases(phrases, labels):
    sections = []
    for index, (phrase, label) in enumerate(zip(phrases, labels, strict=True)):
        if sections:
            previous = sections[-1]
            if (previous.cluster == label and phrase.start - previous.end <= MAX_GAP
                    and phrase.end - previous.start <= MAX_SECTION):
                sections[-1] = Section(previous.start, phrase.end, int(label), (*previous.phrases, index))
                continue
        sections.append(Section(phrase.start, phrase.end, int(label), (index,)))
    return sections
