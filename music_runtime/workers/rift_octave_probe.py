"""CPU Essentia voice-type classification for optional octave QA diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


GENDER_LABELS = ("female", "male")
VOICE_GATE_LABELS = ("instrumental", "voice")
VOICE_GATE_THRESHOLD = 0.70
GENDER_CONFIDENCE_THRESHOLD = 0.80
MIN_CONFIDENT_VOTES_PER_EXCERPT = 2
MIN_GLOBAL_CONFIDENT_VOTES = 4
MIN_GLOBAL_VOTE_SHARE = 0.90


def _assets_module():
    if __package__:
        from .. import octave_probe_assets
        return octave_probe_assets
    # The persistent worker is launched by script path with the RIFT source as
    # cwd. Resolve its sibling package directly instead of relying on PYTHONPATH.
    module_parent = str(Path(__file__).resolve().parents[1])
    added = module_parent not in sys.path
    if added:
        sys.path.insert(0, module_parent)
    try:
        import octave_probe_assets
        return octave_probe_assets
    finally:
        if added and sys.path and sys.path[0] == module_parent:
            sys.path.pop(0)


def _as_mel(hz):
    return np.log1p(np.asarray(hz, dtype=np.float64) / 700.0)


def _log_mel_patches(audio16):
    audio = np.asarray(audio16, dtype=np.float32).reshape(-1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("The octave probe excerpt contains no finite audio.")
    if len(audio) < 16000:
        raise ValueError("The octave probe excerpt is shorter than one second.")
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(audio, (200, 0)), 400)[::160]
    magnitude = np.abs(np.fft.rfft(frames * np.hanning(400).astype(np.float32), 512, axis=1))
    bins = _as_mel(np.fft.rfftfreq(512, 1 / 16000))[:, None]
    edges = np.linspace(_as_mel(125.0), _as_mel(7500.0), 66)
    lower = (bins - edges[:-2]) / np.diff(edges)[:-1]
    upper = (edges[2:] - bins) / np.diff(edges)[1:]
    filters = np.maximum(0, np.minimum(lower, upper)).astype(np.float32)
    log_mel = np.log(magnitude @ filters + 0.01).astype(np.float32)
    starts = np.arange(0, len(log_mel) - 96 + 1, 48, dtype=np.int64)
    if not len(starts):
        raise ValueError("The octave probe excerpt did not produce a complete model patch.")
    patches = log_mel[starts[:, None] + np.arange(96)]
    centers_seconds = (starts + 47.5) / 100.0
    return patches, centers_seconds


class OctaveProbeClassifier:
    """Load the three pinned Essentia models on CPU and classify short excerpts."""

    def __init__(self, model_dir: Path, packages_dir: Path):
        self.model_dir = Path(model_dir)
        self.packages_dir = Path(packages_dir)
        assets = _assets_module()
        state = assets.octave_probe_assets_state(self.model_dir, verify=True)
        if state["state"] != "prepared":
            raise RuntimeError(f"octave_probe_assets_{state['state']}")
        runtime_package = self.packages_dir / "onnxruntime" / "__init__.py"
        if not runtime_package.is_file():
            raise RuntimeError("octave_probe_onnxruntime_unavailable")
        package_path = str(self.packages_dir.resolve())
        sys.path.insert(0, package_path)
        try:
            import onnxruntime as ort
        finally:
            if sys.path and sys.path[0] == package_path:
                sys.path.pop(0)

        expected_root = (self.packages_dir / "onnxruntime").resolve()
        loaded_path = Path(ort.__file__).resolve()
        if not loaded_path.is_relative_to(expected_root):
            raise RuntimeError("octave_probe_onnxruntime_outside_phrase_bundle")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.yamnet = ort.InferenceSession(
            str(self.model_dir / "audioset-yamnet-1.onnx"), options,
            providers=["CPUExecutionProvider"],
        )
        self.gender = ort.InferenceSession(
            str(self.model_dir / "gender-audioset-yamnet-1.onnx"), options,
            providers=["CPUExecutionProvider"],
        )
        self.voice = ort.InferenceSession(
            str(self.model_dir / "voice_instrumental-audioset-yamnet-1.onnx"), options,
            providers=["CPUExecutionProvider"],
        )

    def _classify_excerpt(self, excerpt):
        patches, centers = _log_mel_patches(excerpt["audio16"])
        embeddings = np.concatenate([
            self.yamnet.run(["embeddings:0"], {"melspectrogram:0": batch})[0]
            for batch in np.array_split(patches, (len(patches) + 31) // 32)
        ])
        gender_scores = self.gender.run(None, {"embeddings": embeddings})[0]
        voice_scores = self.voice.run(None, {"embeddings": embeddings})[0]
        if (gender_scores.shape != (len(centers), 2) or voice_scores.shape != (len(centers), 2)
                or not np.isfinite(gender_scores).all() or not np.isfinite(voice_scores).all()
                or not np.allclose(gender_scores.sum(axis=1), 1, atol=0.02)
                or not np.allclose(voice_scores.sum(axis=1), 1, atol=0.02)):
            raise ValueError("invalid_classification_scores")
        active_start = float(excerpt["active_start_seconds"])
        active_end = float(excerpt["active_end_seconds"])
        active = (centers >= active_start) & (centers < active_end)
        voice_gated = active & (voice_scores[:, 1] >= VOICE_GATE_THRESHOLD)
        gender_confidence = np.max(gender_scores, axis=1)
        confident = voice_gated & (gender_confidence >= GENDER_CONFIDENCE_THRESHOLD)
        predicted = np.argmax(gender_scores, axis=1)
        votes = {
            label: int(np.count_nonzero(confident & (predicted == index)))
            for index, label in enumerate(GENDER_LABELS)
        }
        vote_total = sum(votes.values())
        excerpt_label, excerpt_confidence = None, None
        if vote_total >= MIN_CONFIDENT_VOTES_PER_EXCERPT:
            winner = max(GENDER_LABELS, key=lambda label: votes[label])
            if votes[winner] / vote_total >= 0.67:
                excerpt_label = winner
                excerpt_confidence = float(np.mean(
                    gender_confidence[confident & (predicted == GENDER_LABELS.index(winner))]
                ))
        gated_count = int(np.count_nonzero(voice_gated))
        return {
            "phrase_index": int(excerpt["phrase_index"]),
            "start_seconds": round(float(excerpt["start_seconds"]), 4),
            "end_seconds": round(float(excerpt["end_seconds"]), 4),
            "patch_count": int(np.count_nonzero(active)),
            "voice_gate_threshold": VOICE_GATE_THRESHOLD,
            "voice_gated_patch_count": gated_count,
            "mean_voice_score": round(float(np.mean(voice_scores[active, 1])), 4) if np.any(active) else None,
            "gender_confidence_threshold": GENDER_CONFIDENCE_THRESHOLD,
            "confident_vote_counts": votes,
            "gender_mean_scores": {
                label: round(float(np.mean(gender_scores[voice_gated, index])), 4) if gated_count else None
                for index, label in enumerate(GENDER_LABELS)
            },
            "label": excerpt_label or "unknown",
            "confidence": round(excerpt_confidence, 4) if excerpt_confidence is not None else None,
        }

    def classify(self, excerpts):
        results = []
        for excerpt in excerpts:
            try:
                results.append(self._classify_excerpt(excerpt))
            except Exception as exc:
                results.append({
                    "phrase_index": int(excerpt.get("phrase_index", -1)),
                    "start_seconds": round(float(excerpt.get("start_seconds", 0.0)), 4),
                    "end_seconds": round(float(excerpt.get("end_seconds", 0.0)), 4),
                    "patch_count": 0,
                    "voice_gated_patch_count": 0,
                    "confident_vote_counts": {"female": 0, "male": 0},
                    "gender_mean_scores": {"female": None, "male": None},
                    "label": "unknown",
                    "confidence": None,
                    "error": f"excerpt_probe_failed:{type(exc).__name__}",
                })
        known = [item["label"] for item in results if item["label"] in GENDER_LABELS]
        vote_counts = {
            label: sum(item["confident_vote_counts"][label] for item in results)
            for label in GENDER_LABELS
        }
        total = sum(vote_counts.values())
        winner = max(GENDER_LABELS, key=lambda label: vote_counts[label]) if total else None
        winning_share = vote_counts[winner] / total if winner is not None else 0.0
        if (len(known) >= 2 and len(set(known)) == 1 and known[0] == winner
                and total >= MIN_GLOBAL_CONFIDENT_VOTES and winning_share >= MIN_GLOBAL_VOTE_SHARE):
            source_voice = winner
            confidence_values = [item["confidence"] for item in results if item["label"] == source_voice]
            confidence = float(np.mean(confidence_values)) if confidence_values else None
            reason = "consistent_confident_phrase_votes"
        elif len(set(known)) > 1:
            source_voice, confidence, reason = "unknown", None, "conflicting_phrase_votes"
        elif total >= MIN_GLOBAL_CONFIDENT_VOTES and winning_share < MIN_GLOBAL_VOTE_SHARE:
            source_voice, confidence, reason = "unknown", None, "pooled_confident_vote_share_low"
        else:
            source_voice, confidence, reason = "unknown", None, "sparse_confident_phrase_votes"
        return {
            "model": {
                "embedding": "Essentia audioset-yamnet-1.onnx",
                "gender_head": "Essentia gender-audioset-yamnet-1.onnx",
                "voice_gate": "Essentia voice_instrumental-audioset-yamnet-1.onnx",
                "gender_labels": list(GENDER_LABELS),
                "voice_gate_labels": list(VOICE_GATE_LABELS),
                "input_sample_rate": 16000,
                "patch_duration_seconds": 0.96,
                "patch_hop_seconds": 0.48,
                "provider": "CPUExecutionProvider",
                "scores_are_calibrated": False,
            },
            "source_voice": source_voice,
            "source_voice_confidence": round(confidence, 4) if confidence is not None else None,
            "source_voice_reason": reason,
            "vote_counts": vote_counts,
            "global_confident_vote_share": round(winning_share, 4) if total else None,
            "minimum_global_vote_share": MIN_GLOBAL_VOTE_SHARE,
            "voice_gate": {
                "threshold": VOICE_GATE_THRESHOLD,
                "total_gated_patches": sum(item["voice_gated_patch_count"] for item in results),
                "gender_confidence_threshold": GENDER_CONFIDENCE_THRESHOLD,
            },
            "excerpts": results,
        }


__all__ = [
    "GENDER_CONFIDENCE_THRESHOLD", "GENDER_LABELS", "MIN_CONFIDENT_VOTES_PER_EXCERPT",
    "MIN_GLOBAL_CONFIDENT_VOTES", "MIN_GLOBAL_VOTE_SHARE", "OctaveProbeClassifier",
    "VOICE_GATE_LABELS", "VOICE_GATE_THRESHOLD",
]
