from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from music_runtime import octave_probe_assets
from music_runtime.pipeline import _fingerprint_sections, configured_target_fingerprint
from music_runtime.runtime import MusicRuntime
from music_runtime.settings import PipelineError, validate_settings

try:
    from music_runtime.workers import rift_octave_probe, rift_phrase_quality, rift_worker
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "scipy", "pyloudnorm", "librosa"}:
        raise
    rift_octave_probe = rift_phrase_quality = rift_worker = None


@unittest.skipIf(np is None, "NumPy is required for octave probe checks")
class OctaveProbeTests(unittest.TestCase):
    def test_settings_and_disabled_fingerprints_preserve_legacy_identity(self):
        absent = {"rift": {"adaptive_quality": True}}
        disabled = {"rift": {"adaptive_quality": True, "octave_probe": False,
                              "octave_probe_target_voice": "unknown"}}
        enabled = {"rift": {"adaptive_quality": True, "octave_probe": True,
                             "octave_probe_target_voice": "female"}}
        self.assertEqual(_fingerprint_sections(absent), _fingerprint_sections(disabled))
        self.assertEqual(configured_target_fingerprint(absent), configured_target_fingerprint(disabled))
        self.assertNotEqual(configured_target_fingerprint(absent), configured_target_fingerprint(enabled))
        with tempfile.TemporaryDirectory() as temporary:
            valid = validate_settings({"rift": {"octave_probe": True, "octave_probe_target_voice": "male"}},
                                      Path(temporary), require_paths=False)
            self.assertTrue(valid["rift"]["octave_probe"])
            self.assertEqual(valid["rift"]["octave_probe_target_voice"], "male")
            self.assertFalse(validate_settings({}, Path(temporary), require_paths=False)["rift"]["octave_probe"])
            with self.assertRaises(PipelineError):
                validate_settings({"rift": {"octave_probe_target_voice": "alto"}},
                                  Path(temporary), require_paths=False)

    def test_optional_asset_state_distinguishes_missing_prepared_and_corrupt(self):
        payload = b"pinned"
        digest = hashlib.sha256(payload).hexdigest()
        models = {"test.onnx": ("https://invalid.example/model", len(payload), digest)}
        with tempfile.TemporaryDirectory() as temporary, patch.object(octave_probe_assets, "MODELS", models):
            folder = Path(temporary) / "models"
            missing = octave_probe_assets.octave_probe_assets_state(folder, verify=True)
            self.assertEqual(missing["state"], "missing")
            folder.mkdir()
            (folder / "test.onnx").write_bytes(payload)
            prepared = octave_probe_assets.octave_probe_assets_state(folder, verify=True)
            self.assertEqual(prepared["state"], "prepared")
            self.assertNotEqual(prepared["fingerprint"], missing["fingerprint"])
            (folder / "test.onnx").write_bytes(b"broken")
            self.assertEqual(octave_probe_assets.octave_probe_assets_state(folder, verify=True)["state"], "corrupt")

    def test_classifier_abstains_on_conflicts_sparse_votes_and_pooled_dissent(self):
        if rift_octave_probe is None:
            self.skipTest("The isolated RIFT DSP dependencies are unavailable")

        def classify(rows_by_excerpt):
            class Session:
                def __init__(self, outputs):
                    self.outputs = iter(outputs)

                def run(self, _names, inputs):
                    if "embeddings" not in inputs:
                        return [np.zeros((len(next(iter(inputs.values())),), 8), dtype=np.float32)]
                    return [next(self.outputs)]

            probe = object.__new__(rift_octave_probe.OctaveProbeClassifier)
            probe.yamnet = SimpleNamespace(run=lambda _names, inputs: [
                np.zeros((len(next(iter(inputs.values())),), 8), dtype=np.float32)
            ])
            probe.gender = Session(rows_by_excerpt)
            probe.voice = SimpleNamespace(run=lambda _names, inputs: [
                np.tile(np.asarray([[0.05, 0.95]], dtype=np.float32), (len(inputs["embeddings"]), 1))
            ])
            excerpts = [
                {"audio16": np.zeros(32000, dtype=np.float32), "phrase_index": i,
                 "start_seconds": float(i * 5), "end_seconds": float(i * 5 + 2),
                 "active_start_seconds": 0.0, "active_end_seconds": 2.0}
                for i in range(len(rows_by_excerpt))
            ]
            with patch.object(rift_octave_probe, "_log_mel_patches", return_value=(
                np.zeros((3, 96, 64), dtype=np.float32), np.asarray([0.5, 1.0, 1.5]),
            )):
                return probe.classify(excerpts)

        female = np.tile(np.asarray([[0.91, 0.09]], dtype=np.float32), (3, 1))
        male = np.tile(np.asarray([[0.08, 0.92]], dtype=np.float32), (3, 1))
        uncertain = np.tile(np.asarray([[0.55, 0.45]], dtype=np.float32), (3, 1))
        female_two = np.asarray([[0.91, 0.09], [0.91, 0.09], [0.55, 0.45]], dtype=np.float32)
        sparse_male = np.asarray([[0.08, 0.92], [0.55, 0.45], [0.55, 0.45]], dtype=np.float32)
        self.assertEqual(classify([female, female, uncertain])["source_voice"], "female")
        conflict = classify([female, male, uncertain])
        self.assertEqual(conflict["source_voice"], "unknown")
        self.assertEqual(conflict["source_voice_reason"], "conflicting_phrase_votes")
        dissent = classify([female_two, female_two, sparse_male])
        self.assertEqual(dissent["source_voice"], "unknown")
        self.assertEqual(dissent["source_voice_reason"], "pooled_confident_vote_share_low")
        sparse = classify([np.asarray([[0.92, 0.08], [0.55, 0.45], [0.55, 0.45]], dtype=np.float32)])
        self.assertEqual(sparse["source_voice"], "unknown")
        self.assertEqual(sparse["source_voice_reason"], "sparse_confident_phrase_votes")

    def test_acoustic_prior_order_and_f0_target_distribution_order(self):
        if rift_phrase_quality is None:
            self.skipTest("The isolated RIFT DSP dependencies are unavailable")
        order, reason = rift_phrase_quality._octave_probe_candidate_order(
            {"source_voice": "male"}, [], {"octave_probe_target_voice": "female"},
        )
        self.assertEqual(order, [12, 0, -12])
        self.assertEqual(reason, "confident_opposite_acoustic_voice_labels")
        order, reason = rift_phrase_quality._octave_probe_candidate_order(
            {"source_voice": "unknown"},
            [{"median_f0_hz": 220.0, "upper_f0_hz": 330.0}],
            {"octave_probe_target_voice": "unknown", "pitch_range": [45, 55, 65]},
        )
        self.assertIn(order[0], {-12, 0, 12})
        self.assertEqual(reason, "source_f0_quantiles_against_target_pitch_range")

    def test_local_dropout_guard_rejects_aggregate_improvement_and_bad_shift(self):
        if rift_phrase_quality is None:
            self.skipTest("The isolated RIFT DSP dependencies are unavailable")

        def metrics(coverage, expected, local, shift):
            return {"source_voiced_frames": 180, "voiced_coverage": coverage,
                    "source_voiced_ratio": 0.8, "local_windows_checked": 1,
                    "expected_octave_coverage": expected, "post_shift_semitones": shift,
                    "worst_local_voiced_loss": local}

        baseline = metrics(0.80, 0.70, 0.13, 0.0)
        improved_but_local_dropout = metrics(0.94, 0.99, 0.39, 12.0)
        assessment = rift_phrase_quality._octave_probe_candidate_assessment(
            [(improved_but_local_dropout, baseline, index) for index in (1, 2)], 12,
        )
        self.assertGreater(assessment["objective"], 0.80 * 0.55 + 0.70 * 0.45)
        self.assertEqual(assessment["rejection_reason"], "localized_or_global_regression")
        mismatch = rift_phrase_quality._octave_probe_candidate_assessment(
            [(metrics(0.9, 0.95, 0.1, 9.0), baseline, 1),
             (metrics(0.9, 0.95, 0.1, 9.2), baseline, 2)], 12,
        )
        self.assertEqual(mismatch["rejection_reason"], "candidate_median_shift_outside_one_semitone")

    def test_selected_crop_without_reliable_local_evidence_cannot_support_recommendation(self):
        if rift_phrase_quality is None:
            self.skipTest("The isolated RIFT DSP dependencies are unavailable")
        rate = rift_phrase_quality.RATE
        mono = np.zeros(36 * rate, dtype=np.float32)
        phrases = [SimpleNamespace(start=i * 12 * rate, end=(i + 1) * 12 * rate) for i in range(3)]

        sparse_f0 = np.zeros(1200, dtype=np.float64)
        sparse_confidence = np.zeros(1200, dtype=np.float64)
        for start, end in ((0, 200), (585, 615), (1000, 1200)):
            sparse_f0[start:end] = 150.0
            sparse_confidence[start:end] = 1.0
        tracks = [(sparse_f0, sparse_confidence)]
        tracks.extend((np.full(1200, f0), np.ones(1200)) for f0 in (300.0, 500.0))

        excerpts = rift_phrase_quality._select_octave_probe_excerpts(mono, phrases, tracks)
        self.assertEqual(len(excerpts), 3)
        sparse_excerpt = next(item for item in excerpts if item["phrase_index"] == 0)
        self.assertAlmostEqual(sparse_excerpt["start_seconds"], 3.99, places=2)
        self.assertAlmostEqual(sparse_excerpt["end_seconds"], 7.99, places=2)

        pairs = []
        for excerpt in excerpts:
            phrase_index = excerpt["phrase_index"]
            source_f0, source_confidence = tracks[phrase_index]
            offset = round((excerpt["start"] - phrases[phrase_index].start) / (rate / 100))
            source_f0 = source_f0[offset:offset + 401]
            source_confidence = source_confidence[offset:offset + 401]
            self.assertEqual(len(source_f0), 401)
            baseline = rift_phrase_quality._upper_register_metrics(
                source_f0, source_confidence, source_f0, source_confidence, expected_shift=12,
            )
            candidate_f0 = source_f0.copy()
            voiced = source_f0 > 0
            candidate_f0[voiced] *= 2.0
            candidate = rift_phrase_quality._upper_register_metrics(
                source_f0, source_confidence, candidate_f0, source_confidence, expected_shift=12,
            )
            pairs.append((candidate, baseline, phrase_index))

        sparse_pair = next(pair for pair in pairs if pair[2] == 0)
        self.assertEqual(sparse_pair[0]["source_voiced_frames"], 30)
        self.assertAlmostEqual(sparse_pair[0]["source_voiced_ratio"], 30 / 401, places=4)
        self.assertEqual(sparse_pair[0]["local_windows_checked"], 0)

        # Reproduce one unreliable 4-second crop paired with one valid excerpt.
        reliable_pair = next(pair for pair in pairs if pair[2] == 1)
        reproduction = rift_phrase_quality._octave_probe_candidate_assessment(
            [sparse_pair, reliable_pair], 12,
        )
        self.assertEqual(reproduction["metrics"]["sample_count"], 1)
        self.assertEqual(reproduction["rejection_reason"], "unreliable_paired_excerpt_metrics")
        self.assertEqual(reproduction["unreliable_paired_samples"][0]["phrase_index"], 0)
        self.assertIn("source_voiced_frames_below_minimum",
                      reproduction["unreliable_paired_samples"][0]["candidate_reasons"])
        self.assertIn("local_voiced_loss_evidence_unavailable",
                      reproduction["unreliable_paired_samples"][0]["baseline_reasons"])

        # The selected bad crop remains disqualifying even when two other excerpts qualify.
        all_selected = rift_phrase_quality._octave_probe_candidate_assessment(pairs, 12)
        self.assertEqual(all_selected["metrics"]["sample_count"], 2)
        self.assertEqual(all_selected["rejection_reason"], "unreliable_paired_excerpt_metrics")

        valid_control = rift_phrase_quality._octave_probe_candidate_assessment(
            [pair for pair in pairs if pair[2] in {1, 2}], 12,
        )
        self.assertEqual(valid_control["metrics"]["sample_count"], 2)
        self.assertTrue(valid_control["material_improvement"])
        self.assertIsNone(valid_control["rejection_reason"])

    def test_excerpt_selection_and_resampling_reject_dropped_audio(self):
        if rift_phrase_quality is None or rift_worker is None or rift_octave_probe is None:
            self.skipTest("The isolated RIFT DSP dependencies are unavailable")
        rate = rift_phrase_quality.RATE
        mono = np.ones(14 * rate, dtype=np.float32)
        phrases = [SimpleNamespace(start=i * 5 * rate, end=i * 5 * rate + 4 * rate) for i in range(3)]
        tracks = [(np.full(401, f0), np.ones(401)) for f0 in (130.0, 250.0, 520.0)]
        selected = rift_phrase_quality._select_octave_probe_excerpts(mono, phrases, tracks)
        self.assertEqual([item["band"] for item in selected], ["lower", "middle", "upper"])
        self.assertTrue(all(2.5 <= item["end_seconds"] - item["start_seconds"] <= 4.0 for item in selected))
        with self.assertRaises(ValueError):
            rift_octave_probe._log_mel_patches(np.zeros(0, dtype=np.float32))
        with self.assertRaises(ValueError):
            rift_octave_probe._log_mel_patches(np.zeros(8000, dtype=np.float32))

        rates = []

        class FakeClassifier:
            def classify(self, excerpts):
                return {"resampled_lengths": [len(item["audio16"]) for item in excerpts]}

        worker = object.__new__(rift_worker._Rift)
        worker._octave_probe_classifier = FakeClassifier()
        worker.np = np
        worker.librosa = SimpleNamespace(resample=lambda samples, *, orig_sr, target_sr: (
            rates.append((orig_sr, target_sr, len(samples))) or np.zeros(round(len(samples) * target_sr / orig_sr), dtype=np.float32)
        ))
        result = worker._classify_octave_probe([{"audio": np.ones(rate), "phrase_index": 0}])
        self.assertEqual(rates, [(44100, 16000, rate)])
        self.assertGreater(result["resampled_lengths"][0], 0)

    def test_disabled_manual_and_backing_runtime_routes_do_not_forward_probe(self):
        async def run_case(probe, key_shift, role):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                plugin_dir, runtime_dir = root / "plugin", root / "runtime"
                plugin_dir.mkdir()
                runtime_dir.mkdir()
                checkpoint = plugin_dir / "voice.ckpt"
                checkpoint.write_bytes(b"checkpoint")
                source = runtime_dir / "lead.wav"
                source.write_bytes(b"source")
                output = runtime_dir / "converted.wav"
                output.write_bytes(b"converted")
                settings = {
                    "plugin_dir": plugin_dir, "runtime_dir": runtime_dir,
                    "pipeline": {"stage_timeout_seconds": 5},
                    "runtime": {"device": "cpu"},
                    "rift": {
                        "checkpoint": str(checkpoint), "speaker": "voice",
                        "backing_checkpoint": "", "backing_speaker": "",
                        "key_shift": key_shift, "backing_key_shift": 0, "song_octave_shift": 0,
                        "infer_steps": 4, "ds_cfg_strength": 0.1, "spk_cfg_strength": 0.5,
                        "cfg_rescale": 0.7, "batch_size": 1, "robust_f0": 1,
                        "adaptive_quality": True, "timbre_slicing": True,
                        "domain_alignment": True, "domain_formants": False,
                        "allow_octave_jump": True, "octave_probe": probe,
                        "octave_probe_target_voice": "unknown",
                    },
                    "mix": {}, "separation": {},
                }
                runtime = MusicRuntime(settings)
                runtime._audio.stop = AsyncMock()
                runtime._ensure_rift_started_unlocked = AsyncMock()
                runtime._rift.request = AsyncMock(return_value={"path": str(output)})
                with patch("music_runtime.runtime.phrase_assets_ready", return_value=True):
                    await runtime.convert(source, output, role)
                request = runtime._rift.request.await_args.args[0]
                await runtime.close()
                return request["rift"]

        async def scenario():
            return (
                await run_case(False, 0, "lead"),
                await run_case(True, 1, "lead"),
                await run_case(True, 0, "backing"),
            )

        disabled, manual, backing = asyncio.run(scenario())
        self.assertNotIn("octave_probe", disabled)
        self.assertNotIn("octave_probe", manual)
        self.assertNotIn("octave_probe", backing)


if __name__ == "__main__":
    unittest.main()
