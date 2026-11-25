"""Focused tests for conservative RIFT domain planning and profiles."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKERS = ROOT / "music_runtime" / "workers"
sys.path.insert(0, str(WORKERS))

import rift_domain as domain
from rift_profile import validate_profile, read_profile
from music_runtime.settings import PipelineError, validate_settings
from music_runtime.workers.rift_worker import _Rift


def _tracks(notes, confidence=1.0):
    midi = np.asarray(notes, dtype=np.float64)
    f0 = 440.0 * np.power(2.0, (midi - 69.0) / 12.0)
    conf = np.full(len(midi), confidence, dtype=np.float64)
    return [(f0, conf)]


def _base_profile(version=1):
    profile = {
        "version": version,
        "speaker": "target",
        "encoder_sha256": "a" * 64,
        "pitch_range": [55.0, 60.0, 65.0],
        "ltas_db": [0.0] * 18,
        "cvec_mean": [0.0, 1.0],
        "cvec_std": [1.0, 1.0],
    }
    if version == 2:
        profile.update({
            "pitch_tails": [54.0, 66.0],
            "formant_anchor_hz": 920.0,
            "formant_frames": 90,
            "formant_log_mad": 0.12,
        })
    return profile


class DomainPlanTests(unittest.TestCase):
    def test_song_octave_detection_accepts_consistent_low_and_high_register_songs(self):
        for note, expected in ((44.0, 12), (76.0, -12)):
            tracks = _tracks([note] * 120) * 4
            with self.subTest(note=note):
                plan = domain.plan_song_octave(tracks, [55, 60, 65], [54, 66])
                self.assertEqual(plan.octave_shift, expected)
                self.assertEqual(plan.reason, "song_octave_automatic")
                self.assertEqual(plan.reliable_phrases, 4)
                self.assertEqual(plan.pooled_notes, 480)
                self.assertGreaterEqual(plan.shifted_in_range_ratio, 0.65)
                self.assertGreaterEqual(plan.tail_improvement, 0.8)

    def test_song_octave_detection_uses_all_notes_and_rejects_mixed_registers(self):
        tracks = _tracks([44.0] * 120) * 2 + _tracks([60.0] * 120) * 2
        plan = domain.plan_song_octave(tracks, [55, 60, 65], [54, 66])
        self.assertEqual(plan.octave_shift, 0)
        self.assertEqual(plan.reason, "song_octave_mixed_register")
        self.assertEqual(plan.pooled_notes, 480)
        self.assertEqual(plan.reliable_phrases, 4)

    def test_song_octave_detection_rejects_shift_that_still_misses_upper_range(self):
        # The lower side alone appears to support +12, and tail loss improves,
        # but the shifted pooled notes still do not land in the target range.
        tracks = [_tracks([38.0] * 80 + [80.0] * 20)[0] for _ in range(4)]
        plan = domain.plan_song_octave(tracks, [55, 60, 65], [54, 66])
        self.assertEqual(plan.octave_shift, 0)
        self.assertEqual(plan.reason, "song_octave_shift_fails_target_coverage")
        self.assertGreater(plan.tail_improvement, 0.8)
        self.assertLess(plan.shifted_in_range_ratio, 0.65)

    def test_song_octave_detection_requires_reliable_natural_phrases(self):
        tracks = _tracks([44.0] * 120) * 3 + _tracks([44.0] * 99)
        plan = domain.plan_song_octave(tracks, [55, 60, 65])
        self.assertEqual(plan.octave_shift, 0)
        self.assertEqual(plan.reason, "song_octave_insufficient_reliable_phrases")
        self.assertEqual(plan.reliable_phrases, 3)
        self.assertEqual(plan.target_bounds, (53.0, 67.0))

    def test_smallest_useful_lift_within_best_score_tolerance_is_selected(self):
        notes = [55.0] * 100 + [60.0] * 900
        plan = domain.plan_domain(
            _tracks(notes), [60, 65, 75], 0,
            virtual_available=True, allow_octave=False, voiced_share=1.0,
            source_anchor=None, target_anchor=None, protect_formants=False,
        )
        self.assertEqual((plan.octave_shift, plan.virtual_lift), (0, 4))
        self.assertEqual(plan.reason, "pitch_tails_improved")
        self.assertLess(plan.predicted_outside_ratio, 0.2)

    def test_upper_tail_can_select_a_negative_virtual_lift(self):
        notes = [80.0] * 100 + [75.0] * 900
        plan = domain.plan_domain(
            _tracks(notes), [60, 65, 75], 0,
            virtual_available=True, allow_octave=False, voiced_share=0.9,
            source_anchor=None, target_anchor=None, protect_formants=False,
        )
        self.assertEqual((plan.octave_shift, plan.virtual_lift), (0, -4))

    def test_no_auto_lift_with_low_voiced_share_or_missing_virtual_capability(self):
        notes = [55.0] * 100 + [60.0] * 900
        for occupancy, available in ((0.19, True), (1.0, False)):
            with self.subTest(occupancy=occupancy, available=available):
                plan = domain.plan_domain(
                    _tracks(notes), [60, 65, 75], 0,
                    virtual_available=available, allow_octave=False, voiced_share=occupancy,
                    source_anchor=None, target_anchor=None, protect_formants=False,
                )
                self.assertEqual((plan.octave_shift, plan.virtual_lift), (0, 0))

    def test_low_confidence_frames_do_not_drive_auto_pitch(self):
        tracks = _tracks([55.0] * 100, confidence=0.1)
        plan = domain.plan_domain(
            tracks, [60, 65, 75], 0,
            virtual_available=True, allow_octave=False, voiced_share=0.8,
            source_anchor=None, target_anchor=None, protect_formants=False,
        )
        self.assertEqual((plan.octave_shift, plan.virtual_lift), (0, 0))

    def test_manual_non_octave_shift_is_preserved_and_suppresses_virtual_lift(self):
        plan = domain.plan_domain(
            _tracks([55.0] * 100), [60, 65, 75], 4,
            virtual_available=True, allow_octave=True, voiced_share=1.0,
            source_anchor=None, target_anchor=None, protect_formants=False,
        )
        self.assertEqual((plan.octave_shift, plan.virtual_lift), (4, 0))
        self.assertEqual(plan.reason, "manual_shift")

    def test_severe_structural_octave_is_opt_in_and_requires_benefit(self):
        notes = [55.0] * 100
        base = dict(
            tracks=_tracks(notes), pitch_range=[70, 75, 85], manual_shift=0,
            virtual_available=False, voiced_share=0.18, source_anchor=None,
            target_anchor=None, protect_formants=False,
        )
        disabled = domain.plan_domain(**base, allow_octave=False)
        enabled = domain.plan_domain(**base, allow_octave=True)
        self.assertEqual(disabled.octave_shift, 0)
        self.assertEqual(enabled.octave_shift, 12)
        self.assertEqual(enabled.virtual_lift, 0)

    def test_sustained_low_register_retains_an_octave_on_opt_in(self):
        base = dict(
            tracks=_tracks([51.0] * 40 + [54.0] * 75 + [59.0] * 85),
            pitch_range=[55, 60, 65], pitch_tails=[54, 67], manual_shift=0,
            virtual_available=False, allow_octave=False, voiced_share=0.7,
            source_anchor=None, target_anchor=None, protect_formants=False,
        )
        ordinary = domain.plan_domain(**base)
        raised = domain.plan_domain(**base, allow_phrase_octave=True)
        self.assertEqual((ordinary.octave_shift, ordinary.virtual_lift), (0, 0))
        self.assertEqual((raised.octave_shift, raised.virtual_lift), (12, 0))
        self.assertEqual(raised.reason, "sustained_low_register_octave")

    def test_phrase_octave_protects_sparse_low_frames_high_notes_and_manual_shift(self):
        base = dict(
            tracks=_tracks([51.0] * 40 + [54.0] * 75 + [59.0] * 85),
            pitch_range=[55, 60, 65], pitch_tails=[54, 67], manual_shift=0,
            virtual_available=True, allow_octave=False, voiced_share=0.8,
            source_anchor=None, target_anchor=None, protect_formants=False, allow_phrase_octave=True,
        )
        for override in (
            {"voiced_share": 0.45},
            {"tracks": _tracks([51.0] * 99)},
            {"tracks": _tracks([51.0] * 20 + [57.0] * 180)},
            {"tracks": _tracks([51.0] * 40 + [54.0] * 75 + [65.0] * 85)},
            {"tracks": _tracks([51.0] * 100 + [59.0] * 90, confidence=0.2)},
        ):
            with self.subTest(override=list(override)):
                self.assertEqual(domain.plan_domain(**{**base, **override}).octave_shift, 0)
        manual = domain.plan_domain(**{**base, "manual_shift": -2})
        self.assertEqual((manual.octave_shift, manual.virtual_lift), (-2, 0))

    def test_formant_dead_zone_damping_and_clamps(self):
        dead_zone = domain.plan_domain(
            [], [], 0, virtual_available=False, allow_octave=False, voiced_share=0,
            source_anchor=1000, target_anchor=1070, protect_formants=False,
        )
        normal = domain.plan_domain(
            [], [], 0, virtual_available=False, allow_octave=False, voiced_share=0,
            source_anchor=1000, target_anchor=1200, protect_formants=False,
        )
        protected = domain.plan_domain(
            [], [], 0, virtual_available=False, allow_octave=False, voiced_share=0,
            source_anchor=1000, target_anchor=1200, protect_formants=True,
        )
        clamped = domain.plan_domain(
            [], [], 0, virtual_available=False, allow_octave=False, voiced_share=0,
            source_anchor=1000, target_anchor=3500, protect_formants=False,
        )
        missing = domain.plan_domain(
            [], [], 0, virtual_available=False, allow_octave=False, voiced_share=0,
            source_anchor=None, target_anchor=1200, protect_formants=False,
        )
        self.assertEqual(dead_zone.formant_ratio, 1.0)
        self.assertAlmostEqual(normal.formant_ratio, 1.091)
        self.assertAlmostEqual(protected.formant_ratio, 1.0455)
        self.assertEqual(clamped.formant_ratio, 1.22)
        self.assertEqual(missing.formant_ratio, 1.0)


class FormantTests(unittest.TestCase):
    def test_formant_frames_are_confidence_gated_burg_geometric_anchors(self):
        class FakeSound:
            def __init__(self, samples, sampling_frequency):
                self.samples = samples
                self.sampling_frequency = sampling_frequency

            def to_formant_burg(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(get_value_at_time=lambda number, time: {1: 500, 2: 1500, 3: 3000}[number])

        fake = SimpleNamespace(Sound=FakeSound)
        f0 = np.full(5, 220.0)
        confidence = np.array([0.9, 0.2, 0.8, 0.0, 0.95])
        with patch.dict(sys.modules, {"parselmouth": fake}):
            anchors = domain.formant_frames(np.ones(5 * 441, dtype=np.float32), (f0, confidence))
        self.assertEqual(len(anchors), 3)
        np.testing.assert_allclose(anchors, np.exp(np.mean(np.log([500, 1500, 3000]))))

    def test_summary_withholds_unstable_or_short_anchor(self):
        short_anchor, short_count, short_mad = domain.summarize_formants([900.0] * 29)
        noisy_anchor, noisy_count, noisy_mad = domain.summarize_formants([500.0] * 15 + [1800.0] * 15)
        good_anchor, good_count, good_mad = domain.summarize_formants([900.0] * 60)
        self.assertIsNone(short_anchor)
        self.assertEqual(short_count, 29)
        self.assertIsNotNone(short_mad)
        self.assertIsNone(noisy_anchor)
        self.assertEqual(noisy_count, 30)
        self.assertGreater(noisy_mad, domain.MAX_FORMANT_LOG_MAD)
        self.assertAlmostEqual(good_anchor, 900.0)
        self.assertEqual(good_count, 60)
        self.assertEqual(good_mad, 0.0)

    def test_formant_warp_is_same_length_float32_and_invalid_ratio_falls_back(self):
        time = np.arange(16000, dtype=np.float32) / 16000
        samples = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        shifted = domain.warp_formants(samples, 1.15, 16000)
        fallback = domain.warp_formants(samples, 1.5, 16000)
        self.assertEqual((len(shifted), shifted.dtype), (len(samples), np.float32))
        self.assertTrue(np.isfinite(shifted).all())
        np.testing.assert_array_equal(fallback, samples)

    def test_worker_warps_only_supported_contentvec_input_and_rejects_noop(self):
        worker = object.__new__(_Rift)
        worker.np = np
        worker.torch = SimpleNamespace(is_tensor=lambda value: False)
        time = np.arange(16000, dtype=np.float32) / 16000
        samples = sum(
            amplitude * np.sin(2 * np.pi * frequency * time)
            for frequency, amplitude in ((180, 0.2), (520, 0.12), (1450, 0.08), (2700, 0.04))
        ).astype(np.float32)
        warped = worker._warp_contentvec_audio(samples, 1.15)
        self.assertEqual((len(warped), warped.dtype), (len(samples), np.float32))
        self.assertGreater(float(np.sqrt(np.mean((warped - samples) ** 2))), 1e-6)
        with patch("music_runtime.workers.rift_domain.warp_formants", return_value=samples.copy()):
            with self.assertRaisesRegex(ValueError, "left non-silent audio unchanged"):
                worker._warp_contentvec_audio(samples, 1.15)


class ProfileV2Tests(unittest.TestCase):
    def test_v1_and_v2_profiles_validate_and_v1_remains_readable(self):
        validate_profile(_base_profile(1), "target")
        validate_profile(_base_profile(2), "target")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v1.json"
            path.write_text(json.dumps(_base_profile(1)), encoding="utf-8")
            profile, digest = read_profile(path, "target")
        self.assertEqual(profile["version"], 1)
        self.assertEqual(len(digest), 64)

    def test_v2_rejects_invalid_tail_or_formant_statistics(self):
        profile = _base_profile(2)
        profile["pitch_tails"] = [56.0, 66.0]
        with self.assertRaises(ValueError):
            validate_profile(profile, "target")


class DomainSettingsTests(unittest.TestCase):
    def test_domain_flags_default_and_require_booleans(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = validate_settings({}, Path(folder), require_paths=False)
            self.assertEqual(
                (settings["rift"]["domain_alignment"], settings["rift"]["domain_formants"],
                 settings["rift"]["allow_octave_jump"]),
                (True, True, False),
            )
            with self.assertRaises(PipelineError):
                validate_settings({"rift": {"domain_alignment": 1}}, Path(folder), require_paths=False)
        profile = _base_profile(2)
        profile["formant_anchor_hz"] = 10000.0
        with self.assertRaises(ValueError):
            validate_profile(profile, "target")


if __name__ == "__main__":
    unittest.main()
