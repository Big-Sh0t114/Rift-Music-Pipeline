from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from music_runtime.audio import limit_master_true_peak, mix_wavs, wav_info
from music_runtime.runtime import MusicRuntime
from music_runtime.settings import PipelineError, load_settings, save_settings, validate_settings

try:
    from music_runtime.workers import rift_quality as quality
except ModuleNotFoundError as exc:
    if exc.name not in {"numpy", "scipy", "pyloudnorm"}:
        raise
    quality = None


@unittest.skipIf(quality is None, "Use the isolated RIFT environment for these DSP checks")
class AdaptiveQualityTests(unittest.TestCase):
    def test_vad_pause_sets_boundary_and_unrecognised_singing_keeps_coverage(self):
        from music_runtime.workers.rift_sections import phrase_boundaries

        np = quality.np
        rate = quality.RATE
        samples = np.r_[np.full(3 * rate, 0.12), np.zeros(rate // 3), np.full(3 * rate, 0.009)]
        probabilities = np.zeros(int(np.ceil(len(samples) / rate / 0.032)))
        phrases = phrase_boundaries(samples, probabilities)
        self.assertEqual(len(phrases), 2)
        self.assertEqual(phrases[1].boundary, "breath")
        self.assertAlmostEqual(phrases[1].start / rate, 3.16, delta=0.07)
        self.assertEqual(phrases[-1].end, len(samples))
        held = np.full(15 * rate, 0.05)
        held[10 * rate:10 * rate + rate // 20] = 0.015
        long_phrases = phrase_boundaries(held, np.ones(470))
        self.assertEqual(long_phrases[1].boundary, "energy_minimum")
        self.assertAlmostEqual(long_phrases[1].start / rate, 10.0, delta=0.1)

    def test_clustered_phrases_merge_by_style_and_borrow_real_context(self):
        from music_runtime.workers.rift_sections import Phrase, cluster_phrases, merge_phrases

        np = quality.np
        rate = quality.RATE
        spans = [Phrase(i * 3 * rate, (i + 1) * 3 * rate, "breath") for i in range(4)]
        vectors = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        labels = cluster_phrases(spans, vectors, np.zeros((4, 4)))
        self.assertEqual(labels.tolist(), [0, 0, 1, 1])
        self.assertEqual([x.phrases for x in merge_phrases(spans, labels)], [(0, 1), (2, 3)])
        self.assertEqual(cluster_phrases(spans, np.tile(vectors[0], (4, 1)), np.zeros((4, 4))).tolist(), [0] * 4)

        t = np.arange(4 * rate) / rate
        source = np.r_[0.2 * np.sin(2 * np.pi * 150 * t), np.zeros(rate // 3),
                       0.2 * np.sin(2 * np.pi * 550 * t)].astype(np.float32)
        calls = []

        embeddings = iter([(1, 0), (0, 1)])

        class Models:
            def probabilities(self, audio16):
                return np.zeros(int(np.ceil(len(audio16) / 512)))

            def embedding(self, _):
                return np.array(next(embeddings))

        def pitch(samples):
            count = len(samples) // 441 + 1
            return np.full(count, 220.0), np.ones(count)

        def infer(segments, guidance, shift, **kwargs):
            calls.append(segments)
            return [(start, audio, len(audio)) for start, audio in segments]

        rendered, report = quality.convert_adaptive(
            source, None, pitch, infer,
            {"key_shift": 0, "pitch_range": [], "robust_f0": 1, "spk_cfg_strength": 0.55,
             "ds_cfg_strength": 0.1, "cfg_rescale": 0.7, "timbre_slicing": True}, phrase_models=Models(),
        )
        self.assertEqual([len(segments) for segments in calls], [1, 1])
        self.assertEqual(len(rendered), len(source))
        self.assertEqual([entry["cluster"] for entry in report["sections"]], [0, 1])
        self.assertEqual(report["sections"][0]["context_after_seconds"], 0.3)
        self.assertEqual(report["sections"][1]["context_before_seconds"], 0.3)
        self.assertLess(calls[1][0][0] / rate, report["sections"][1]["start_seconds"])
        json.dumps(report, allow_nan=False)

    def test_profile_matches_broad_voiced_spectrum_and_preserves_duration(self):
        np = quality.np
        timbre = quality.timbre
        t = np.arange(quality.RATE, dtype=np.float32) / quality.RATE
        samples = np.random.default_rng(235).normal(0, 0.05, len(t)).astype(np.float32)
        voiced = (np.ones(101) * 220.0, np.ones(101))
        spectrum = timbre.extract_spectrum(samples, voiced)
        reference = spectrum.relative_db + np.linspace(-10, 10, 18)
        matched, drift, largest = timbre.match_spectrum(samples, spectrum, reference)
        updated = timbre.extract_spectrum(matched, voiced)
        self.assertLess(np.linalg.norm(reference - updated.relative_db), np.linalg.norm(reference - spectrum.relative_db))
        self.assertGreater(drift, 0)
        self.assertLessEqual(largest, 6.0)
        self.assertEqual(len(matched), len(samples))
        self.assertIsNone(timbre.extract_spectrum(samples, (np.zeros(101), np.ones(101))))

    def test_phrase_retry_can_reduce_high_band_without_losing_voicing(self):
        np = quality.np
        t = np.arange(quality.RATE * 2) / quality.RATE
        dry = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        calls = []

        class Models:
            def probabilities(self, audio16):
                return np.ones(int(np.ceil(len(audio16) / 512)))

            def embedding(self, _):
                return np.array([1.0, 0.0])

        def pitch(audio):
            frames = len(audio) // 441 + 1
            return np.full(frames, 220.0), np.ones(frames)

        def infer(segments, guidance, shift, **kwargs):
            calls.append(guidance)
            start, signal = segments[0]
            output = signal + 0.18 * np.sin(2 * np.pi * 6000 * t) if len(calls) == 1 else signal
            return [(start, output, len(output))]

        _, report = quality.convert_adaptive(
            dry, None, pitch, infer, {"timbre_slicing": True, "key_shift": 0, "pitch_range": [], "robust_f0": 1,
                                      "spk_cfg_strength": 0.55, "ds_cfg_strength": 0.1, "cfg_rescale": 0.7},
            phrase_models=Models(),
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1].detail, 0.0)
        self.assertTrue(report["sections"][0]["accepted_retry"])

    def test_failed_quality_retry_preserves_the_first_valid_phrase_render(self):
        np = quality.np
        t = np.arange(quality.RATE * 2, dtype=np.float32) / quality.RATE
        dry = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        calls = []

        class Models:
            def probabilities(self, audio16):
                return np.ones(int(np.ceil(len(audio16) / 512)))

            def embedding(self, _):
                return np.array([1.0, 0.0])

        def pitch(audio):
            frames = len(audio) // 441 + 1
            return np.full(frames, 220.0), np.ones(frames)

        def infer(segments, guidance, shift, **kwargs):
            calls.append(segments)
            if len(calls) > 1:
                raise RuntimeError("retry fixture failure")
            start, source = segments[0]
            output = source + 0.18 * np.sin(2 * np.pi * 6000 * np.arange(len(source)) / quality.RATE)
            return [(start, output.astype(np.float32), len(source))]

        _, report = quality.convert_adaptive(
            dry, None, pitch, infer,
            {"timbre_slicing": True, "key_shift": 0, "pitch_range": [], "robust_f0": 1,
             "spk_cfg_strength": 0.55, "ds_cfg_strength": 0.1, "cfg_rescale": 0.7},
            phrase_models=Models(),
        )
        self.assertEqual(len(calls), 2)
        self.assertTrue(report["sections"][0]["valid"])
        self.assertFalse(report["sections"][0]["accepted_retry"])
        self.assertEqual(report["sections"][0]["retry_domain_fallback_reason"], "retry_render_failed:RuntimeError")

    def test_virtual_lifts_split_merged_sections_only_at_natural_phrase_boundaries(self):
        from music_runtime.workers import rift_phrase_quality as phrase_quality
        from music_runtime.workers.rift_sections import Phrase, Section

        np = quality.np
        phrases = [Phrase(0, 100, "breath"), Phrase(100, 220, "breath"), Phrase(220, 300, "breath")]
        tracks = [(np.full(10, 220.0), np.ones(10)) for _ in phrases]
        clusters = {0: phrase_quality.Cluster(0, 0.0, "clean", quality.Guidance(1, 0.5, 0.1, 0.7),
                                              None, None, 0.0, 1.0)}
        lifted = [phrase_quality.PhraseDomain(4, "pitch_tails_improved"),
                  phrase_quality.PhraseDomain(0, "pitch_improvement_below_floor"),
                  phrase_quality.PhraseDomain(0, "pitch_improvement_below_floor")]
        sections, plans = phrase_quality._split_planned_sections(
            [Section(0, 300, 0, (0, 1, 2))], phrases, lifted, tracks,
            np.zeros(300, dtype=np.float32), clusters, None, {"domain_formants": False},
        )
        self.assertEqual([(value.start, value.end, value.phrases) for value in sections],
                         [(0, 100, (0,)), (100, 300, (1, 2))])
        self.assertEqual([plan.virtual_lift_candidate for plan in plans], [4, 0])

    def test_retained_low_octave_uses_final_pitch_without_a_rollback(self):
        from music_runtime.workers import rift_phrase_quality as phrase_quality
        from music_runtime.workers.rift_sections import Phrase

        np = quality.np
        rate = quality.RATE
        source = (0.12 * np.sin(2 * np.pi * 180 * np.arange(4 * rate) / rate)).astype(np.float32)
        notes = [51] * 40 + [54] * 75 + [59] * 85
        calls = []

        class Models:
            def probabilities(self, _):
                return np.ones(1)

            def embedding(self, _):
                return np.array([1.0, 0.0])

        def pitch(audio):
            midi = notes if len(audio) == 2 * rate else [61] * ((len(audio) // 441) + 1)
            f0 = 440 * 2 ** ((np.asarray(midi) - 69) / 12)
            return f0, np.ones(len(f0))

        def infer(segments, _, shift, **_options):
            calls.append(shift)
            return [(index, audio, len(audio)) for index, audio in segments]

        settings = {"pitch_range": [55, 60, 65], "key_shift": 0, "domain_alignment": True,
                    "allow_octave_jump": True, "timbre_slicing": True, "robust_f0": 1,
                    "spk_cfg_strength": 0.55, "ds_cfg_strength": 0.1, "cfg_rescale": 0.7}
        bounds = [Phrase(0, 2 * rate, "pause"), Phrase(2 * rate, 4 * rate - 441, "breath")]
        with patch.object(phrase_quality, "phrase_boundaries", return_value=bounds), \
                patch.object(phrase_quality, "cluster_phrases", return_value=np.zeros(2, dtype=int)), \
                patch.object(phrase_quality, "_retune_candidate", side_effect=AssertionError("unexpected rollback")) as retune:
            rendered, report = phrase_quality.convert_phrases(source, pitch, infer, settings, Models(), None)
        retune.assert_not_called()
        self.assertEqual(len(rendered), len(source))
        self.assertEqual(calls, [12, 0])
        self.assertEqual([s["domain_octave_shift_applied"] for s in report["sections"]], [12, 0])
        self.assertEqual(report["sections"][0]["domain_reason"], "sustained_low_register_octave")

    def test_song_octave_lock_is_global_and_manual_key_shift_has_priority(self):
        from music_runtime.workers import rift_phrase_quality as phrase_quality
        from music_runtime.workers.rift_sections import Phrase

        np = quality.np
        rate = quality.RATE
        source = (0.12 * np.sin(2 * np.pi * 180 * np.arange(4 * rate) / rate)).astype(np.float32)
        bounds = [Phrase(0, 2 * rate, "breath"), Phrase(2 * rate, 4 * rate, "breath")]

        class Models:
            def probabilities(self, audio16):
                return np.ones(int(np.ceil(len(audio16) / 512)))

            def embedding(self, _):
                return np.array([1.0, 0.0])

        def run_case(key_shift):
            pitch_calls = 0
            shifts = []

            def pitch(audio):
                nonlocal pitch_calls
                pitch_calls += 1
                # The first cluster would independently shift up an octave;
                # the second cluster is already inside the target range.
                note = 47 if pitch_calls in (1, 3, 4) else 61
                frames = len(audio) // 441 + 1
                f0 = np.full(frames, 440 * 2 ** ((note - 69) / 12))
                return f0, np.ones(frames)

            def infer(segments, _, shift, **_options):
                shifts.append(shift)
                return [(start, audio.copy(), len(audio)) for start, audio in segments]

            settings = {"pitch_range": [55, 60, 65], "key_shift": key_shift,
                        "song_octave_shift": 12, "domain_alignment": True,
                        "allow_octave_jump": True, "allow_song_octave_detection": True,
                        "timbre_slicing": True, "robust_f0": 1, "spk_cfg_strength": 0.55,
                        "ds_cfg_strength": 0.1, "cfg_rescale": 0.7}
            with patch.object(phrase_quality, "phrase_boundaries", return_value=bounds), \
                    patch.object(phrase_quality, "cluster_phrases", return_value=np.array([0, 1])), \
                    patch.object(phrase_quality, "_retune_candidate", side_effect=AssertionError("locked shift must not roll back")) as retune:
                rendered, report = phrase_quality.convert_phrases(source, pitch, infer, settings, Models(), None)
            retune.assert_not_called()
            self.assertEqual(len(rendered), len(source))
            return shifts, report

        shifts, locked = run_case(0)
        self.assertEqual(shifts, [12, 12])
        self.assertEqual(locked["song_octave_shift"], 12)
        self.assertEqual(locked["song_octave_source"], "locked")
        self.assertEqual([section["domain_rift_key_shift_applied"] for section in locked["sections"]], [12, 12])
        self.assertEqual([section["domain_virtual_lift_applied"] for section in locked["sections"]], [0, 0])
        self.assertEqual([section["domain_octave_shift_applied"] for section in locked["sections"]], [12, 12])

        shifts, manual = run_case(3)
        self.assertEqual(shifts, [3, 3])
        self.assertEqual(manual["song_octave_shift"], 0)
        self.assertIsNone(manual["song_octave_source"])
        self.assertEqual([section["domain_rift_key_shift_applied"] for section in manual["sections"]], [3, 3])

    def _run_mocked_song_octave_rescue(self, *, full_dropout=False, fail_same_shift_retry=False,
                                       fail_rescue_retunes=False):
        from music_runtime.workers import rift_phrase_quality as phrase_quality
        from music_runtime.workers.rift_sections import Phrase

        np = quality.np
        rate = quality.RATE
        source = (0.12 * np.sin(2 * np.pi * 220 * np.arange(6 * rate) / rate)).astype(np.float32)
        calls, retunes = [], []

        class Models:
            def probabilities(self, audio16):
                return np.ones(int(np.ceil(len(audio16) / 512)))

            def embedding(self, _):
                return np.array([1.0, 0.0])

        def pitch(audio):
            count = len(audio) // 441 + 1
            if len(audio) and audio[0] > 0.5:
                f0, confidence = np.zeros(count), np.zeros(count)
                for frame in range(count):
                    start, end = frame * 441, min(len(audio), (frame + 1) * 441)
                    if end > start and float(np.mean(audio[start:end])) > 0.5:
                        f0[frame], confidence[frame] = 440.0, 1.0
                return f0, confidence
            return np.full(count, 220.0), np.ones(count)

        def infer(segments, _guidance, shift, **_options):
            calls.append(shift)
            if full_dropout and shift == 12:
                if fail_same_shift_retry and calls.count(12) == 2:
                    raise RuntimeError("same-shift retry failed")
                return [(start, np.zeros_like(audio), len(audio)) for start, audio in segments]
            code = 1.0 if shift == 12 else 2.0
            outputs = []
            for start, audio in segments:
                generated = np.full_like(audio, code)
                if not full_dropout and shift == 12:
                    generated[200 * 441:270 * 441] = 0.0
                outputs.append((start, generated, len(generated)))
            return outputs

        def retune(audio, lift, _ffmpeg, _pitch_fn):
            retunes.append(lift)
            if fail_rescue_retunes:
                raise RuntimeError("Rubber Band pitch check disagreed with the requested lift")
            return audio.copy(), {"length_drift_samples": 0, "f0_shift_semitones": -float(lift)}

        settings = {
            "timbre_slicing": True, "song_octave_shift": 12, "key_shift": 0,
            "pitch_range": [], "domain_alignment": False, "domain_formants": False,
            "domain_ffmpeg": "mock-ffmpeg", "robust_f0": 1, "spk_cfg_strength": 0.55,
            "ds_cfg_strength": 0.1, "cfg_rescale": 0.7,
        }
        with patch.object(phrase_quality, "phrase_boundaries",
                          return_value=[Phrase(0, len(source), "breath")]), \
             patch.object(phrase_quality, "cluster_phrases", return_value=np.zeros(1, dtype=int)), \
             patch.object(phrase_quality, "_retune_candidate", side_effect=retune):
            rendered, report = quality.convert_adaptive(
                source, None, pitch, infer, settings, phrase_models=Models(),
            )
        return rendered, report, calls, retunes

    def test_song_plus_twelve_rescue_catches_local_dropout_and_restores_final_pitch(self):
        rendered, report, calls, retunes = self._run_mocked_song_octave_rescue()
        section = report["sections"][0]
        rescue = section["domain_upper_register_rescue"]
        self.assertEqual(calls, [12, 12, 8])
        self.assertEqual(retunes, [-4])
        self.assertEqual(len(rendered), quality.RATE * 6)
        self.assertTrue(rescue["candidate_attempted"])
        self.assertTrue(rescue["candidate_applied"])
        self.assertEqual(rescue["trigger_reasons"], ["localized_voiced_dropout"])
        self.assertGreater(rescue["before"]["worst_local_voiced_loss"], 0.4)
        self.assertEqual(rescue["rift_key_shift_applied"], 8)
        self.assertAlmostEqual(rescue["post_shift_semitones"], 12.0, places=2)
        self.assertEqual(section["domain_rift_key_shift_applied"], 8)
        self.assertEqual(section["domain_octave_shift_applied"], 12)
        self.assertLessEqual(rescue["after"]["voiced_frame_loss"], 0.08)
        self.assertGreaterEqual(rescue["after"]["expected_octave_coverage"], 0.65)

    def test_complete_rescue_candidate_beats_higher_scoring_incomplete_candidate(self):
        from music_runtime.workers import rift_phrase_quality as phrase_quality

        def metrics(coverage, expected, local_loss):
            return {
                "source_voiced_frames": 600, "source_voiced_ratio": 1.0,
                "voiced_coverage": coverage, "voiced_frame_loss": 1.0 - coverage,
                "expected_octave_coverage": expected, "post_shift_semitones": 12.0,
                "worst_local_voiced_loss": local_loss,
                "worst_local_window_seconds": [1.5, 3.0], "local_windows_checked": 7,
            }

        primary = metrics(0.70, 0.55, 0.80)
        higher_scoring_incomplete = metrics(0.91, 0.95, 0.20)
        lower_scoring_complete = metrics(0.94, 0.80, 0.30)
        with patch.object(phrase_quality, "_upper_register_metrics",
                          side_effect=[primary, higher_scoring_incomplete, lower_scoring_complete]), \
             patch.object(phrase_quality, "_rescue_candidate_quality",
                          side_effect=[(True, None, False), (True, None, True)]), \
             patch.object(phrase_quality, "_rescue_score", side_effect=[0.906, 0.829]):
            _, report, calls, retunes = self._run_mocked_song_octave_rescue()

        section = report["sections"][0]
        rescue = section["domain_upper_register_rescue"]
        self.assertEqual(calls, [12, 12, 8, 4])
        self.assertEqual(retunes, [-4, -8])
        self.assertEqual([candidate["rift_shift_semitones"] for candidate in rescue["candidates"]], [8, 4])
        self.assertFalse(rescue["candidates"][0]["candidate_applied"])
        self.assertTrue(rescue["candidates"][1]["candidate_applied"])
        self.assertEqual(section["domain_rift_key_shift_applied"], 4)
        self.assertEqual(section["domain_octave_shift_applied"], 12)
        self.assertEqual(rescue["after"]["voiced_coverage"], 0.94)
        self.assertEqual(rescue["after"]["expected_octave_coverage"], 0.80)

    def test_song_plus_twelve_rescue_can_replace_an_unusable_primary_after_retry_error(self):
        rendered, report, calls, retunes = self._run_mocked_song_octave_rescue(
            full_dropout=True, fail_same_shift_retry=True,
        )
        section = report["sections"][0]
        rescue = section["domain_upper_register_rescue"]
        self.assertEqual(calls, [12, 12, 8])
        self.assertEqual(retunes, [-4])
        self.assertEqual(len(rendered), quality.RATE * 6)
        self.assertTrue(section["valid"])
        self.assertTrue(rescue["candidate_applied"])
        self.assertEqual(rescue["before"]["voiced_coverage"], 0.0)
        self.assertEqual(section["domain_rift_key_shift_applied"], 8)
        self.assertEqual(section["domain_octave_shift_applied"], 12)

    def test_failed_optional_rescue_retunes_preserve_the_valid_plus_twelve_primary(self):
        rendered, report, calls, retunes = self._run_mocked_song_octave_rescue(
            fail_rescue_retunes=True,
        )
        section = report["sections"][0]
        rescue = section["domain_upper_register_rescue"]
        self.assertEqual(calls, [12, 12, 8, 4, 0])
        self.assertEqual(retunes, [-4, -8, -12])
        self.assertEqual(len(rendered), quality.RATE * 6)
        self.assertTrue(section["valid"])
        self.assertFalse(rescue["candidate_applied"])
        self.assertEqual(rescue["rejection_reason"], "candidate_render_failed:RuntimeError")
        self.assertEqual(section["domain_rift_key_shift_applied"], 12)
        self.assertEqual(section["domain_octave_shift_applied"], 12)

    def test_automatic_song_octave_gate_applies_only_after_phrase_consensus(self):
        from music_runtime.workers import rift_phrase_quality as phrase_quality
        from music_runtime.workers.rift_sections import Phrase

        np = quality.np
        rate = quality.RATE
        source = (0.12 * np.sin(2 * np.pi * 104 * np.arange(8 * rate) / rate)).astype(np.float32)
        bounds = [Phrase(i * 2 * rate, (i + 1) * 2 * rate, "breath") for i in range(4)]
        shifts = []

        class Models:
            def probabilities(self, audio16):
                return np.ones(int(np.ceil(len(audio16) / 512)))

            def embedding(self, _):
                return np.array([1.0, 0.0])

        def pitch(audio):
            frames = len(audio) // 441 + 1
            return np.full(frames, 440 * 2 ** ((44 - 69) / 12)), np.ones(frames)

        def infer(segments, _, shift, **_options):
            shifts.append(shift)
            return [(start, audio.copy(), len(audio)) for start, audio in segments]

        settings = {"pitch_range": [55, 60, 65], "key_shift": 0, "domain_alignment": True,
                    "allow_octave_jump": True, "allow_song_octave_detection": True,
                    "timbre_slicing": True, "robust_f0": 1, "spk_cfg_strength": 0.55,
                    "ds_cfg_strength": 0.1, "cfg_rescale": 0.7}
        with patch.object(phrase_quality, "phrase_boundaries", return_value=bounds), \
                patch.object(phrase_quality, "cluster_phrases", return_value=np.zeros(4, dtype=int)):
            _, report = phrase_quality.convert_phrases(source, pitch, infer, settings, Models(), None)
        self.assertEqual(shifts, [12])
        self.assertEqual(report["song_octave_shift"], 12)
        self.assertEqual(report["song_octave_source"], "automatic")
        self.assertEqual(report["song_octave_diagnostics"]["reliable_phrases"], 4)
        self.assertEqual(report["sections"][0]["domain_virtual_lift_applied"], 0)

    def test_cluster_octave_requires_compatible_pitch_for_each_phrase(self):
        from music_runtime.workers import rift_domain, rift_phrase_quality

        np = quality.np
        settings = {"pitch_range": [55, 60, 65], "key_shift": 0, "domain_alignment": True,
                    "allow_octave_jump": True, "domain_ffmpeg": "ffmpeg.exe"}
        plans = []
        for note in (47, 61):
            track = (np.full(200, 440 * 2 ** ((note - 69) / 12)), np.ones(200))
            plan = rift_phrase_quality._phrase_domain_plan(rift_domain, None, track, 12, None, settings)
            plans.append(plan)
        self.assertEqual([plan.octave_shift for plan in plans], [12, 0])
        self.assertEqual(plans[1].reason, "cluster_octave_rejected")

    def test_failed_or_worse_structural_fallback_preserves_a_valid_lifted_render(self):
        from music_runtime.workers import rift_phrase_quality as phrase_quality

        np = quality.np
        source = (0.12 * np.sin(2 * np.pi * 220 * np.arange(quality.RATE * 2) / quality.RATE)).astype(np.float32)

        class Models:
            def probabilities(self, audio16):
                return np.ones(int(np.ceil(len(audio16) / 512)))

            def embedding(self, _):
                return np.array([1.0, 0.0])

        def run_case(fail_fallback):
            pitch_calls, shifts = 0, []

            def pitch(audio):
                nonlocal pitch_calls
                pitch_calls += 1
                frames = len(audio) // 441 + 1
                value = 220.0 if pitch_calls <= 2 else 0.0
                return np.full(frames, value), np.ones(frames)

            def infer(segments, guidance, shift, **kwargs):
                shifts.append(shift)
                if fail_fallback and len(shifts) == 2:
                    raise RuntimeError("unlifted fallback failed")
                start, prepared = segments[0]
                return [(start, prepared.copy(), len(prepared))]

            domain = SimpleNamespace(plan_domain=lambda tracks, pitch_range, manual_shift, **kwargs: SimpleNamespace(
                octave_shift=manual_shift, virtual_lift=3 if kwargs["virtual_available"] else 0,
                formant_ratio=1.0, reason="fixture_plan", predicted_outside_ratio=0.0,
            ))
            settings = {
                "timbre_slicing": True, "key_shift": 0, "pitch_range": [], "robust_f0": 1,
                "spk_cfg_strength": 0.55, "ds_cfg_strength": 0.1, "cfg_rescale": 0.7,
                "domain_alignment": True, "domain_formants": False, "domain_ffmpeg": "ffmpeg.exe",
                "allow_octave_jump": False,
            }
            with patch.object(phrase_quality, "_domain_module", return_value=domain), \
                 patch.object(phrase_quality, "_retune_candidate", side_effect=lambda audio, lift, ffmpeg, pitch_fn:
                              (audio.copy(), {"length_drift_samples": 0, "f0_shift_semitones": 0.0})):
                _, report = quality.convert_adaptive(source, None, pitch, infer, settings, phrase_models=Models())
            return shifts, report["sections"][0]

        for fail_fallback, expected_reason in (
            (False, "fallback_did_not_improve_voicing"),
            (True, "fallback_render_failed:RuntimeError"),
        ):
            shifts, section = run_case(fail_fallback)
            self.assertEqual(shifts, [3, 0, 3])
            self.assertTrue(section["valid"])
            self.assertEqual(section["domain_virtual_lift_applied"], 3)
            self.assertEqual(section["domain_virtual_fallback_rejected_reason"], expected_reason)

    def test_accepted_raw_contentvec_retry_reports_applied_zero_alpha(self):
        from music_runtime.workers import rift_phrase_quality as phrase_quality

        np = quality.np
        t = np.arange(quality.RATE * 2, dtype=np.float32) / quality.RATE
        source = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        calls = []

        class Models:
            def probabilities(self, audio16):
                return np.ones(int(np.ceil(len(audio16) / 512)))

            def embedding(self, _):
                return np.array([1.0, 0.0])

        def pitch(audio):
            frames = len(audio) // 441 + 1
            return np.full(frames, 220.0), np.ones(frames)

        def content_features(audio, *, formant_ratio=1.0):
            features = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (100, 1))
            times = np.linspace(0, len(audio) / quality.RATE, len(features), endpoint=False)
            return features, times

        def infer(segments, guidance, shift, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise RuntimeError("requested-ratio transform failed")
            start, prepared = segments[0]
            if len(calls) == 1:
                high = 0.18 * np.sin(2 * np.pi * 6000 * np.arange(len(prepared)) / quality.RATE)
                return [(start, (prepared + high).astype(np.float32), len(prepared))]
            return [(start, prepared.copy(), len(prepared))]

        profile = {"speaker": "target", "pitch_range": [55, 60, 65],
                   "ltas_db": np.linspace(-24, 24, 18).tolist(),
                   "cvec_mean": [0.0, 0.0], "cvec_std": [1.0, 1.0]}
        domain = SimpleNamespace(
            formant_frames=lambda samples, track: np.full(40, 1000.0),
            summarize_formants=lambda values: (1000.0, len(values), 0.1),
            plan_domain=lambda tracks, pitch_range, manual_shift, **kwargs: SimpleNamespace(
                octave_shift=manual_shift, virtual_lift=0, formant_ratio=1.1,
                reason="fixture_formant_plan", predicted_outside_ratio=0.0,
            ),
        )
        settings = {"timbre_slicing": True, "key_shift": 0, "pitch_range": [], "robust_f0": 1,
                    "spk_cfg_strength": 0.55, "ds_cfg_strength": 0.1, "cfg_rescale": 0.7,
                    "domain_alignment": False, "domain_formants": True, "voice_profile": profile}
        with patch.object(phrase_quality, "_domain_module", return_value=domain), \
             patch.object(phrase_quality, "_protect_formants", return_value=False):
            _, report = quality.convert_adaptive(source, None, pitch, infer, settings,
                                                  phrase_models=Models(), content_features=content_features)
        section = report["sections"][0]
        self.assertGreater(calls[0]["cvec_alpha"], 0.0)
        self.assertEqual(calls[1]["formant_ratio"], 1.1)
        self.assertEqual(calls[2]["cvec_alpha"], 0.0)
        self.assertTrue(section["accepted_retry"])
        self.assertEqual(section["cvec_alpha"], 0.0)
        self.assertEqual(section["domain_formant_ratio_applied"], 1.0)

    def test_voice_profile_routes_drift_inside_inference_and_records_actual_settings(self):
        np = quality.np
        t = np.arange(quality.RATE, dtype=np.float32) / quality.RATE
        sample = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        calls = []

        def pitch(audio):
            return np.full(len(audio) // 441 + 1, 220.0), np.ones(len(audio) // 441 + 1)

        def infer(segments, guidance, shift, **options):
            calls.append((guidance, shift, options))
            return [(start, audio, len(audio)) for start, audio in segments]

        _, report = quality.convert_adaptive(
            sample, lambda audio, _: [(0, audio)], pitch, infer,
            {"key_shift": 0, "pitch_range": [], "robust_f0": 1, "spk_cfg_strength": 0.55,
             "ds_cfg_strength": 0.1, "cfg_rescale": 0.7,
             "voice_profile": {"speaker": "target", "ltas_db": list(np.linspace(-24, 24, 18)), "pitch_range": [59, 61, 62]}},
        )
        guidance, shift, options = calls[0]
        entry = report["chunks"][0]
        self.assertEqual((report["voice_profile"], shift, entry["timbre_route"]), ("target", 0, "high_drift"))
        self.assertGreaterEqual(guidance.speaker, 0.7)
        self.assertLessEqual(guidance.detail, 0.1)
        self.assertLessEqual(entry["ltas_max_eq_db"], 6.0)
        self.assertAlmostEqual(entry["voiced_loudness_lufs"], -18.0, places=1)
        self.assertGreaterEqual(options["cvec_alpha"], 0.3)
        self.assertLessEqual(options["cvec_alpha"], 0.5)
        self.assertTrue(json.loads(json.dumps(report, allow_nan=False))["needs_review"])

    def test_clusters_choose_octaves_and_keep_overlapping_chunks_together(self):
        np = quality.np
        bounds = [(0, 100), (80, 180), (200, 300), (320, 420)]
        tracks = [(np.full(100, freq), np.ones(100)) for freq in (130.81, 130.81, 261.63, 261.63)]
        colors = [600, 800, 1600, 1400]
        labels, shifts, loss = quality.octave_plan(bounds, tracks, colors, [57, 60, 64], 0)
        self.assertEqual(labels.tolist(), [0, 0, 1, 1])
        self.assertEqual(shifts, {0: 12, 1: 0})
        self.assertEqual(loss, 0.0)
        labels, shifts, _ = quality.octave_plan(bounds, tracks, colors, [57, 60, 64], -2)
        self.assertEqual(labels.tolist(), [0, 0, 0, 0])
        self.assertEqual(shifts, {0: -2})
        short_tracks = [*tracks[:2], (tracks[2][0][:15], tracks[2][1][:15]), (tracks[3][0][:15], tracks[3][1][:15])]
        labels, _, _ = quality.octave_plan(bounds, short_tracks, colors, [57, 60, 64], 0)
        self.assertEqual(labels.tolist(), [0, 0, 0, 0])

    def test_adain_is_soft_and_keeps_tensor_shape_and_dtype(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("Use the RIFT environment for the ContentVec alignment check")
        features = torch.tensor([[[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]]], dtype=torch.float16)
        mean = torch.tensor([2.0, 4.0])
        std = torch.tensor([1.2, 1.2])
        aligned = quality.timbre.align_cvec(features, mean, std, 0.4, torch)
        self.assertEqual((aligned.shape, aligned.dtype), (features.shape, features.dtype))
        torch.testing.assert_close(aligned.float().mean(dim=1)[0], torch.tensor([1.4, 3.4]), atol=0.002, rtol=0)

    def test_long_run_is_bounded_and_overlap_preserves_level(self):
        np = quality.np
        samples = np.ones(quality.RATE * 31, dtype=np.float32)
        bounds = list(quality.chunk_bounds([(0, samples)]))
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], len(samples))
        self.assertTrue(all(end - start <= quality.MAX_CHUNK for start, end in bounds))
        self.assertEqual([end - start for (_, end), (start, _) in zip(bounds, bounds[1:])], [quality.OVERLAP] * 2)
        mixed = quality.stitch(len(samples), bounds, [samples[start:end] for start, end in bounds])
        np.testing.assert_allclose(mixed[quality.RATE:-quality.RATE], 1, atol=1e-6)

    def test_antiphase_stereo_has_an_audible_model_input(self):
        np = quality.np
        left = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(quality.downmix(np.column_stack([left, -left])), left)

    def test_only_octaves_are_automatic_and_manual_shift_wins(self):
        np = quality.np
        notes = [(np.full(20, 220.0), np.ones(20))]
        self.assertEqual(quality.choose_shift(notes, [67, 69, 71], 0), (12, 0.0))
        self.assertEqual(quality.choose_shift(notes, [67, 69, 71], -2), (-2, 1.0))

    def test_missing_voiced_frames_trigger_one_retry_and_keep_improvement(self):
        np = quality.np
        t = np.arange(quality.RATE, dtype=np.float32) / quality.RATE
        source = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        calls = []

        def pitch(samples):
            count = len(samples) // 441 + 1
            return np.full(count, 220.0 if quality.rms(samples) > 0.02 else 0), np.ones(count)

        def infer(segments, guidance, shift):
            calls.append(guidance)
            return [(start, samples if len(calls) > 1 else samples * 0.02, len(samples)) for start, samples in segments]

        result, report = quality.convert_adaptive(
            source, lambda audio, _: [(0, audio)], pitch, infer,
            {"key_shift": 0, "pitch_range": [], "robust_f0": 1, "spk_cfg_strength": 0.55,
             "ds_cfg_strength": 0.1, "cfg_rescale": 0.7},
        )
        self.assertEqual(len(calls), 2)
        self.assertLess(calls[1].speaker, calls[0].speaker)
        self.assertEqual(calls[1].robust_f0, 2)
        self.assertTrue(report["chunks"][0]["accepted_retry"])
        self.assertEqual(report["chunks"][0]["voiced_frame_loss"], 0)
        self.assertEqual(len(result), len(source))
        self.assertGreater(quality.rms(result), 0.05)

    def test_high_band_attenuation_is_bounded_and_keeps_timing(self):
        np = quality.np
        t = np.arange(quality.RATE, dtype=np.float32) / quality.RATE
        body = 0.2 * np.sin(2 * np.pi * 220 * t)
        high = np.sin(2 * np.pi * 6000 * t)
        source, output = body + 0.015 * high, body + 0.08 * high
        before = quality.high_band_excess(source, output)
        adjusted, after, cut = quality.tame_high_band(source, output, before)
        self.assertGreater(before, 6)
        self.assertLess(after, before)
        self.assertGreater(cut, 0)
        self.assertLessEqual(cut, 6)
        self.assertEqual(len(adjusted), len(output))
        self.assertAlmostEqual(float(np.dot(adjusted, body)), float(np.dot(output, body)), places=3)

    def test_rubberband_protocol_accepts_bounded_positive_and_negative_length_drift(self):
        from music_runtime.workers import rift_retune

        np = quality.np
        samples = np.linspace(-0.2, 0.2, quality.RATE, dtype=np.float32)
        for drift in (4, -4):
            returned = np.r_[samples, np.zeros(drift, dtype=np.float32)] if drift > 0 else samples[:drift]
            completed = subprocess.CompletedProcess([], 0, returned.astype("<f4").tobytes(), b"")
            with patch.object(rift_retune.subprocess, "run", return_value=completed) as run:
                result = rift_retune.retune_pitch(samples, -4, "ffmpeg.exe", sample_rate=quality.RATE)
            self.assertEqual(len(result.samples), len(samples))
            self.assertEqual(result.drift_samples, drift)
            command = run.call_args.args[0]
            self.assertEqual(run.call_args.kwargs["input"], samples.tobytes())
            self.assertIn("f32le", command)
            self.assertIn("44100", command)
            self.assertIn("-ac", command)
            filter_spec = command[command.index("-af") + 1]
            self.assertIn("formant=preserved", filter_spec)
            self.assertIn("pitchq=quality", filter_spec)
            self.assertIn("transients=smooth", filter_spec)

    def test_rubberband_protocol_rejects_large_drift_and_bad_return(self):
        from music_runtime.workers import rift_retune

        np = quality.np
        samples = np.ones(quality.RATE, dtype=np.float32)
        too_long = np.r_[samples, np.zeros(quality.RATE // 4, dtype=np.float32)]
        completed = subprocess.CompletedProcess([], 0, too_long.astype("<f4").tobytes(), b"")
        with patch.object(rift_retune.subprocess, "run", return_value=completed):
            with self.assertRaises(rift_retune.RetuneError):
                rift_retune.retune_pitch(samples, 2, "ffmpeg.exe", sample_rate=quality.RATE)

    def test_virtual_retune_f0_check_matches_requested_compensation(self):
        from music_runtime.workers import rift_phrase_quality as phrase_quality
        from music_runtime.workers import rift_retune

        np = quality.np
        samples = np.ones(quality.RATE, dtype=np.float32)
        calls = 0

        def pitch(_):
            nonlocal calls
            calls += 1
            frequency = 220.0 if calls == 1 else 220.0 * (2 ** (-3 / 12))
            return np.full(100, frequency), np.ones(100)

        fixture = rift_retune.RetuneResult(samples.copy(), len(samples), len(samples), 0)
        with patch.object(rift_retune, "retune_pitch", return_value=fixture) as retune:
            _, result = phrase_quality._retune_candidate(samples, 3, "ffmpeg.exe", pitch)
        self.assertAlmostEqual(retune.call_args.args[1], -3)
        self.assertAlmostEqual(result["f0_shift_semitones"], -3, places=5)

    def test_worker_formant_warp_is_shape_preserving_and_encoder_scoped(self):
        from music_runtime.workers import rift_domain
        from music_runtime.workers.rift_worker import _Rift

        np = quality.np

        class Torch:
            @staticmethod
            def is_tensor(value):
                return False

        worker = object.__new__(_Rift)
        worker.np = np
        worker.torch = Torch()
        audio = np.sin(np.arange(4096, dtype=np.float32) * 0.07)
        original = audio.copy()
        with patch.object(rift_domain, "warp_formants", side_effect=lambda values, ratio, rate: values * 0.9) as warp:
            result = worker._warp_contentvec_audio(audio, 1.08)
        self.assertEqual(result.shape, audio.shape)
        np.testing.assert_array_equal(audio, original)
        np.testing.assert_allclose(result, original * 0.9)
        self.assertEqual(warp.call_args.args[2], 16000)

        with patch.object(rift_domain, "warp_formants", side_effect=lambda values, ratio, rate: values):
            with self.assertRaises(ValueError):
                worker._warp_contentvec_audio(audio, 1.08)


class QualitySettingsTests(unittest.TestCase):
    def test_profile_round_trips_and_rejects_invalid_percentiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_settings({"rift": {"lead_pitch_range": [55, 62, 72]}}, root)
            self.assertEqual(load_settings(root)["rift"]["lead_pitch_range"], [55.0, 62.0, 72.0])
            with self.assertRaises(PipelineError):
                validate_settings({"rift": {"lead_pitch_range": [72, 62, 55]}}, root)

    def test_role_profile_is_snapshotted_and_changes_its_cache_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.ckpt").write_bytes(b"fixture")
            profile = {"version": 1, "speaker": "target", "encoder_sha256": "a" * 64, "pitch_range": [55, 60, 65],
                       "ltas_db": [0] * 18, "cvec_mean": [0, 1], "cvec_std": [1, 1]}
            path = root / "target.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            settings = validate_settings({"rift": {"checkpoint": "model.ckpt", "speaker": "target",
                                                    "lead_voice_profile": "target.json"}}, root)
            current = MusicRuntime(settings)
            previous = current.model_fingerprints["rift"]
            profile["ltas_db"][0] = 1.0
            path.write_text(json.dumps(profile), encoding="utf-8")
            updated = MusicRuntime(settings).model_fingerprints["rift"]
            self.assertEqual(previous, current.model_fingerprints["rift"])
            self.assertNotEqual(previous["lead"], updated["lead"])
            self.assertEqual(previous["backing"], updated["backing"])
            profile["speaker"] = "different"
            path.write_text(json.dumps(profile), encoding="utf-8")
            self.assertTrue(MusicRuntime(settings)._profile_error)


class TruePeakTests(unittest.TestCase):
    def test_mix_keeps_stereo_and_headroom_at_maximum_gains(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, samples, frames in (("instrument", (-12000, 24000), 4410),
                                          ("lead", (28000,), 4400), ("backing", (10000,), 4400)):
                with wave.open(str(root / f"{name}.wav"), "wb") as handle:
                    handle.setparams((len(samples), 2, 44100, 0, "NONE", ""))
                    handle.writeframes(struct.pack(f"<{len(samples)}h", *samples) * frames)
            mix = mix_wavs(root / "instrument.wav", root / "lead.wav", root / "backing.wav",
                           root / "mix.wav", gains=(4.0, 4.0))
            with wave.open(str(mix), "rb") as handle:
                self.assertEqual((handle.getnchannels(), handle.getnframes()), (2, 4410))
                gain = 0.98 * 32767 / 176000
                self.assertEqual(struct.unpack("<2h", handle.readframes(1)), (round(140000 * gain), round(176000 * gain)))
                handle.setpos(4400)
                self.assertEqual(struct.unpack("<2h", handle.readframes(1)), (round(-12000 * gain), round(24000 * gain)))

    def test_intersample_peak_is_attenuated_without_changing_timeline(self):
        ffmpeg = os.environ.get("RIFT_TEST_FFMPEG") or shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("FFmpeg is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "mix.wav"
            output = Path(temporary) / "master.wav"
            peak_sample = round(0.97 * 32767)
            with wave.open(str(source), "wb") as handle:
                handle.setparams((1, 2, 48000, 0, "NONE", ""))
                handle.writeframes(struct.pack("<4h", peak_sample, peak_sample, -peak_sample, -peak_sample) * 12000)
            asyncio.run(limit_master_true_peak(source, output, ffmpeg=ffmpeg, target_dbtp=-1.0))
            self.assertEqual(wav_info(output), wav_info(source))
            result = subprocess.run([ffmpeg, "-nostats", "-i", str(output), "-af",
                                     "ebur128=peak=true:framelog=verbose", "-f", "null", "-"],
                                    capture_output=True, check=True, text=True, errors="replace")
            peak = float(result.stderr.rsplit("True peak:", 1)[1].split()[1])
            self.assertTrue(math.isfinite(peak))
            self.assertLess(peak, -1.0)
