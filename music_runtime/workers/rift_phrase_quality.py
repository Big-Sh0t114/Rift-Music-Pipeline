"""Cluster-level conditioning and inference for phrase-based vocal conversion."""

from dataclasses import dataclass
import math

import numpy as np
import pyloudnorm

if __package__:
    from . import rift_quality as quality, rift_timbre as timbre
    from .rift_sections import (ANALYSIS_CONTEXT, EDGE_FADE, RATE, VAD_RATE, Section, analysis_audio,
                                cluster_phrases, merge_phrases, phrase_boundaries, style_features)
else:
    import rift_quality as quality
    import rift_timbre as timbre
    from rift_sections import (ANALYSIS_CONTEXT, EDGE_FADE, RATE, VAD_RATE, Section, analysis_audio,
                               cluster_phrases, merge_phrases, phrase_boundaries, style_features)


@dataclass
class Cluster:
    shift: int
    outside: float | None
    route: str
    guidance: quality.Guidance
    gains: np.ndarray | None
    drift: float | None
    alpha: float
    gain: float
    cvec: tuple | None = None
    domain_plan: object | None = None
    formant_ratio: float = 1.0
    source_anchor: float | None = None
    source_anchor_frames: int = 0
    source_anchor_log_mad: float | None = None
    domain_reason: str | None = None


@dataclass
class PhraseDomain:
    virtual_lift: int = 0
    reason: str | None = None
    octave_shift: int | None = None


@dataclass
class SectionDomain:
    virtual_lift_candidate: int = 0
    octave_shift_candidate: int = 0
    formant_ratio_candidate: float = 1.0
    source_anchor: float | None = None
    source_anchor_frames: int = 0
    source_anchor_log_mad: float | None = None
    reason: str | None = None
    granularity: str = "legacy"


def _phrase_domain_plan(domain, samples, track, cluster_shift, profile, settings):
    if not settings.get("domain_alignment", False):
        return PhraseDomain(0, "domain_alignment_disabled")
    f0, confidence = track
    f0 = np.asarray(f0, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    size = min(len(f0), len(confidence))
    track = f0[:size], confidence[:size]
    manual_shift = int(settings.get("key_shift", 0))
    occupancy = float(np.count_nonzero((track[0] > 0) & (track[1] >= 0.3)) / max(1, size))
    pitch_range = settings["pitch_range"] or (profile["pitch_range"] if profile else [])
    pitch_tails = profile.get("pitch_tails") if profile and not settings["pitch_range"] else None
    try:
        inherited = 0
        if not manual_shift and cluster_shift and domain.supports_cluster_octave(track, pitch_range, cluster_shift, occupancy):
            inherited = cluster_shift
        planning_track = (track[0] * (2.0 ** (inherited / 12.0)), track[1])
        allow_phrase_octave = bool(settings.get("allow_octave_jump", False)) and not (inherited or manual_shift)
        plan = domain.plan_domain(
            [planning_track], pitch_range, manual_shift,
            virtual_available=bool(settings.get("domain_ffmpeg")), allow_octave=False,
            voiced_share=occupancy, source_anchor=None, target_anchor=None, protect_formants=False,
            pitch_tails=pitch_tails, allow_phrase_octave=allow_phrase_octave,
        )
        final_shift = plan.octave_shift or inherited
        reason = "cluster_octave_rejected" if not final_shift and cluster_shift else plan.reason
        return PhraseDomain(plan.virtual_lift, reason, final_shift)
    except Exception as exc:
        return PhraseDomain(0, f"phrase_plan_failed:{type(exc).__name__}", manual_shift)


def _formant_ratio_for_section(domain, phrase_indices, phrases, tracks, mono, profile, cluster, settings):
    candidate = SectionDomain(
        formant_ratio_candidate=cluster.formant_ratio,
        reason=cluster.domain_reason,
        granularity="cluster_fallback",
    )
    if not settings.get("domain_formants", False) or domain is None:
        return candidate
    samples = [mono[phrases[index].start:phrases[index].end] for index in phrase_indices]
    member_tracks = [tracks[index] for index in phrase_indices]
    source_anchor, count, log_mad = _source_formant_anchor(domain, samples, member_tracks)
    candidate.source_anchor = source_anchor
    candidate.source_anchor_frames = count
    candidate.source_anchor_log_mad = log_mad
    target_anchor = _profile_formant_anchor(profile)
    if source_anchor is None or target_anchor is None:
        return candidate
    level = 20.0 * np.log10(max(quality.rms(np.concatenate(samples)), quality.FLOOR))
    try:
        local = domain.plan_domain(
            member_tracks, settings["pitch_range"] or (profile["pitch_range"] if profile else []),
            int(settings.get("key_shift", 0)), virtual_available=False, allow_octave=False,
            voiced_share=float(_voiced_count(member_tracks) / max(1, sum(len(f0) for f0, _ in member_tracks))),
            source_anchor=source_anchor, target_anchor=target_anchor,
            protect_formants=_protect_formants(samples, member_tracks, float(level)),
        )
        ratio = float(local.formant_ratio)
        if math.isfinite(ratio) and 0.5 <= ratio <= 2.0:
            candidate.formant_ratio_candidate = round(ratio, 4)
            candidate.reason = getattr(local, "reason", candidate.reason)
            candidate.granularity = "section"
    except Exception as exc:
        candidate.reason = f"section_formant_plan_failed:{type(exc).__name__}"
    return candidate


def _split_planned_sections(sections, phrases, phrase_plans, tracks, mono, clusters, profile, settings):
    domain = _domain_module() if settings.get("domain_formants", False) else None
    planned, plans = [], []
    for section in sections:
        groups = []
        for phrase_index in section.phrases:
            phrase_plan = phrase_plans[phrase_index]
            octave = clusters[section.cluster].shift if phrase_plan.octave_shift is None else phrase_plan.octave_shift
            pitch_plan = (octave, phrase_plan.virtual_lift)
            if groups and groups[-1][0] == pitch_plan:
                groups[-1][1].append(phrase_index)
            else:
                groups.append((pitch_plan, [phrase_index]))
        for (octave, lift), indices in groups:
            first, last = phrases[indices[0]], phrases[indices[-1]]
            planned.append(Section(first.start, last.end, section.cluster, tuple(indices)))
            section_domain = _formant_ratio_for_section(domain, indices, phrases, tracks, mono,
                                                        profile, clusters[section.cluster], settings)
            section_domain.virtual_lift_candidate = lift
            section_domain.octave_shift_candidate = octave
            if octave != clusters[section.cluster].shift:
                section_domain.reason = phrase_plans[indices[0]].reason
            else:
                section_domain.reason = section_domain.reason or phrase_plans[indices[0]].reason
            section_domain.granularity = "phrase_pitch+" + section_domain.granularity
            plans.append(section_domain)
    return planned, plans


def _retune_candidate(samples, lift, ffmpeg, pitch):
    if __package__:
        from .rift_retune import retune_pitch
    else:
        from rift_retune import retune_pitch
    result = retune_pitch(samples, -float(lift), ffmpeg, sample_rate=RATE)
    f0_delta = None
    try:
        before, before_conf = pitch(samples)
        after, after_conf = pitch(result.samples)
        size = min(len(before), len(before_conf), len(after), len(after_conf))
        valid = (np.asarray(before[:size]) > 0) & (np.asarray(after[:size]) > 0)
        valid &= (np.asarray(before_conf[:size]) >= 0.3) & (np.asarray(after_conf[:size]) >= 0.3)
        if np.count_nonzero(valid) >= 10:
            cents = 12.0 * np.log2(np.asarray(after[:size], dtype=np.float64)[valid]
                                   / np.asarray(before[:size], dtype=np.float64)[valid])
            f0_delta = float(np.median(cents))
            if abs(f0_delta + float(lift)) > 2.0:
                raise RuntimeError("Rubber Band pitch check disagreed with the requested virtual lift.")
    except RuntimeError:
        raise
    except Exception:
        # F0 validation is opportunistic; a failing pitch estimator must not
        # block an otherwise valid length-preserved Rubber Band return.
        f0_delta = None
    return result.samples, {"length_drift_samples": result.drift_samples,
                            "f0_shift_semitones": f0_delta}


def _upper_register_metrics(source_f0, source_confidence, output_f0, output_confidence, expected_shift=12):
    """Measure reliable source voicing and expected final-pitch coverage.

    Pitch estimates can move by a few frames after inference and retuning, so
    coverage accepts a 30 ms neighborhood. The local guard only considers
    windows with substantial, confident source voicing; silence and breath
    cannot by themselves trigger the optional rescue.
    """
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1)
              for value in (source_f0, source_confidence, output_f0, output_confidence)]
    size = min(map(len, arrays))
    if size <= 0:
        return None
    source_pitch, source_conf, output_pitch, output_conf = [value[:size] for value in arrays]
    source_voiced = (np.isfinite(source_pitch) & (source_pitch >= 40.0) & (source_pitch <= 2000.0)
                     & np.isfinite(source_conf) & (source_conf >= 0.3))
    output_voiced = (np.isfinite(output_pitch) & (output_pitch >= 40.0) & (output_pitch <= 3000.0)
                     & np.isfinite(output_conf) & (output_conf >= 0.3))
    source_count = int(np.count_nonzero(source_voiced))
    if source_count == 0:
        return None

    voiced_nearby = np.zeros(size, dtype=bool)
    expected_nearby = np.zeros(size, dtype=bool)
    for offset in range(-3, 4):
        src_start, src_end = max(0, -offset), min(size, size - offset)
        if src_end <= src_start:
            continue
        out_start, out_end = src_start + offset, src_end + offset
        paired_source = source_voiced[src_start:src_end]
        paired_output = output_voiced[out_start:out_end]
        voiced_nearby[src_start:src_end] |= paired_output
        valid = paired_source & paired_output
        if np.any(valid):
            semitones = 12.0 * np.log2(output_pitch[out_start:out_end][valid]
                                       / source_pitch[src_start:src_end][valid])
            expected = np.abs(semitones - float(expected_shift)) <= 2.5
            expected_slice = expected_nearby[src_start:src_end]
            expected_slice[valid] |= expected

    coverage = float(np.mean(voiced_nearby[source_voiced]))
    expected_coverage = float(np.mean(expected_nearby[source_voiced]))
    exact = source_voiced & output_voiced
    exact_shifts = 12.0 * np.log2(output_pitch[exact] / source_pitch[exact]) if np.any(exact) else np.empty(0)
    measured_shift = float(np.median(exact_shifts)) if len(exact_shifts) else None

    window_size, hop = 150, 75  # 1.5 second windows, stepped by 750 ms.
    window_losses = []
    for start in range(0, size, hop):
        end = min(size, start + window_size)
        if end <= start:
            continue
        in_window = source_voiced[start:end]
        count = int(np.count_nonzero(in_window))
        minimum = max(50, int(np.ceil((end - start) * 0.30)))
        if count < minimum:
            continue
        window_losses.append((1.0 - float(np.mean(voiced_nearby[start:end][in_window])), start, end))
        if end == size:
            break
    worst_local = max(window_losses, default=None, key=lambda value: value[0])
    worst_local_loss = worst_local[0] if worst_local is not None else None
    return {
        "source_voiced_frames": source_count,
        "source_voiced_ratio": source_count / size,
        "voiced_coverage": coverage,
        "voiced_frame_loss": 1.0 - coverage,
        "expected_octave_coverage": expected_coverage,
        "post_shift_semitones": measured_shift,
        "worst_local_voiced_loss": worst_local_loss,
        "worst_local_window_seconds": ([round(worst_local[1] / 100.0, 3), round(worst_local[2] / 100.0, 3)]
                                        if worst_local is not None else None),
        "local_windows_checked": len(window_losses),
    }


def _upper_register_gate(metrics):
    if metrics is None or metrics["source_voiced_frames"] < 100 or metrics["source_voiced_ratio"] < 0.2:
        return [], "source_pitch_not_reliable"
    reasons = []
    if metrics["voiced_frame_loss"] >= quality.LOSS_LIMIT:
        reasons.append("global_voiced_frame_loss")
    if metrics["expected_octave_coverage"] < 0.75:
        reasons.append("expected_octave_coverage_low")
    if metrics["worst_local_voiced_loss"] is not None and metrics["worst_local_voiced_loss"] >= 0.4:
        reasons.append("localized_voiced_dropout")
    return reasons, "quality_gate_not_triggered" if not reasons else None


def _rescue_metrics_json(metrics):
    if metrics is None:
        return None
    return {key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in metrics.items()}


def _rescue_candidate_quality(metrics, primary):
    if metrics is None:
        return False, "candidate_pitch_not_reliable", False
    shift = metrics["post_shift_semitones"]
    if shift is None or abs(shift - 12.0) > 1.0:
        return False, "final_pitch_not_near_requested_octave", False
    if metrics["expected_octave_coverage"] < 0.35:
        return False, "expected_octave_coverage_unusable", False
    if metrics["voiced_coverage"] + 0.02 < primary["voiced_coverage"]:
        return False, "voicing_worse_than_primary", False
    if metrics["expected_octave_coverage"] + 0.02 < primary["expected_octave_coverage"]:
        return False, "octave_coverage_worse_than_primary", False
    improvements = (
        primary["voiced_frame_loss"] - metrics["voiced_frame_loss"] >= 0.05,
        metrics["expected_octave_coverage"] - primary["expected_octave_coverage"] >= 0.08,
        primary["worst_local_voiced_loss"] is not None
        and metrics["worst_local_voiced_loss"] is not None
        and primary["worst_local_voiced_loss"] - metrics["worst_local_voiced_loss"] >= 0.15,
    )
    if not any(improvements):
        return False, "candidate_did_not_materially_improve_voicing", False
    complete = metrics["voiced_frame_loss"] <= 0.08 and metrics["expected_octave_coverage"] >= 0.65
    return True, None, complete


def _rescue_score(metrics):
    local_coverage = (1.0 - metrics["worst_local_voiced_loss"]
                      if metrics["worst_local_voiced_loss"] is not None
                      else metrics["voiced_coverage"])
    return (0.45 * metrics["expected_octave_coverage"]
            + 0.35 * metrics["voiced_coverage"] + 0.20 * local_coverage)


def _domain_module():
    try:
        if __package__:
            from . import rift_domain
        else:
            import rift_domain
        return rift_domain
    except Exception:
        return None


def _song_octave_choice(tracks, profile, settings, domain):
    """Resolve manual, locked, and opted-in automatic whole-song pitch shifts."""
    requested = settings.get("song_octave_shift", 0)
    if requested is None:
        requested = 0
    if type(requested) is not int or requested not in (-12, 0, 12):
        raise ValueError("song_octave_shift must be -12, 0, 12, or None")

    manual_shift = int(settings.get("key_shift", 0))
    if manual_shift:
        return 0, None, "manual_key_shift_priority", None
    if requested:
        return requested, "locked", "song_octave_locked", None
    if not settings.get("domain_alignment", False):
        return 0, None, "domain_alignment_disabled", None
    if not settings.get("allow_octave_jump", False):
        return 0, None, "song_octave_jump_disabled", None
    if settings.get("allow_song_octave_detection") is not True:
        return 0, None, "song_octave_detection_disabled", None
    if domain is None:
        return 0, None, "song_octave_planner_unavailable", None

    pitch_range = settings.get("pitch_range") or (profile["pitch_range"] if profile else [])
    pitch_tails = profile.get("pitch_tails") if profile and not settings.get("pitch_range") else None
    plan = domain.plan_song_octave(tracks, pitch_range, pitch_tails)
    shift = int(plan.octave_shift)
    return shift, "automatic" if shift else None, plan.reason, plan


def _profile_formant_anchor(profile):
    if not profile:
        return None
    try:
        anchor = float(profile.get("formant_anchor_hz"))
        count = int(profile.get("formant_frames", 0))
        log_mad = float(profile.get("formant_log_mad"))
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(anchor) and 250.0 <= anchor <= 3500.0 and count >= 30
            and math.isfinite(log_mad) and 0.0 <= log_mad <= 0.25):
        return None
    return anchor


def _source_formant_anchor(domain, samples, tracks):
    frames = []
    for source, track in zip(samples, tracks, strict=True):
        try:
            values = np.asarray(domain.formant_frames(source, track), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        values = values[np.isfinite(values) & (values >= 250.0) & (values <= 3500.0)]
        if len(values):
            frames.append(values)
    if not frames:
        return None, 0, None
    try:
        anchor, count, log_mad = domain.summarize_formants(np.concatenate(frames))
        anchor, count = float(anchor), int(count)
        log_mad = float(log_mad) if log_mad is not None else None
    except Exception:
        return None, sum(len(value) for value in frames), None
    if not (count >= 30 and math.isfinite(anchor) and 250.0 <= anchor <= 3500.0
            and log_mad is not None and math.isfinite(log_mad) and 0.0 <= log_mad <= 0.25):
        return None, count, log_mad
    return anchor, count, log_mad


def _voiced_count(tracks):
    voiced = 0
    for f0, confidence in tracks:
        f0 = np.asarray(f0)
        confidence = np.asarray(confidence)
        size = min(len(f0), len(confidence))
        voiced += int(np.count_nonzero((f0[:size] > 0) & (confidence[:size] >= 0.3)))
    return voiced


def _low_hnr_evidence(samples, tracks):
    try:
        import parselmouth
    except Exception:
        return False
    values = []
    for source, (f0, confidence) in zip(samples, tracks, strict=True):
        try:
            sound = parselmouth.Sound(source, sampling_frequency=RATE)
            harmonicity = sound.to_harmonicity_cc(
                time_step=0.01, minimum_pitch=75, silence_threshold=0.1, periods_per_window=1.0,
            )
            hnr = np.asarray(harmonicity.values[0], dtype=np.float64)
            size = min(len(hnr), len(f0), len(confidence))
            voiced = (np.asarray(f0[:size]) > 0) & (np.asarray(confidence[:size]) >= 0.3)
            valid = hnr[:size][voiced & np.isfinite(hnr[:size])]
            if len(valid):
                values.append(valid)
        except Exception:
            continue
    return bool(values and float(np.median(np.concatenate(values))) < 12.0)


def _protect_formants(samples, tracks, cluster_level_db):
    voiced = [np.asarray(f0)[(np.asarray(f0) > 0) & (np.asarray(confidence) >= 0.3)]
              for f0, confidence in tracks]
    voiced = [value for value in voiced if len(value)]
    pitch_values = np.concatenate(voiced) if voiced else np.empty(0)
    quiet_high_f0 = bool(len(pitch_values) and cluster_level_db < -24.0
                         and float(np.median(pitch_values)) >= 250.0)
    return _low_hnr_evidence(samples, tracks) or quiet_high_f0


def cluster_parameters(mono, phrases, tracks, profile, meter, settings, total_voiced_frames=None):
    pitch_range = settings["pitch_range"] or (profile["pitch_range"] if profile else [])
    shift, outside = quality.choose_shift(tracks, pitch_range, settings["key_shift"])
    domain_plan = None
    formant_ratio = 1.0
    source_anchor = None
    source_anchor_frames = 0
    source_anchor_log_mad = None
    domain_reason = None
    alignment_enabled = bool(settings.get("domain_alignment", False))
    formants_enabled = bool(settings.get("domain_formants", False))
    # Insert gaps so changing phrases never counts as a jitter event.
    f0, confidence = [np.concatenate([np.r_[value, 0] for value in columns]) for columns in zip(*tracks)]
    samples = [mono[phrase.start:phrase.end] for phrase in phrases]
    peaksq = sum(float(np.dot(audio, audio)) for audio in samples) / sum(len(audio) for audio in samples)
    kind, guidance = quality.route(f0, confidence, 10 * np.log10(max(peaksq, 1e-16)), settings)
    gains = drift = None
    alpha = 0.0
    if profile:
        weighted_bands = []
        for source, track in zip(samples, tracks, strict=True):
            frame = timbre.extract_spectrum(source, track)
            if frame:
                weighted_bands.append(frame.band_power * frame.active_frames)
        if weighted_bands:
            bands = np.sum(weighted_bands, axis=0)
            spectrum = 10 * np.log10(np.maximum(bands, max(1e-14, float(bands.max()) * 1e-6)))
            gains, drift = timbre.spectral_gains(spectrum - spectrum.mean(), profile["ltas_db"])
            alpha = float(np.interp(drift, [5, 17], [0.30, 0.50]))
            if drift >= timbre.DRIFT_THRESHOLD_DB:
                severity = min((drift - timbre.DRIFT_THRESHOLD_DB) / 10, 1.0)
                guidance = quality.Guidance(2, min(0.80, max(0.70 + 0.10 * severity, settings["spk_cfg_strength"])),
                                            min(guidance.detail, 0.1 * (1 - severity)), 0.75)
                kind = "high_drift"

    power, weight = 0.0, 0
    for source, track in zip(samples, tracks, strict=True):
        loudness = timbre.voiced_loudness(timbre.equalize(source, gains), track, meter)
        if loudness is not None:
            count = int(np.count_nonzero(track[0] > 0))
            power += 10 ** (loudness / 10) * count
            weight += count
    if not weight:
        # Breath-only clusters use their own aggregate RMS, keeping one gain
        # across their phrases even when RMVPE found no usable pitch frames.
        power, weight = max(peaksq, 1e-10), 1
    cluster_level_db = float(10 * np.log10(power / weight))
    if alignment_enabled or formants_enabled:
        domain = _domain_module()
        if domain is None:
            domain_reason = "domain_planner_unavailable"
        else:
            if formants_enabled:
                source_anchor, source_anchor_frames, source_anchor_log_mad = _source_formant_anchor(domain, samples, tracks)
            target_anchor = _profile_formant_anchor(profile) if formants_enabled else None
            try:
                domain_plan = domain.plan_domain(
                    tracks, pitch_range, settings["key_shift"],
                    # Structural shift remains cluster-wide; reversible support
                    # lift is planned later per natural phrase.
                    virtual_available=alignment_enabled and bool(settings.get("domain_ffmpeg")),
                    allow_octave=bool(settings.get("allow_octave_jump", False)),
                    voiced_share=float(_voiced_count(tracks) / max(1, total_voiced_frames or _voiced_count(tracks))),
                    source_anchor=source_anchor, target_anchor=target_anchor,
                    protect_formants=formants_enabled and _protect_formants(samples, tracks, cluster_level_db),
                )
                domain_reason = getattr(domain_plan, "reason", None)
                if alignment_enabled:
                    shift = int(domain_plan.octave_shift)
                    plan_outside = getattr(domain_plan, "predicted_outside_ratio", None)
                    if plan_outside is not None and math.isfinite(float(plan_outside)):
                        outside = float(plan_outside)
                if formants_enabled:
                    candidate_ratio = float(getattr(domain_plan, "formant_ratio", 1.0))
                    if math.isfinite(candidate_ratio) and 0.5 <= candidate_ratio <= 2.0:
                        formant_ratio = candidate_ratio
            except Exception as exc:
                # An opt-in planner failure must leave the established phrase
                # route usable and report that the candidate was not applied.
                domain_plan = None
                domain_reason = f"domain_planner_failed:{type(exc).__name__}"
    gain = float(10 ** (np.clip(-18 - 10 * np.log10(power / weight), -18, 18) / 20))
    return Cluster(shift, outside, kind, guidance, gains, drift, alpha, gain,
                   domain_plan=domain_plan, formant_ratio=formant_ratio,
                   source_anchor=source_anchor, source_anchor_frames=source_anchor_frames,
                   source_anchor_log_mad=source_anchor_log_mad, domain_reason=domain_reason)


def content_moments(mono, phrases, tracks, cluster, extract_content):
    count, sums, squares = 0, None, None
    def collect(formant_ratio):
        nonlocal count, sums, squares
        count, sums, squares = 0, None, None
        for phrase, track in zip(phrases, tracks, strict=True):
            samples = timbre.equalize(mono[phrase.start:phrase.end], cluster.gains) * cluster.gain
            if len(samples) < quality.MIN_MODEL_SAMPLES:
                continue
            if formant_ratio == 1.0:
                features, times = extract_content(samples)
            else:
                features, times = extract_content(samples, formant_ratio=formant_ratio)
            voiced = features[timbre.voiced_at(times, track)].astype(np.float64)
            count += len(voiced)
            if len(voiced):
                frame_sums = voiced.sum(axis=0)
                frame_squares = (voiced ** 2).sum(axis=0)
                sums = frame_sums if sums is None else sums + frame_sums
                squares = frame_squares if squares is None else squares + frame_squares
    try:
        collect(cluster.formant_ratio)
    except Exception:
        if cluster.formant_ratio == 1.0:
            raise
        cluster.domain_reason = f"formant_content_fallback:{cluster.domain_reason or 'transform_failed'}"
        cluster.formant_ratio = 1.0
        collect(1.0)
    if count < 3:
        return None
    mean = sums / count
    return mean.astype(np.float32), np.sqrt(np.maximum(1e-8, squares / count - mean ** 2)).astype(np.float32)


def _select_octave_probe_excerpts(mono, phrases, tracks):
    """Pick deterministic lower, middle, and upper reliably voiced phrases."""
    candidates = []
    for index, (phrase, track) in enumerate(zip(phrases, tracks, strict=True)):
        f0, confidence = [np.asarray(value, dtype=np.float64).reshape(-1) for value in track]
        size = min(len(f0), len(confidence))
        if not size:
            continue
        f0, confidence = f0[:size], confidence[:size]
        voiced = (np.isfinite(f0) & (f0 >= 40) & (f0 <= 2000)
                  & np.isfinite(confidence) & (confidence >= 0.3))
        count = int(np.count_nonzero(voiced))
        duration = (phrase.end - phrase.start) / RATE
        if count < 180 or duration < 2.5 or count / size < 0.2:
            continue
        voiced_f0 = f0[voiced]
        candidates.append({
            "phrase_index": index,
            "median_f0_hz": float(np.median(voiced_f0)),
            "upper_f0_hz": float(np.quantile(voiced_f0, 0.85)),
            "voiced_frames": count,
            "voiced_ratio": count / size,
            "phrase_start": phrase.start,
            "phrase_end": phrase.end,
            "phrase_center": int((phrase.start + phrase.end) / 2),
            "voiced_center": phrase.start + int(np.median(np.flatnonzero(voiced))) * (RATE // 100),
        })
    if not candidates:
        return []

    median_target = float(np.median([item["median_f0_hz"] for item in candidates]))
    targets = (
        ("lower", min(item["median_f0_hz"] for item in candidates), "median_f0_hz"),
        ("middle", median_target, "median_f0_hz"),
        ("upper", max(item["upper_f0_hz"] for item in candidates), "upper_f0_hz"),
    )
    selected = []
    for band, target, key in targets:
        used = {item["phrase_index"] for item in selected}
        choices = sorted(candidates, key=lambda item: (
            abs(12.0 * math.log2(max(1.0, item[key]) / max(1.0, target))),
            -item["voiced_frames"], item["phrase_start"],
        ))
        unselected = [item for item in choices if item["phrase_index"] not in used]
        if not unselected:
            continue
        spaced = [item for item in unselected if all(
            abs(item["phrase_center"] - value["phrase_center"]) >= 3 * RATE for value in selected
        )]
        item = (spaced or unselected)[0]
        phrase = phrases[item["phrase_index"]]
        duration_samples = min(4 * RATE, phrase.end - phrase.start)
        center = item["voiced_center"]
        clip_start = max(phrase.start, min(phrase.end - duration_samples, center - duration_samples // 2))
        clip_end = clip_start + duration_samples
        context = int(0.25 * RATE)
        probe_start, probe_end = max(0, clip_start - context), min(len(mono), clip_end + context)
        selected.append({
            **item, "band": band, "start": int(clip_start), "end": int(clip_end),
            "probe_start": int(probe_start), "probe_end": int(probe_end),
            "active_start_seconds": (clip_start - probe_start) / RATE,
            "active_end_seconds": (clip_end - probe_start) / RATE,
            "start_seconds": clip_start / RATE, "end_seconds": clip_end / RATE,
            "audio": np.asarray(mono[probe_start:probe_end], dtype=np.float32).copy(),
        })
    return selected[:3]


def _octave_probe_candidate_order(classification, excerpts, settings):
    source_voice = classification.get("source_voice", "unknown")
    target_voice = settings.get("octave_probe_target_voice", "unknown")
    if source_voice in {"female", "male"} and target_voice in {"female", "male"} and source_voice != target_voice:
        preferred = 12 if source_voice == "male" and target_voice == "female" else -12
        return [preferred, 0, -preferred], "confident_opposite_acoustic_voice_labels"
    profile = settings.get("voice_profile")
    target_range = settings.get("pitch_range") or (profile.get("pitch_range") if profile else [])
    if isinstance(target_range, (list, tuple)) and len(target_range) == 3:
        values = [value for item in excerpts for value in (item["median_f0_hz"], item["upper_f0_hz"])]
        if values and all(float(value) > 0 for value in target_range):
            notes = 69.0 + 12.0 * np.log2(np.asarray(values) / 440.0)
            source_quantiles = np.quantile(notes, [0.1, 0.5, 0.9])
            target_notes = np.asarray(target_range, dtype=np.float64)
            stable = {0: 0, 12: 1, -12: 2}
            order = sorted((-12, 0, 12), key=lambda shift: (
                float(np.mean(np.abs(source_quantiles + shift - target_notes))), stable[shift],
            ))
            return order, "source_f0_quantiles_against_target_pitch_range"
    return [0, 12, -12], "no_target_pitch_distribution_neutral_order"


def _octave_probe_metrics_summary(metrics_list):
    usable = [item for item in metrics_list if item is not None]
    total = sum(item["source_voiced_frames"] for item in usable)
    if not usable or total <= 0:
        return None
    coverage = sum(item["voiced_coverage"] * item["source_voiced_frames"] for item in usable) / total
    expected = sum(item["expected_octave_coverage"] * item["source_voiced_frames"] for item in usable) / total
    local = [item["worst_local_voiced_loss"] for item in usable if item["worst_local_voiced_loss"] is not None]
    shifts = [item["post_shift_semitones"] for item in usable if item["post_shift_semitones"] is not None]
    return {
        "sample_count": len(usable), "source_voiced_frames": total,
        "voiced_coverage": round(float(coverage), 4),
        "voiced_frame_loss": round(float(1.0 - coverage), 4),
        "expected_shift_coverage": round(float(expected), 4),
        "median_shift_semitones": round(float(np.median(shifts)), 3) if shifts else None,
        "worst_local_voiced_loss": round(float(max(local)), 4) if local else None,
    }


_OCTAVE_PROBE_MIN_RELIABLE_SOURCE_FRAMES = 100
_OCTAVE_PROBE_MIN_RELIABLE_SOURCE_RATIO = 0.2
_OCTAVE_PROBE_MIN_RELIABLE_LOCAL_WINDOWS = 1


def _octave_probe_metrics_reliability_reasons(metrics):
    if metrics is None:
        return ["metrics_unavailable"]
    reasons = []
    if metrics.get("source_voiced_frames", 0) < _OCTAVE_PROBE_MIN_RELIABLE_SOURCE_FRAMES:
        reasons.append("source_voiced_frames_below_minimum")
    if metrics.get("source_voiced_ratio", 0.0) < _OCTAVE_PROBE_MIN_RELIABLE_SOURCE_RATIO:
        reasons.append("source_voiced_ratio_below_minimum")
    if (metrics.get("local_windows_checked", 0) < _OCTAVE_PROBE_MIN_RELIABLE_LOCAL_WINDOWS
            or metrics.get("worst_local_voiced_loss") is None):
        reasons.append("local_voiced_loss_evidence_unavailable")
    return reasons


def _octave_probe_candidate_assessment(pairs, requested_shift, *, render_failures=0, sample_failures=0):
    local_regressions, shift_mismatches = [], []
    unreliable_pairs, qualified_pairs = [], []
    for metrics, baseline, phrase_index in pairs:
        candidate_reasons = _octave_probe_metrics_reliability_reasons(metrics)
        baseline_reasons = _octave_probe_metrics_reliability_reasons(baseline)
        if candidate_reasons or baseline_reasons:
            unreliable_pairs.append({
                "phrase_index": phrase_index,
                "candidate_reasons": candidate_reasons,
                "baseline_reasons": baseline_reasons,
            })
            continue
        qualified_pairs.append((metrics, baseline, phrase_index))
        measured_shift = metrics["post_shift_semitones"]
        if measured_shift is None or abs(measured_shift - requested_shift) > 1.0:
            shift_mismatches.append({"phrase_index": phrase_index, "measured_shift_semitones": measured_shift})
        if metrics["voiced_coverage"] < baseline["voiced_coverage"] - 0.08:
            local_regressions.append({"phrase_index": phrase_index, "reason": "voiced_coverage_regressed"})
        worst, base_worst = metrics["worst_local_voiced_loss"], baseline["worst_local_voiced_loss"]
        if worst is not None and (worst >= 0.35 or (base_worst is not None and worst - base_worst >= 0.12)):
            local_regressions.append({"phrase_index": phrase_index, "reason": "localized_voiced_loss"})
    candidate_summary = _octave_probe_metrics_summary([item[0] for item in qualified_pairs])
    baseline_summary = _octave_probe_metrics_summary([item[1] for item in qualified_pairs])
    objective = baseline_objective = None
    material = False
    if candidate_summary is not None and baseline_summary is not None:
        objective = 0.55 * candidate_summary["voiced_coverage"] + 0.45 * candidate_summary["expected_shift_coverage"]
        baseline_objective = 0.55 * baseline_summary["voiced_coverage"] + 0.45 * baseline_summary["expected_shift_coverage"]
        material = objective >= baseline_objective + 0.06 and (
            candidate_summary["voiced_coverage"] >= baseline_summary["voiced_coverage"] + 0.05
            or candidate_summary["expected_shift_coverage"] >= baseline_summary["expected_shift_coverage"] + 0.08
        )
    if render_failures or sample_failures:
        reason = "candidate_or_baseline_metrics_unavailable"
    elif unreliable_pairs:
        reason = "unreliable_paired_excerpt_metrics"
    elif shift_mismatches:
        reason = "candidate_median_shift_outside_one_semitone"
    elif local_regressions:
        reason = "localized_or_global_regression"
    elif candidate_summary is None or candidate_summary["sample_count"] < 2 or candidate_summary["source_voiced_frames"] < 300:
        reason = "insufficient_paired_samples"
    elif not material:
        reason = "no_material_objective_improvement"
    else:
        reason = None
    return {
        "metrics": candidate_summary,
        "paired_actual_metrics": baseline_summary,
        "objective": round(objective, 4) if objective is not None else None,
        "material_improvement": bool(material),
        "local_regressions": local_regressions,
        "measured_shift_failures": shift_mismatches,
        "unreliable_paired_samples": unreliable_pairs,
        "candidate_render_failures": render_failures,
        "sample_comparison_failures": sample_failures,
        "rejection_reason": reason,
    }


def _octave_probe_render_candidate(mono, excerpt, shift, pitch, infer, settings, models, extract_content,
                                   preserve_rng):
    """Rerun the bounded excerpt through the ordinary phrase pipeline at one forced shift."""
    start = max(0, excerpt["start"] - ANALYSIS_CONTEXT)
    end = min(len(mono), excerpt["end"] + ANALYSIS_CONTEXT)
    candidate_settings = dict(settings)
    candidate_settings["song_octave_shift"] = int(shift)
    candidate_settings["allow_octave_jump"] = False
    candidate_settings["allow_song_octave_detection"] = False
    candidate_settings.pop("_octave_probe_classify", None)
    candidate_settings.pop("_octave_probe_preserve_rng", None)
    with preserve_rng():
        candidate_audio, _ = convert_phrases(
            np.asarray(mono[start:end], dtype=np.float32), pitch, infer,
            candidate_settings, models, extract_content,
        )
    begin, finish = excerpt["start"] - start, excerpt["end"] - start
    if begin < 0 or finish > len(candidate_audio) or finish <= begin:
        raise RuntimeError("probe_render_interval_mismatch")
    return np.asarray(candidate_audio[begin:finish], dtype=np.float32).copy()


def _song_octave_probe(mono, final_audio, phrases, tracks, settings, contexts, infer, pitch,
                       actual_song_shift, actual_song_source, models, extract_content):
    classify = settings.get("_octave_probe_classify")
    preserve_rng = settings.get("_octave_probe_preserve_rng")
    if not callable(classify) or not callable(preserve_rng):
        return None
    record = {
        "enabled": True,
        "schema_version": 1,
        "implementation": "rift_octave_probe_shadow_v1",
        "asset_state": settings.get("octave_probe_asset_state", "unavailable"),
        "asset_fingerprint": settings.get("octave_probe_asset_fingerprint"),
        "target_voice": settings.get("octave_probe_target_voice", "unknown"),
        "actual_v9": {"song_octave_shift": actual_song_shift, "source": actual_song_source},
        "recommendation": None,
        "candidate_comparisons": [],
    }
    if record["asset_state"] != "prepared":
        record.update({"status": "unavailable", "reason": f"assets_{record['asset_state']}"})
        return record
    try:
        with preserve_rng():
            excerpts = _select_octave_probe_excerpts(mono, phrases, tracks)
            record["selected_excerpts"] = [{
                "phrase_index": item["phrase_index"], "band": item["band"],
                "start_seconds": round(item["start_seconds"], 4),
                "end_seconds": round(item["end_seconds"], 4),
                "median_f0_hz": round(item["median_f0_hz"], 2),
                "upper_f0_hz": round(item["upper_f0_hz"], 2),
                "voiced_frames": item["voiced_frames"],
            } for item in excerpts]
            if not excerpts:
                record.update({"status": "insufficient", "reason": "no_reliable_spaced_phrases"})
                return record
            classification = classify(excerpts)
            record["classification"] = classification
            if classification.get("source_voice") not in {"female", "male", "unknown"}:
                raise ValueError("invalid_classifier_summary")
            order, order_reason = _octave_probe_candidate_order(classification, excerpts, settings)
            record["candidate_order"] = order
            record["candidate_order_reason"] = order_reason

            per_excerpt = []
            for excerpt in excerpts:
                context = contexts.get(excerpt["phrase_index"])
                if not context or not context["valid"]:
                    per_excerpt.append({"phrase_index": excerpt["phrase_index"], "baseline": None,
                                        "candidates": {}, "reason": "actual_output_interval_unavailable"})
                    continue
                source = np.asarray(mono[excerpt["start"]:excerpt["end"]], dtype=np.float32)
                baseline_output = np.asarray(final_audio[excerpt["start"]:excerpt["end"]], dtype=np.float32)
                try:
                    source_f0, source_confidence = pitch(source)
                    baseline_f0, baseline_confidence = pitch(baseline_output)
                    baseline = _upper_register_metrics(
                        source_f0, source_confidence, baseline_f0, baseline_confidence,
                        expected_shift=context["effective_shift"],
                    )
                except Exception as exc:
                    per_excerpt.append({"phrase_index": excerpt["phrase_index"], "baseline": None,
                                        "candidates": {}, "reason": f"baseline_pitch_failed:{type(exc).__name__}"})
                    continue
                evaluations = {}
                for shift in order:
                    if shift == context["effective_shift"]:
                        candidate_metrics, trial_error, trial_count = baseline, None, 0
                    else:
                        try:
                            rendered = _octave_probe_render_candidate(
                                mono, excerpt, shift, pitch, infer, settings, models, extract_content, preserve_rng,
                            )
                            candidate_f0, candidate_confidence = pitch(rendered)
                            candidate_metrics = _upper_register_metrics(
                                source_f0, source_confidence, candidate_f0, candidate_confidence,
                                expected_shift=shift,
                            )
                            trial_error, trial_count = None, 1
                        except Exception as exc:
                            candidate_metrics = None
                            trial_error, trial_count = f"candidate_render_failed:{type(exc).__name__}", 1
                    evaluations[shift] = {
                        "metrics": _rescue_metrics_json(candidate_metrics),
                        "error": trial_error,
                        "trial_count": trial_count,
                    }
                per_excerpt.append({
                    "phrase_index": excerpt["phrase_index"],
                    "effective_actual_shift": context["effective_shift"],
                    "baseline": _rescue_metrics_json(baseline),
                    "candidates": evaluations,
                })

            record["sample_comparisons"] = per_excerpt
            qualified = []
            for rank, shift in enumerate(order):
                pairs, render_failures, sample_failures, trials = [], 0, 0, 0
                for sample in per_excerpt:
                    candidate = sample.get("candidates", {}).get(shift)
                    baseline = sample.get("baseline")
                    if candidate is None or baseline is None:
                        sample_failures += 1
                        continue
                    trials += candidate["trial_count"]
                    metrics = candidate["metrics"]
                    if metrics is None:
                        render_failures += 1
                        continue
                    pairs.append((metrics, baseline, sample["phrase_index"]))
                assessment = _octave_probe_candidate_assessment(
                    pairs, shift, render_failures=render_failures, sample_failures=sample_failures,
                )
                entry = {
                    "shift": shift, "rank": rank + 1, "render_trials": trials,
                    **assessment,
                }
                record["candidate_comparisons"].append(entry)
                if entry["rejection_reason"] is None:
                    qualified.append(entry)
            record["trial_bound"] = 6
            record["trials_run"] = sum(item["render_trials"] for item in record["candidate_comparisons"])
            record["heuristics"] = {
                "objective": "0.55*voiced_coverage+0.45*expected_shift_coverage",
                "material_objective_gain": 0.06,
                "material_voiced_coverage_gain": 0.05,
                "material_expected_shift_coverage_gain": 0.08,
                "maximum_local_voiced_loss": 0.35,
                "maximum_local_regression": 0.12,
                "minimum_samples": 2,
                "minimum_source_voiced_frames": 300,
                "minimum_reliable_source_voiced_frames_per_excerpt": _OCTAVE_PROBE_MIN_RELIABLE_SOURCE_FRAMES,
                "minimum_reliable_source_voiced_ratio_per_excerpt": _OCTAVE_PROBE_MIN_RELIABLE_SOURCE_RATIO,
                "minimum_reliable_local_windows_per_excerpt": _OCTAVE_PROBE_MIN_RELIABLE_LOCAL_WINDOWS,
                "scope": "shadow_only_initial_heuristics_not_a_validated_hoarseness_score",
            }
            if qualified:
                qualified.sort(key=lambda item: (-item["objective"], item["rank"]))
                chosen = qualified[0]
                record["recommendation"] = {"shift": chosen["shift"], "reason": "material_objective_improvement"}
                record["status"], record["reason"] = "complete", None
            else:
                record["status"], record["reason"] = "complete", "no_candidate_met_shadow_quality_guards"
    except Exception as exc:
        record["status"] = "unavailable"
        record["reason"] = f"probe_failed:{type(exc).__name__}"
    return record


def convert_phrases(mono, pitch, infer, settings, models, extract_content):
    if models is None:
        raise RuntimeError("Prepare the Silero and WeSpeaker phrase models before conversion.")
    audio16 = analysis_audio(mono)
    phrases = phrase_boundaries(mono, models.probabilities(audio16))
    if not phrases:
        raise RuntimeError("The source contains no audible vocal phrases.")
    tracks, embeddings, styles = [], [], []
    for phrase in phrases:
        samples = mono[phrase.start:phrase.end]
        track = pitch(samples)
        tracks.append(track)
        styles.append(style_features(samples, track))
        short = audio16[round(phrase.start * VAD_RATE / RATE):round(phrase.end * VAD_RATE / RATE)]
        embeddings.append(models.embedding(np.pad(short, (0, max(0, 400 - len(short))))))
    profile = settings.get("voice_profile")
    domain = _domain_module() if settings.get("domain_alignment", False) else None
    song_octave_shift, song_octave_source, song_octave_reason, song_octave_plan = _song_octave_choice(
        tracks, profile, settings, domain,
    )
    # Treat a whole-song shift as a final musical choice. Passing it through
    # the established manual-shift lane suppresses cluster and phrase octave
    # overrides while leaving spectral, ContentVec, and guidance settings intact.
    pitch_settings = settings
    if song_octave_shift:
        pitch_settings = {**settings, "key_shift": song_octave_shift, "allow_octave_jump": False}
    labels = cluster_phrases(phrases, np.stack(embeddings), np.stack(styles))
    sections = merge_phrases(phrases, labels)
    meter = pyloudnorm.Meter(RATE, block_size=0.1)
    clusters = {}
    memberships = {}
    total_voiced_frames = _voiced_count(tracks)
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        group = [phrases[i] for i in indices], [tracks[i] for i in indices]
        memberships[label] = group
        clusters[label] = cluster_parameters(mono, *group, profile, meter, pitch_settings, total_voiced_frames)

    if song_octave_shift:
        phrase_reason = "song_octave_locked" if song_octave_source == "locked" else "song_octave_automatic"
        phrase_plans = [PhraseDomain(0, phrase_reason, song_octave_shift) for _ in phrases]
    else:
        phrase_plans = [
            _phrase_domain_plan(domain, mono[phrase.start:phrase.end], track,
                                clusters[int(label)].shift, profile, pitch_settings)
            for phrase, track, label in zip(phrases, tracks, labels, strict=True)
        ]
    sections, section_domains = _split_planned_sections(
        sections, phrases, phrase_plans, tracks, mono, clusters, profile, pitch_settings,
    )

    inputs, peaks = [], {label: 0.0 for label in clusters}
    for section in sections:
        start = max(0, section.start - ANALYSIS_CONTEXT)
        end = min(len(mono), section.end + ANALYSIS_CONTEXT)
        data = timbre.equalize(mono[start:end], clusters[section.cluster].gains)
        peaks[section.cluster] = max(peaks[section.cluster], float(np.max(np.abs(data))))
        inputs.append((start, end, data))
    for label, cluster in clusters.items():
        cluster.gain = min(cluster.gain, quality.PEAK_LIMIT / max(peaks[label], quality.FLOOR))

    cvec_by_section = {}
    group_keys = {}
    for index, section in enumerate(sections):
        plan = section_domains[index]
        key = (section.cluster, plan.formant_ratio_candidate)
        group_keys.setdefault(key, set()).update(section.phrases)
    for (label, ratio), indices in group_keys.items():
        cluster = clusters[label]
        cvec = None
        if extract_content and (cluster.alpha or ratio != 1.0):
            ordered = sorted(indices)
            pooled_cluster = Cluster(cluster.shift, cluster.outside, cluster.route, cluster.guidance,
                                     cluster.gains, cluster.drift, cluster.alpha, cluster.gain,
                                     formant_ratio=ratio, domain_reason=cluster.domain_reason)
            candidate = content_moments(mono, [phrases[i] for i in ordered], [tracks[i] for i in ordered],
                                        pooled_cluster, extract_content)
            # A failed transform reruns pooling with unwarped features; only
            # use those statistics on sections that also apply ratio 1.
            if pooled_cluster.formant_ratio == ratio:
                cvec = candidate
        cvec_by_section[(label, ratio)] = cvec

    combined = np.zeros(len(mono), dtype=np.float32)
    checks = []
    rescue_summaries = []
    probe_enabled = settings.get("octave_probe") is True and callable(settings.get("_octave_probe_classify"))
    probe_contexts = {} if probe_enabled else None
    for i, (section, (start, end, data)) in enumerate(zip(sections, inputs, strict=True)):
        cluster = clusters[section.cluster]
        data = np.pad(data * cluster.gain, (0, max(0, quality.MIN_MODEL_SAMPLES - len(data))))
        loudness = float(meter.integrated_loudness(data))
        trim = slice(section.start - start, section.end - start)
        source = mono[section.start:section.end]
        f0, source_confidence = pitch(source)

        domain_plan = section_domains[i]
        final_shift = domain_plan.octave_shift_candidate
        lift_candidate = domain_plan.virtual_lift_candidate
        formant_candidate = domain_plan.formant_ratio_candidate
        source_stats = cvec_by_section.get((section.cluster, formant_candidate))
        alpha = cluster.alpha if source_stats is not None else 0.0

        def generate(guidance, alpha, prepared, target_loudness, virtual_lift, formant_ratio,
                     cvec_source_stats, allow_formant_fallback=True):
            shift = final_shift + virtual_lift
            options = {"target_loudness": target_loudness, "cvec_alpha": alpha,
                       "cvec_source_stats": cvec_source_stats}
            if formant_ratio != 1.0:
                options["formant_ratio"] = formant_ratio
            try:
                (_, audio, _), = infer([(start, prepared)], guidance, shift, **options)
            except Exception as exc:
                if formant_ratio == 1.0 or not allow_formant_fallback:
                    raise
                # The profile statistics were pooled at the attempted ratio;
                # use neutral ContentVec alignment for a raw-feature rerender.
                core, measured, state = generate(guidance, 0.0, prepared, target_loudness,
                                                 virtual_lift, 1.0, None, False)
                state["formant_fallback_reason"] = f"transform_failed:{type(exc).__name__}"
                return core, measured, state
            predicted = np.asarray(audio, dtype=np.float32).reshape(-1)
            retune = {"length_drift_samples": 0, "f0_shift_semitones": None}
            if virtual_lift:
                predicted, retune = _retune_candidate(
                    predicted, virtual_lift, settings.get("domain_ffmpeg"), pitch,
                )
            core = np.asarray(predicted[:end - start][trim], dtype=np.float32).copy()
            return core, quality.measure(source, core, f0, pitch), {
                "rift_key_shift_applied": shift,
                "virtual_lift_applied": virtual_lift,
                "formant_ratio_applied": formant_ratio,
                "cvec_alpha_applied": alpha,
                "retune": retune,
            }

        guidance = cluster.guidance
        render_data, render_loudness = data, loudness
        if np.isfinite(loudness):
            try:
                output, (loss, excess, valid), render_state = generate(
                    guidance, alpha, data, loudness, lift_candidate, formant_candidate, source_stats,
                )
            except Exception as exc:
                if not lift_candidate:
                    raise
                output, (loss, excess, valid), render_state = generate(
                    guidance, alpha, data, loudness, 0, formant_candidate, source_stats,
                )
                render_state["virtual_fallback_reason"] = f"retune_or_inference_failed:{type(exc).__name__}"
        else:
            output, (loss, excess, valid) = np.zeros_like(source), (1.0, None, False)
            render_state = {"rift_key_shift_applied": final_shift, "virtual_lift_applied": 0,
                            "formant_ratio_applied": 1.0, "cvec_alpha_applied": 0.0,
                            "retune": {"length_drift_samples": 0, "f0_shift_semitones": None}}
        if render_state["virtual_lift_applied"] and (not valid or (loss is not None and loss > quality.LOSS_LIMIT)):
            primary = (output, (loss, excess, valid), render_state)
            try:
                fallback = generate(
                    guidance, render_state["cvec_alpha_applied"], data, loudness, 0,
                    render_state["formant_ratio_applied"],
                    source_stats if render_state["formant_ratio_applied"] == formant_candidate else None,
                )
            except Exception as exc:
                if not valid:
                    raise
                fallback = None
                render_state["virtual_fallback_rejected_reason"] = f"fallback_render_failed:{type(exc).__name__}"
            if fallback is not None:
                alternative, (fallback_loss, fallback_excess, fallback_valid), fallback_state = fallback
                improves_voicing = (loss is not None and fallback_loss is not None
                                    and fallback_loss < loss - 1e-6)
                accept_fallback = fallback_valid and (not valid or improves_voicing)
                if accept_fallback:
                    output, loss, excess, valid = alternative, fallback_loss, fallback_excess, fallback_valid
                    fallback_state["virtual_fallback_reason"] = "section_quality_validation"
                    render_state = fallback_state
                    alpha = render_state["cvec_alpha_applied"]
                elif valid:
                    output, (loss, excess, valid), render_state = primary
                    render_state["virtual_fallback_rejected_reason"] = (
                        "fallback_invalid" if not fallback_valid else "fallback_did_not_improve_voicing"
                    )
                else:
                    output, (loss, excess, valid), render_state = primary
                    render_state["virtual_fallback_rejected_reason"] = "fallback_invalid"
        actual_lift = render_state["virtual_lift_applied"]
        formant_ratio_applied = render_state["formant_ratio_applied"]
        alpha = render_state["cvec_alpha_applied"]
        retry = np.isfinite(loudness) and (not valid or (loss is not None and loss > quality.LOSS_LIMIT)
                                           or (excess is not None and excess > 6))
        cap_high_boost = False
        accepted = False
        retry_domain_fallback_reason = None
        if retry:
            conservative = quality.Guidance(2, max(0.0, guidance.speaker - 0.15), 0.0, 0.75)
            retry_alpha = min(alpha, 0.30)
            retry_data, retry_loudness = data, loudness
            if excess is not None and excess > 6 and cluster.gains is not None:
                upper = timbre.CENTERS >= 3200
                cap_high_boost = bool(np.any(cluster.gains[upper] > 0))
                if cap_high_boost:
                    gains = cluster.gains.copy()
                    gains[upper] = np.minimum(0, gains[upper])
                    retry_data = quality.limit_peak(timbre.equalize(mono[start:end], gains) * cluster.gain)
                    retry_data = np.pad(retry_data, (0, max(0, quality.MIN_MODEL_SAMPLES - len(retry_data))))
                    retry_loudness = float(meter.integrated_loudness(retry_data))
            try:
                alternate = generate(conservative, retry_alpha, retry_data, retry_loudness, actual_lift,
                                     formant_ratio_applied,
                                     source_stats if formant_ratio_applied == formant_candidate else None) if np.isfinite(retry_loudness) else None
            except Exception as exc:
                rescue_available = (song_octave_shift == 12
                                    and isinstance(settings.get("domain_ffmpeg"), str)
                                    and bool(settings.get("domain_ffmpeg", "").strip()))
                if not valid and not rescue_available:
                    raise
                # Keep an unusable +12 primary alive for the bounded optional
                # lower-shift ladder; elsewhere, retain the established error
                # behavior for a failed retry with no usable render.
                alternate = None
                retry_domain_fallback_reason = f"retry_render_failed:{type(exc).__name__}"
            alternative, (new_loss, new_excess, new_valid), alternate_state = alternate or (
                output, (None, None, False), render_state,
            )
            pitch_improves = loss is not None and new_loss is not None and loss > quality.LOSS_LIMIT and new_loss < loss
            keeps_voicing = loss is None or (new_loss is not None and new_loss <= loss + 0.02)
            band_improves = (excess is not None and new_excess is not None and excess > 6
                             and new_excess < excess - 0.5 and keeps_voicing)
            if new_valid and (not valid or pitch_improves or band_improves):
                prior_state = render_state
                output, loss, excess, valid = alternative, new_loss, new_excess, True
                guidance, alpha, accepted = conservative, alternate_state["cvec_alpha_applied"], True
                render_data, render_loudness = retry_data, retry_loudness
                render_state = alternate_state
                for key in ("virtual_fallback_reason", "formant_fallback_reason"):
                    if key in prior_state:
                        render_state.setdefault(key, prior_state[key])
                actual_lift = render_state["virtual_lift_applied"]
                formant_ratio_applied = render_state["formant_ratio_applied"]

        rescue = {
            "candidate_attempted": False, "candidate_applied": False,
            "candidate_model": "RIFT lower shift + formant-preserving Rubber Band",
            "model": None, "rift_key_shift_applied": None,
            "post_shift_target_semitones": 12 if song_octave_shift == 12 else None,
            "before": None, "after": None, "candidates": [], "rejection_reason": None,
        }
        if song_octave_shift != 12:
            rescue["rejection_reason"] = "not_whole_song_plus_12"
        elif not isinstance(settings.get("domain_ffmpeg"), str) or not settings.get("domain_ffmpeg", "").strip():
            rescue["rejection_reason"] = "rubberband_ffmpeg_not_configured"
        else:
            if valid:
                try:
                    primary_f0, primary_confidence = pitch(output)
                    primary_metrics = _upper_register_metrics(
                        f0, source_confidence, primary_f0, primary_confidence,
                    )
                except Exception:
                    primary_metrics = None
            else:
                primary_metrics = _upper_register_metrics(
                    f0, source_confidence, np.zeros_like(f0), np.zeros_like(source_confidence),
                )
            rescue["before"] = _rescue_metrics_json(primary_metrics)
            trigger_reasons, gate_reason = _upper_register_gate(primary_metrics)
            if gate_reason:
                rescue["rejection_reason"] = gate_reason
            else:
                rescue["candidate_attempted"] = True
                best = None
                rescue_alpha = render_state["cvec_alpha_applied"]
                rescue_ratio = render_state["formant_ratio_applied"]
                rescue_stats = source_stats if rescue_ratio == formant_candidate else None
                for compensation in (4, 8, 12):
                    virtual_lift = -compensation
                    rift_shift = final_shift + virtual_lift
                    try:
                        candidate_output, (candidate_loss, candidate_excess, candidate_valid), candidate_state = generate(
                            guidance, rescue_alpha, render_data, render_loudness, virtual_lift,
                            rescue_ratio, rescue_stats,
                        )
                        candidate_f0, candidate_confidence = pitch(candidate_output)
                        candidate_metrics = _upper_register_metrics(
                            f0, source_confidence, candidate_f0, candidate_confidence,
                        )
                    except Exception as exc:
                        rescue["candidates"].append({
                            "model": "RIFT lower shift + formant-preserving Rubber Band",
                            "rift_shift_semitones": rift_shift,
                            "compensation_semitones": compensation,
                            "candidate_attempted": True, "candidate_applied": False,
                            "rejection_reason": f"candidate_render_failed:{type(exc).__name__}",
                        })
                        rescue["rejection_reason"] = f"candidate_render_failed:{type(exc).__name__}"
                        detail = str(exc).casefold()
                        ffmpeg_unavailable = ("ffmpeg pitch retuning failed" in detail
                                              or "retuning exited with status" in detail
                                              or "no ffmpeg executable" in detail)
                        if ffmpeg_unavailable:
                            break
                        continue

                    passes_metrics, candidate_rejection, complete = _rescue_candidate_quality(
                        candidate_metrics, primary_metrics,
                    )
                    if not candidate_valid:
                        passes_metrics, candidate_rejection, complete = False, "candidate_audio_unusable", False
                    candidate_entry = {
                        "model": "RIFT lower shift + formant-preserving Rubber Band",
                        "rift_shift_semitones": rift_shift,
                        "compensation_semitones": compensation,
                        "candidate_attempted": True, "candidate_applied": False,
                        "post_shift_target_semitones": 12,
                        "post_shift_semitones": (round(candidate_metrics["post_shift_semitones"], 3)
                                                  if candidate_metrics is not None
                                                  and candidate_metrics["post_shift_semitones"] is not None else None),
                        "before": rescue["before"],
                        "after": _rescue_metrics_json(candidate_metrics),
                        "rejection_reason": candidate_rejection,
                    }
                    rescue["candidates"].append(candidate_entry)
                    if passes_metrics:
                        score = _rescue_score(candidate_metrics)
                        if complete:
                            # Completion is the primary preference: this is
                            # the first (smallest-compensation) candidate that
                            # meets the voicing and expected-octave targets.
                            best = {
                                "score": score, "output": candidate_output,
                                "loss": candidate_loss, "excess": candidate_excess,
                                "state": candidate_state, "metrics": candidate_metrics,
                                "entry": candidate_entry,
                            }
                            break
                        if best is None or score > best["score"]:
                            best = {
                                "score": score, "output": candidate_output,
                                "loss": candidate_loss, "excess": candidate_excess,
                                "state": candidate_state, "metrics": candidate_metrics,
                                "entry": candidate_entry,
                            }

                if best is not None:
                    output, loss, excess = best["output"], best["loss"], best["excess"]
                    valid, render_state = True, best["state"]
                    actual_lift = render_state["virtual_lift_applied"]
                    formant_ratio_applied = render_state["formant_ratio_applied"]
                    alpha = render_state["cvec_alpha_applied"]
                    best["entry"]["candidate_applied"] = True
                    rescue.update({
                        "candidate_applied": True,
                        "model": "RIFT lower shift + formant-preserving Rubber Band",
                        "rift_key_shift_applied": render_state["rift_key_shift_applied"],
                        "post_shift_semitones": best["metrics"]["post_shift_semitones"],
                        "after": _rescue_metrics_json(best["metrics"]),
                        "rejection_reason": None,
                    })
                elif rescue["rejection_reason"] is None:
                    rescue["rejection_reason"] = "no_candidate_passed_quality_checks"
                rescue["trigger_reasons"] = trigger_reasons
        rescue_summaries.append(rescue)
        actual_lift = render_state["virtual_lift_applied"]
        formant_ratio_applied = render_state["formant_ratio_applied"]
        alpha = render_state["cvec_alpha_applied"]
        if probe_contexts is not None:
            for phrase_index in section.phrases:
                probe_contexts[phrase_index] = {
                    "valid": bool(valid),
                    # Retuning around a lower RIFT key returns to final_shift,
                    # so this remains the effective final pitch displacement.
                    "effective_shift": int(final_shift if valid else 0),
                }
        if valid:
            output, excess, cut = quality.tame_high_band(source, output, excess)
            fade = min(EDGE_FADE, len(output) // 2)
            if fade:
                ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
                output[:fade] *= ramp
                output[-fade:] *= ramp[::-1]
            combined[section.start:section.end] = output
        else:
            cut = 0.0
        checks.append({"section": i, "cluster": section.cluster, "start_seconds": round(section.start / RATE, 4),
                       "end_seconds": round(section.end / RATE, 4), "phrase_indices": section.phrases,
                       "context_before_seconds": round((section.start - start) / RATE, 4),
                       "context_after_seconds": round((end - section.end) / RATE, 4),
                       "route": cluster.route, "voiced_frame_loss": loss, "high_band_excess_db": excess,
                       "high_band_cut_db": cut, "retried": bool(retry), "accepted_retry": accepted, "valid": valid,
                       "retry_high_band_boost_capped": cap_high_boost,
                       "key_shift": final_shift, "cvec_alpha": alpha, "speaker_cfg": guidance.speaker,
                       "detail_cfg": guidance.detail, "cfg_rescale": guidance.rescale,
                       "domain_plan_granularity": domain_plan.granularity,
                       "domain_reason": domain_plan.reason or cluster.domain_reason,
                       "domain_octave_shift_candidate": final_shift,
                       "domain_octave_shift_applied": final_shift if valid else 0,
                       "domain_virtual_lift_candidate": domain_plan.virtual_lift_candidate,
                       "domain_virtual_lift_applied": actual_lift,
                       "domain_rift_key_shift_applied": render_state["rift_key_shift_applied"],
                       "domain_formant_ratio_candidate": domain_plan.formant_ratio_candidate,
                       "domain_formant_ratio_applied": formant_ratio_applied,
                       "domain_source_formant_anchor_hz": domain_plan.source_anchor,
                       "domain_source_formant_frames": domain_plan.source_anchor_frames,
                       "domain_source_formant_log_mad": domain_plan.source_anchor_log_mad,
                       "domain_retune": render_state["retune"],
                       "domain_virtual_fallback_reason": render_state.get("virtual_fallback_reason"),
                       "domain_virtual_fallback_rejected_reason": render_state.get("virtual_fallback_rejected_reason"),
                       "domain_formant_fallback_reason": render_state.get("formant_fallback_reason"),
                       "domain_upper_register_rescue": rescue,
                       "retry_domain_fallback_reason": retry_domain_fallback_reason})

    if not any(check["valid"] for check in checks):
        raise RuntimeError("No audible output remained after the RIFT retry.")
    output_audio = quality.restore_dynamics(mono, quality.polish_vocal(combined))
    report = {
        "timbre_slicing": True, "segmentation": "silero_wespeaker_phrases", "voice_profile": profile["speaker"] if profile else None,
        "phrases": [{"phrase": i, "cluster": int(label), "start_seconds": round(phrase.start / RATE, 4),
                     "end_seconds": round(phrase.end / RATE, 4), "boundary": phrase.boundary}
                    for i, (phrase, label) in enumerate(zip(phrases, labels, strict=True))],
        "clusters": [{"cluster": label, "key_shift": c.shift, "pitch_outside_range_ratio": c.outside,
                      "route": c.route, "ltas_drift_db": c.drift, "gain_db": round(20 * np.log10(c.gain), 3),
                      "eq_db": c.gains.tolist() if c.gains is not None else None, "cvec_alpha": c.alpha,
                      "domain_octave_shift_candidate": c.shift,
                      "domain_virtual_lift_candidates": [
                          {"phrase": index, "virtual_lift": phrase_plans[index].virtual_lift,
                           "final_octave_shift": phrase_plans[index].octave_shift,
                           "reason": phrase_plans[index].reason}
                          for index, value in enumerate(labels) if int(value) == label
                      ],
                      "domain_virtual_lift_cluster_diagnostic": (
                          getattr(c.domain_plan, "virtual_lift", 0) if c.domain_plan is not None else 0
                      ),
                      "domain_formant_ratio_candidate": c.formant_ratio,
                      "domain_formant_anchor_hz": c.source_anchor,
                      "domain_formant_frames": c.source_anchor_frames,
                      "domain_formant_log_mad": c.source_anchor_log_mad,
                      "domain_reason": c.domain_reason,
                      "pitch_tails_p05_p95": profile.get("pitch_tails") if profile else None}
                     for label, c in clusters.items()],
        "sections": checks, "chunks": checks,
        "upper_register_voicing_rescue": {
            "scope": "whole_song_plus_12",
            "attempted_sections": sum(bool(item["candidate_attempted"]) for item in rescue_summaries),
            "applied_sections": sum(bool(item["candidate_applied"]) for item in rescue_summaries),
            "sections": rescue_summaries,
        },
        "song_octave_shift": int(song_octave_shift),
        "song_octave_source": song_octave_source,
        "song_octave_reason": song_octave_reason,
        "song_octave_diagnostics": ({
            "reliable_phrases": song_octave_plan.reliable_phrases,
            "pooled_notes": song_octave_plan.pooled_notes,
            "target_bounds": song_octave_plan.target_bounds,
            "pooled_outside_ratio": song_octave_plan.pooled_outside_ratio,
            "phrase_majority_ratio": song_octave_plan.phrase_majority_ratio,
            "shifted_in_range_ratio": song_octave_plan.shifted_in_range_ratio,
            "tail_improvement": song_octave_plan.tail_improvement,
        } if song_octave_plan is not None else None),
        "needs_review": any((cluster.outside or 0) > 0.25 for cluster in clusters.values()) or any(
            not check["valid"] or (check["voiced_frame_loss"] or 0) > quality.LOSS_LIMIT
            or (check["high_band_excess_db"] or 0) > 6 for check in checks
        ),
    }
    if probe_contexts is not None:
        report["song_octave_probe"] = _song_octave_probe(
            mono, output_audio, phrases, tracks, settings, probe_contexts, infer, pitch,
            int(song_octave_shift), song_octave_source, models, extract_content,
        )
    return output_audio, report
