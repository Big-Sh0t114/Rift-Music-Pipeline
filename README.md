# NachoBot Music Runtime

Standalone music production service for a GPU host. It searches and resolves Netease tracks, downloads and converts audio, runs UVR separation and RIFT-SVC conversion, mixes the WAV output, and stores durable jobs and artifacts. It does not import NachoBot Core, send messages, or write to the bot's music catalog.

## Host setup

The bundled installer and pinned runtime currently target **Windows x64 with Python 3.11.14**. The Netease API uses the pinned Node 24.11.1 Windows x64 archive, and the managed FFmpeg path is `ffmpeg.exe`. The CPU and CUDA audio locks and the RIFT `uv.lock` are included for reproducible preparation. Select `--device cuda` on a compatible NVIDIA host. The HTTP application is built on `aiohttp`, but the checked-in installer does not provide a supported Linux or macOS preparation path; treat those hosts as unverified until each provider, FFmpeg, UVR, and RIFT runtime is prepared independently.

Create an isolated Python 3.11 environment and install only the service dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The code defaults to `%LOCALAPPDATA%\NachoBot\MusicRuntime` on Windows (or `$XDG_DATA_HOME/nachobot-music-runtime` on other systems). Set `MUSIC_RUNTIME_DATA_DIR` before launching to choose a different durable directory. Copy `config.example.toml` to `config.toml` in that data directory, then configure the RIFT checkpoints and speaker names. The runtime will create its database, downloaded models, isolated environments, credentials, work files, QR images, and completed WAVs beneath this directory.

This default data directory is separate from NachoBot's existing `.runtime\mus_library`; the service does not import or migrate that database, credentials, cache, models, or artifacts. Do not point `MUSIC_RUNTIME_DATA_DIR` at that active directory.

## Prepare and log in

Run these commands on the GPU host. Preparation can download large model and runtime files; it is explicit and does not run during import or test startup.

```powershell
python scripts/manage.py check
python scripts/manage.py prepare --device cuda
python scripts/manage.py prepare-phrases
python scripts/manage.py prepare-octave-probe  # only when enabling optional probe QA
python scripts/manage.py login
python scripts/manage.py speakers
```

`login` prepares the small login component, writes a QR PNG under the runtime data directory, opens it locally on Windows, and waits for confirmation in the Netease app. The saved cookie remains in the runtime data directory. The authenticated readiness check validates the saved account through the local Netease service without loading UVR or RIFT models.

`check` reports preparation state. `speakers` reads names from the configured RIFT checkpoint. `prepare` supports `--component` values `all`, `control`, `audio`, `rift`, `login`, and `netease`.

## Start the API

Set a long random bearer token in the service environment. The token is read only from `NACHOBOT_MUSIC_API_TOKEN`; it is not stored in TOML or returned by any endpoint.

```powershell
$env:NACHOBOT_MUSIC_API_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(48))"
$env:MUSIC_RUNTIME_HOST = "127.0.0.1"
$env:MUSIC_RUNTIME_PORT = "8765"
python scripts/manage.py serve
```

The default listener is loopback. For a remote client, bind only on a protected GPU host network, restrict inbound firewall rules to the client, and place the connection behind a TLS reverse proxy. The service itself does not terminate TLS. Bearer tokens must only travel over a trusted or encrypted connection.

Every route, including health, requires `Authorization: Bearer <NACHOBOT_MUSIC_API_TOKEN>`. Responses are JSON except the WAV stream. The health endpoint returns HTTP 200 while the service is alive, with `ready: false` and a safe `reason` such as `pipeline_disabled`, `login_required`, `login_expired`, `netease_not_prepared`, `audio_models_not_ready`, or `rift_not_ready`. A new job is rejected with HTTP 503 until all provider, account, UVR, and RIFT checks pass. A retry with an already accepted `request_id` is replayed before readiness is checked, and status or completed artifact reads continue while readiness is false.

### API routes

- `GET /v1/health` — `{ "alive": true, "ready": true }` (or `ready: false` with a safe `reason`).
- `POST /v1/jobs` — JSON `{ "request_id": "<32 lowercase hex>", "query": "song query" }`; returns a durable job. Reusing a request ID with the same exact query returns the same job after a lost response or restart. Reusing it with another query returns HTTP 409.
- `GET /v1/jobs/{job_id}` — status and progress. IDs are 12 lowercase hexadecimal characters.
- `POST /v1/jobs/{job_id}/selection` — JSON `{ "candidate_id": "<provider song id>" }`. Retrying the same choice is safe; a different choice conflicts.
- `GET /v1/jobs/{job_id}/artifact` — streams the completed WAV. The route supports byte ranges and returns HTTP 409 before completion.

Job responses contain `id`, `status`, `stage`, `progress`, `query`, `song`, and `candidates`, plus safe error fields when present. Completed jobs use `status: "ready"` and include artifact `size` and `sha256`. When different requests resolve to the same song and processing profile, their original IDs remain usable aliases of the canonical output. Distinct unresolved requests never merge before the user selects a candidate.

The service serializes GPU work, owns one SQLite database under the runtime data directory, and holds a process lock for that directory. On restart, interrupted production jobs return to the queue while preserving stage artifacts for validation and reuse.

## Adaptive RIFT conversion

`[rift].adaptive_quality` defaults to `true`. RIFT conditions the dry stems at 44.1 kHz, splits on quiet gaps, and caps continuous chunks at 15 seconds with a 500 ms overlap. It uses RMVPE pitch confidence and pitch jumps to choose conservative guidance settings. If a chunk loses more than 15% of its voiced frames or returns unusable audio, RIFT tries it once with weaker guidance and retains the better result. It applies up to 6 dB of attenuation when a chunk has substantial excess energy in the 4–8 kHz band.

`[rift].timbre_slicing` defaults to `true` in adaptive mode. Prepare the pinned Silero and WeSpeaker models once with `python scripts/manage.py prepare-phrases`. Silero pause candidates are checked against local energy to keep sung passages the speech model misses. Continuous phrases longer than 12 seconds split at a nearby low-energy point.

WeSpeaker embeddings and four acoustic measurements group phrases by average-linkage clustering into up to four clusters. Successive phrases in the same cluster merge up to 14 seconds. With a speaker profile, each cluster shares its LTAS gain curve, guidance, and pooled ContentVec statistics at the same formant ratio. The worker also applies a shared loudness gain without a speaker profile. Each merged segment receives up to 300 ms of real audio context per edge. The worker trims that context after inference and applies a 35 ms fade on each edge. QA records the detected phrases, boundary types, cluster settings, and merged sections. Set `timbre_slicing = false` for silence- and duration-based cuts.

An optional `lead_pitch_range` or `backing_pitch_range` takes three MIDI notes in `[P10, median, P90]` order for the configured speaker. A nonzero manual key shift takes precedence. In the phrase path (`timbre_slicing = true`), `domain_alignment = true` lets RIFT use a reversible ±1–5 semitone lift for an individual natural phrase when its voiced pitch evidence supports it and the configured FFmpeg executable can run Rubber Band. The generated phrase is retuned by the opposite amount before context trimming. If retuning or its pitch and length checks fail, the worker rerenders at the structural shift and records the candidate and applied values in QA.

The adaptive phrase path uses speaker profiles and explicit pitch ranges for conservative local matching. Set `allow_octave_jump = true` to permit evidence-based octave changes there. A phrase can retain +12 semitones when at least half of its confident voiced frames are below target P10, its own P10 is at least three semitones below that floor, and its raised pitch distribution fits the lower and upper guards. The phrase needs at least one second of confident voiced frames and at least 50% voiced occupancy. The worker also checks each phrase before applying an octave suggested by its cluster. QA identifies retained low phrases as `sustained_low_register_octave` and records the final shift.

With adaptive phrase processing and `allow_octave_jump = true`, the lead can also report one evidence-based song-wide octave decision. The worker writes that nonzero automatic decision to a small job-local sidecar; the pipeline reads only that bounded sidecar and passes the same shift to backing, preserving alignment. If either role has a nonzero manual key shift, song-wide automatic detection is suppressed and the manual role settings remain in effect. The silence- and duration-based path (`timbre_slicing = false`) does not make this automatic decision.

The optional `[rift].octave_probe` defaults to `false`. When enabled for a configured lead voice, it runs only on the eligible adaptive, phrase-sliced, domain-aligned automatic route with octave jumps enabled and no manual/global shift. Prepare its separate, SHA-pinned Essentia ONNX bundle explicitly with `python scripts/manage.py prepare-octave-probe`; the CPU ONNX Runtime package is reused from `prepare-phrases`. No startup download is performed. If the optional models are missing, corrupt, or fail to load, the normal conversion still completes and QA records a short unavailable reason.

The probe classifies up to three spaced lower, middle, and upper reliably voiced excerpts from the dry lead at 16 kHz. Its `female`/`male` labels are acoustic model labels and do not identify a person. It gates gender votes on the Essentia voice score, uses confidence thresholds, and reports `unknown` when phrase votes conflict or pooled confidence is sparse. `[rift].octave_probe_target_voice` accepts `unknown`, `female`, or `male` and affects candidate ordering only.

At most six short whole-candidate comparisons are made for the selected excerpts. Each candidate reruns the normal phrase pipeline on that excerpt plus context, including existing retries and the +12 upper-register rescue where applicable; it does not rerun the full song. QA stores the selected intervals, actual v9 shifts, candidate order, per-sample pitch coverage/local loss, and a nullable `recommendation.shift` in `song_octave_probe`. A numeric zero is an explicit recommendation; `null` means no recommendation. The suggestion stays in QA and never changes PCM or the lead-to-backing sidecar. These initial coverage and local-loss thresholds are shadow-only listening heuristics, not a validated hoarseness score. With three four-second excerpts, the base work covers about 12 seconds of classifier audio and at most about 24 seconds of candidate audio, plus context and any ordinary retry/rescue renders. Actual CPU/GPU time depends on the checkpoint and host.

Use `[rift].song_octave_shift = 12` or `-12` for a deliberate whole-song lock. It applies that exact octave shift to both lead and backing, bypasses automatic song detection, and works with or without adaptive phrase processing. `0` (the default) leaves the profile-based and evidence-based automatic policies in control. Only `-12`, `0`, and `12` are accepted. A nonzero lock cannot be combined with either per-role manual `key_shift` override.

For a retained whole-song +12 lead with configured FFmpeg, confident source F0 can trigger a bounded upper-register dropout rescue after the normal same-shift retry. It checks global voiced-frame loss, expected +12 F0 coverage, and 1.5-second voiced windows (at least 50 confident source frames and 40% local loss). It then tries RIFT at +8, +4, and 0 semitones, followed by formant-preserving Rubber Band compensation in the real-audio context before trimming. A candidate must be usable, materially improve voiced coverage, and measure within 1 semitone of the requested final +12. The worker prefers the smallest compensation that reaches at most 8% voiced-frame loss and at least 65% expected-octave coverage; otherwise it keeps the best valid improvement. Failed retuning, unusable candidates, or inconclusive pitch evidence leave the primary render in place. Per-section `domain_upper_register_rescue` QA and the top-level `upper_register_voicing_rescue` summary record trigger reasons, candidate attempts, RIFT and measured post shifts, before/after voicing and octave coverage, and rejection reasons.

In the phrase path, `domain_formants = true` uses reliable Praat formant anchors from a version 2 speaker profile to warp only the 16 kHz ContentVec input. RMVPE F0, loudness analysis, and spectral matching use the unwarped waveform; ContentVec conditioning can change the final converted audio. Pooled ContentVec source statistics use the same formant ratio as the sections that consume them; sections without matching statistics use zero ContentVec alignment weight. The formant path can still run when that weight is zero. QA reports per-phrase pitch candidates and per-section formant ratios, applied values, and fallback reasons. The worker restores the stem's slow loudness envelope and original timeline before mixing. FFmpeg measures the final mix's true peak and applies global attenuation when needed to stay below -1 dBTP.

Inspect the lead and backing QA JSON files under `jobs/<job_id>/stages/voice_converting/` for per-chunk pitch loss, 4–8 kHz energy change, and retry outcomes. `needs_review` flags sections for listening review. Set `adaptive_quality = false` to use fixed guidance settings and sample-peak mastering.

### Speaker profiles

Build a profile from the clean WAV or FLAC files for one speaker. The command uses the configured RIFT environment and extracts voiced MIDI percentiles and tails, an 18-band LTAS, ContentVec channel statistics, and a confidence-checked formant anchor. The reader continues to accept version 1 profiles; version 2 profiles add the pitch tails and optional formant summary. Use a new JSON destination for each profile.

```powershell
python scripts/manage.py profile-speaker --speaker speaker530 --input C:\training\speaker530 --output C:\profiles\speaker530.json
```

Set `[rift].lead_voice_profile` or `[rift].backing_voice_profile` to that JSON path and start the runtime with `adaptive_quality = true`. Profiles are loaded at startup and checked against the selected speaker and ContentVec weights. A new profile changes the job and conversion cache fingerprints. Use separate profiles when the two roles use different speakers.

Each voiced cluster gets broad spectral matching limited to ±6 dB, voiced loudness conditioning around −18 LUFS, and soft ContentVec alignment. Clusters with larger spectral drift use weaker source detail guidance and stronger target speaker guidance. Phrase-local domain planning uses the profile's `[P10, median, P90]` range. Retained high octaves use version 2 P95 as the upper reference and use `P90 + 2` as the estimate for version 1 profiles and explicit pitch ranges. QA records the pitch tails, spectral drift, EQ, ContentVec weight, guidance, final shift, virtual lift, and formant ratio.

## Bounded verification

The standalone unit and API checks use temporary data directories, fake providers, and small local WAV fixtures. They do not download models, contact Netease, load checkpoints, or run GPU inference. `check` and `prepare` on the target Windows GPU host are required to validate its actual CUDA, UVR, RIFT checkpoint, Netease API, and login state.
