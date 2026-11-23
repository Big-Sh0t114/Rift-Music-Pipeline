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

`[rift].adaptive_quality` defaults to `true`. RIFT conditions the dry stems at 44.1 kHz, splits on quiet gaps, and caps continuous chunks at 15 seconds with a 500 ms overlap. It uses RMVPE pitch confidence and pitch jumps to choose conservative guidance settings. If a chunk loses more than 15% of its voiced frames or returns unusable audio, RIFT tries it once with weaker guidance and retains the better result.

An optional `lead_pitch_range` or `backing_pitch_range` takes three MIDI notes in `[P10, median, P90]` order for the configured speaker. With the corresponding key shift at zero, RIFT can choose an octave shift based on pitch coverage. A nonzero manual key shift takes precedence. The worker restores the stem's slow loudness envelope and original timeline before mixing. FFmpeg measures the final mix's true peak and applies global attenuation when needed to stay below -1 dBTP.

Inspect `lead.qa.json` and `backing.qa.json` under `jobs/<job_id>/stages/voice_converting/` for per-chunk pitch loss, 4–8 kHz energy change, and retry outcomes. `needs_review` flags an estimate for listening review. Set `adaptive_quality = false` to use fixed guidance settings and sample-peak mastering.

## Bounded verification

The standalone unit and API checks use temporary data directories, fake providers, and small local WAV fixtures. They do not download models, contact Netease, load checkpoints, or run GPU inference. `check` and `prepare` on the target Windows GPU host are required to validate its actual CUDA, UVR, RIFT checkpoint, Netease API, and login state.
