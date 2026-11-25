from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from music_runtime.job_store import JobStore, SelectionConflict  # noqa: E402
from music_runtime.service import MUSIC_SERVICE_KEY, MusicService, ServiceAlreadyRunning, create_app  # noqa: E402
from music_runtime.settings import PipelineError  # noqa: E402


TOKEN = "api-test-token-" + ("x" * 40)


class FakeProvider:
    def __init__(self, ready: bool = True):
        self.ready = ready

    async def check_account_ready(self) -> None:
        if not self.ready:
            raise PipelineError("login_required", "No Netease login is saved.")


class FakeRuntime:
    def __init__(self, ready: bool = True):
        self.ready = ready

    async def check_ready(self) -> dict:
        return {
            "ready": self.ready,
            "components": {
                "rift": {"ready": self.ready},
                "separation": {"ready": self.ready},
                "netease": {"ready": self.ready},
            },
        }


class FakePipeline:
    def __init__(self, store: JobStore, data_dir: Path, *, ready: bool = True, enabled: bool = True):
        self.store = store
        self.data_dir = data_dir
        self.enabled = enabled
        self.provider = FakeProvider(ready)
        self.runtime = FakeRuntime(ready)

    async def ensure_started(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def submit(self, request_id: str, query: str) -> dict:
        job, created = self.store.create_request_job(request_id, query)
        if created:
            if query == "ambiguous":
                self.store.update(job["id"], status="selection_required", stage="selection_required", progress="Choose a result", candidates=[
                    {"id": "11", "title": "First", "artist": "A"},
                    {"id": "22", "title": "Second", "artist": "B"},
                ])
            else:
                self._complete(job["id"], query)
        return self.store.get_job_view(job["id"])

    def _complete(self, job_id: str, query: str) -> None:
        root = self.data_dir / "artifacts"
        root.mkdir(parents=True, exist_ok=True)
        artifact = root / f"{job_id}.wav"
        artifact.write_bytes(b"RIFF-test-wav-payload")
        self.store.update(job_id, song={"id": "99", "title": query, "artist": "Test Artist"})
        self.store.record_artifact(job_id, "output", "fingerprint", {"wav": str(artifact)})
        self.store.update(job_id, status="ready", stage="ready", progress="Completed")

    async def choose(self, job_id: str, candidate_id: str) -> dict:
        try:
            result, started = self.store.select_candidate(job_id, candidate_id)
        except KeyError:
            raise PipelineError("job_not_found", "No matching music job was found.") from None
        except SelectionConflict as exc:
            raise PipelineError("selection_conflict", str(exc)) from None
        if started:
            canonical_id = result["canonical_id"]
            self._complete(canonical_id, result["song"]["title"])
        return self.store.get_job_view(job_id)


class ServiceApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name)
        self.client: TestClient | None = None
        await self._open_client(ready=True)

    async def asyncTearDown(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None
        self.temp.cleanup()

    async def _open_client(self, *, ready: bool, enabled: bool = True) -> None:
        store = JobStore(self.data / "jobs.sqlite")
        pipeline = FakePipeline(store, self.data, ready=ready, enabled=enabled)
        service = MusicService(
            settings={"runtime_dir": self.data, "pipeline": {"max_queue": 5}},
            pipeline=pipeline,
        )
        self.client = TestClient(TestServer(create_app(service, TOKEN)))
        await self.client.start_server()
        self.auth = {"Authorization": f"Bearer {TOKEN}"}

    async def _submit(self, request_id: str, query: str) -> dict:
        response = await self.client.post(
            "/v1/jobs", headers=self.auth,
            json={"request_id": request_id, "query": query},
        )
        return response, await response.json()

    async def test_auth_covers_health_and_health_reports_not_ready(self) -> None:
        response = await self.client.get("/v1/health")
        self.assertEqual(response.status, 401)
        response = await self.client.get("/v1/health", headers={"Authorization": "Bearer wrong", "Cookie": "secret-cookie"})
        self.assertEqual(response.status, 401)
        self.assertNotIn("secret-cookie", await response.text())
        response = await self.client.get("/v1/health", headers=self.auth)
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.json(), {"alive": True, "ready": True})

    async def test_exact_request_replay_survives_restart_and_readiness_failure(self) -> None:
        response, first = await self._submit("a" * 32, "complete song")
        self.assertEqual(response.status, 202)
        self.assertEqual(first["status"], "ready")
        response, replay = await self._submit("a" * 32, "complete song")
        self.assertEqual(response.status, 202)
        self.assertEqual(first["id"], replay["id"])

        await self.client.close()
        self.client = None
        await self._open_client(ready=False)

        health = await self.client.get("/v1/health", headers=self.auth)
        self.assertEqual((await health.json())["reason"], "login_required")
        response, after_restart = await self._submit("a" * 32, "complete song")
        self.assertEqual(response.status, 202)
        self.assertEqual(after_restart["id"], first["id"])
        response, new_job_error = await self._submit("b" * 32, "new request")
        self.assertEqual(response.status, 503)
        self.assertEqual(new_job_error["error_code"], "login_required")

        status = await self.client.get(f"/v1/jobs/{first['id']}", headers=self.auth)
        state = await status.json()
        self.assertEqual(status.status, 200)
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["artifact"]["sha256"], hashlib.sha256(b"RIFF-test-wav-payload").hexdigest())
        artifact = await self.client.get(f"/v1/jobs/{first['id']}/artifact", headers=self.auth)
        self.assertEqual(artifact.status, 200)
        self.assertEqual(await artifact.read(), b"RIFF-test-wav-payload")

    async def test_request_id_mismatch_is_conflict_before_readiness_check(self) -> None:
        response, first = await self._submit("c" * 32, "complete song")
        self.assertEqual(response.status, 202)
        await self.client.close()
        self.client = None
        await self._open_client(ready=False)
        response, body = await self._submit("c" * 32, "different query")
        self.assertEqual(response.status, 409)
        self.assertEqual(body["error_code"], "request_id_conflict")

    async def test_disabled_admin_mode_keeps_existing_state_readable(self) -> None:
        await self.client.close()
        self.client = None
        store = JobStore(self.data / "jobs.sqlite")
        existing, _ = store.create_request_job("f" * 32, "complete song")
        root = self.data / "artifacts"
        root.mkdir(parents=True, exist_ok=True)
        wav = root / f"{existing['id']}.wav"
        wav.write_bytes(b"RIFF-test-wav-payload")
        store.record_artifact(existing["id"], "output", "fingerprint", {"wav": str(wav)})
        store.update(existing["id"], status="ready", stage="ready", song={"id": "99", "title": "Known", "artist": "A"})
        await self._open_client(ready=False, enabled=False)

        health = await self.client.get("/v1/health", headers=self.auth)
        self.assertEqual((await health.json())["reason"], "pipeline_disabled")
        response, replay = await self._submit("f" * 32, "complete song")
        self.assertEqual(response.status, 202)
        self.assertEqual(replay["id"], existing["id"])
        state = await self.client.get(f"/v1/jobs/{existing['id']}", headers=self.auth)
        self.assertEqual((await state.json())["status"], "ready")
        artifact = await self.client.get(f"/v1/jobs/{existing['id']}/artifact", headers=self.auth)
        self.assertEqual(await artifact.read(), b"RIFF-test-wav-payload")
        response, error = await self._submit("e" * 32, "new song")
        self.assertEqual(response.status, 503)
        self.assertEqual(error["error_code"], "pipeline_disabled")

    async def test_selection_same_retry_is_idempotent_and_different_choice_conflicts(self) -> None:
        response, job = await self._submit("d" * 32, "ambiguous")
        self.assertEqual(response.status, 202)
        self.assertEqual(job["status"], "selection_required")
        path = f"/v1/jobs/{job['id']}/selection"
        selected = await self.client.post(path, headers=self.auth, json={"candidate_id": "11"})
        self.assertEqual(selected.status, 200)
        selected_job = await selected.json()
        self.assertEqual(selected_job["status"], "ready")
        retry = await self.client.post(path, headers=self.auth, json={"candidate_id": "11"})
        self.assertEqual(retry.status, 200)
        self.assertEqual((await retry.json())["id"], job["id"])
        conflict = await self.client.post(path, headers=self.auth, json={"candidate_id": "22"})
        self.assertEqual(conflict.status, 409)

    async def test_artifact_supports_ranges_and_rejects_tampered_paths(self) -> None:
        response, job = await self._submit("e" * 32, "complete song")
        self.assertEqual(response.status, 202)
        path = f"/v1/jobs/{job['id']}/artifact"
        ranged = await self.client.get(path, headers={**self.auth, "Range": "bytes=2-6"})
        self.assertEqual(ranged.status, 206)
        self.assertEqual(await ranged.read(), b"FF-te")
        self.assertEqual(ranged.headers["Content-Range"], "bytes 2-6/21")

        store = self.client.server.app[MUSIC_SERVICE_KEY].store
        outside = self.data.parent / "outside.wav"
        outside.write_bytes(b"secret external file")
        store.record_artifact(job["id"], "output", "fingerprint", {"wav": str(outside)})
        status = await self.client.get(f"/v1/jobs/{job['id']}", headers=self.auth)
        body = await status.json()
        self.assertEqual(status.status, 200)
        self.assertEqual(body["status"], "failed")
        self.assertNotIn(str(outside), await status.text())
        denied = await self.client.get(path, headers=self.auth)
        self.assertEqual(denied.status, 409)
        self.assertNotEqual(await denied.read(), b"secret external file")

    async def test_invalid_inputs_and_unmatched_paths_are_http_errors(self) -> None:
        response = await self.client.post("/v1/jobs", headers=self.auth, json={"request_id": "bad", "query": "song"})
        self.assertGreaterEqual(response.status, 400)
        response = await self.client.get("/v1/jobs/not-a-job-id", headers=self.auth)
        self.assertGreaterEqual(response.status, 400)
        response = await self.client.get("/v1/jobs/../../outside", headers=self.auth, allow_redirects=False)
        self.assertGreaterEqual(response.status, 400)

    async def test_single_service_process_lock(self) -> None:
        service = self.client.server.app[MUSIC_SERVICE_KEY]
        with self.assertRaises(ServiceAlreadyRunning):
            MusicService(
                settings={"runtime_dir": self.data, "pipeline": {"max_queue": 5}},
                pipeline=FakePipeline(service.store, self.data),
            )


if __name__ == "__main__":
    unittest.main()
