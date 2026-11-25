from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

from music_runtime.job_store import JobStore, RequestConflict, SelectionConflict  # noqa: E402


class JobStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = JobStore(self.root / "jobs.sqlite")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def create(self, request_id: str, query: str) -> dict:
        job, created = self.store.create_request_job(request_id, query)
        self.assertTrue(created)
        return job

    def test_request_identity_is_exact_and_survives_terminal_state(self) -> None:
        job = self.create("a" * 32, "exact query")
        self.store.update(job["id"], status="failed", error_code="test")
        replay, created = self.store.create_request_job("a" * 32, "exact query")
        self.assertFalse(created)
        self.assertEqual(replay["id"], job["id"])
        self.assertEqual(replay["status"], "failed")
        with self.assertRaises(RequestConflict):
            self.store.create_request_job("a" * 32, "different query")

    def test_distinct_unresolved_requests_do_not_merge(self) -> None:
        first = self.create("a" * 32, "same title")
        second = self.create("b" * 32, "same title")
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(self.store.resolve_job_id(first["id"]), first["id"])
        self.assertEqual(self.store.resolve_job_id(second["id"]), second["id"])

    def test_choice_is_atomic_and_retryable_through_canonical_alias(self) -> None:
        canonical = self.create("b" * 32, "known song id")
        self.store.update(canonical["id"], status="ready", song={"id": "11", "title": "Song", "artist": "A"})
        self.store.bind_dedup_key(canonical["id"], "resolved-fingerprint")

        chooser = self.create("a" * 32, "ambiguous title")
        self.store.update(
            chooser["id"], status="selection_required", candidates=[
                {"id": "11", "title": "Song", "artist": "A"},
                {"id": "22", "title": "Other", "artist": "B"},
            ],
        )
        selected, changed = self.store.select_candidate(chooser["id"], "11")
        self.assertTrue(changed)
        self.assertEqual(selected["status"], "queued")
        self.assertEqual(self.store.bind_dedup_key(chooser["id"], "resolved-fingerprint"), canonical["id"])

        retried, changed = self.store.select_candidate(chooser["id"], "11")
        self.assertFalse(changed)
        self.assertEqual(retried["id"], chooser["id"])
        self.assertEqual(retried["canonical_id"], canonical["id"])
        self.assertEqual(retried["status"], "ready")
        with self.assertRaises(SelectionConflict):
            self.store.select_candidate(chooser["id"], "22")

    def test_failed_job_releases_resolved_dedup_key(self) -> None:
        failed = self.create("a" * 32, "first query")
        self.store.bind_dedup_key(failed["id"], "same-song-profile")
        self.store.update(failed["id"], status="failed", error_code="test")
        self.assertIsNone(self.store.find_dedup_key("same-song-profile"))
        retry = self.create("b" * 32, "second query")
        self.assertEqual(self.store.bind_dedup_key(retry["id"], "same-song-profile"), retry["id"])

    def test_restart_requeues_running_stage_and_keeps_checkpoints(self) -> None:
        job = self.create("a" * 32, "song")
        artifact = self.root / "stage.wav"
        artifact.write_bytes(b"RIFF-test")
        self.store.record_artifact(job["id"], "separating", "fingerprint", {"vocals": str(artifact)})
        self.store.update(job["id"], status="mixing", stage="mixing", progress="Mixing")
        self.store.recover_after_restart()
        resumed = self.store.get_job(job["id"])
        self.assertEqual(resumed["status"], "queued")
        self.assertEqual(resumed["progress"], "Resuming after restart")
        self.assertTrue(Path(resumed["artifacts"]["separating"]["vocals"]).is_file())

    def test_refuses_legacy_database_without_importing_its_state(self) -> None:
        path = self.root / "legacy.sqlite"
        with closing(sqlite3.connect(path)) as conn:
            conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO jobs VALUES('legacy')")
            conn.commit()
        with self.assertRaisesRegex(RuntimeError, "fresh MUSIC_RUNTIME_DATA_DIR"):
            JobStore(path)
        with closing(sqlite3.connect(path)) as conn:
            self.assertEqual(conn.execute("SELECT id FROM jobs").fetchone()[0], "legacy")


if __name__ == "__main__":
    unittest.main()
