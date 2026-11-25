"""Standalone SQLite job state; this module never stores delivery targets."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class RequestConflict(ValueError):
    """A request ID was replayed with a different payload."""


class SelectionConflict(ValueError):
    """A selection retry conflicts with the already tracked choice/state."""


class QueueLimitError(RuntimeError):
    def __init__(self, limit: int):
        self.limit = int(limit)
        super().__init__(f"job queue limit is {self.limit}")


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc, traceback):
        try:
            return super().__exit__(exc_type, exc, traceback)
        finally:
            self.close()


class JobStore:
    """Durable jobs and resolved-stem aliases, owned by one service process."""

    _ACTIVE = (
        "queued", "resolving", "selection_required", "downloading", "converting",
        "separating", "splitting", "cleaning", "voice_converting", "mixing", "paused",
    )
    _RUNNING = (
        "resolving", "downloading", "converting", "separating", "splitting",
        "cleaning", "voice_converting", "mixing", "paused",
    )

    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, factory=_ClosingConnection
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            tables = {
                str(row[0]) for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "jobs" in tables:
                columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(jobs)")}
                if "request_id" not in columns or "selected_candidate_id" not in columns:
                    raise RuntimeError(
                        "This database is not a standalone music runtime database; "
                        "choose a fresh MUSIC_RUNTIME_DATA_DIR."
                    )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    dedup_key TEXT,
                    query TEXT NOT NULL,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'queued',
                    stage TEXT NOT NULL DEFAULT '',
                    progress TEXT NOT NULL DEFAULT 'queued',
                    song_json TEXT NOT NULL DEFAULT '{}',
                    candidates_json TEXT NOT NULL DEFAULT '[]',
                    selected_candidate_id TEXT NOT NULL DEFAULT '',
                    error_code TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    artifacts_json TEXT NOT NULL DEFAULT '{}',
                    fingerprints_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS api_requests (
                    request_id TEXT PRIMARY KEY,
                    payload_query TEXT NOT NULL,
                    job_id TEXT NOT NULL REFERENCES jobs(id),
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_aliases (
                    alias_job_id TEXT PRIMARY KEY REFERENCES jobs(id),
                    canonical_job_id TEXT NOT NULL REFERENCES jobs(id),
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stage_cache (
                    stage TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    outputs_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(stage, fingerprint)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS jobs_dedup_key_unique
                    ON jobs(dedup_key) WHERE dedup_key IS NOT NULL;
                CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs(created_at DESC);
                PRAGMA user_version=1;
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for key in ("params_json", "song_json", "candidates_json", "artifacts_json", "fingerprints_json"):
            raw = result.pop(key, "{}")
            try:
                result[key.removesuffix("_json")] = json.loads(raw)
            except (TypeError, ValueError):
                result[key.removesuffix("_json")] = [] if key == "candidates_json" else {}
        return result

    @classmethod
    def _resolve_conn(cls, conn: sqlite3.Connection, job_id: str) -> str | None:
        current = str(job_id)
        if conn.execute("SELECT 1 FROM jobs WHERE id=?", (current,)).fetchone() is None:
            return None
        seen: set[str] = set()
        for _ in range(256):
            if current in seen:
                raise ValueError("job alias cycle detected")
            seen.add(current)
            row = conn.execute(
                "SELECT canonical_job_id FROM job_aliases WHERE alias_job_id=?", (current,)
            ).fetchone()
            if row is None:
                return current
            current = str(row["canonical_job_id"])
        raise ValueError("job alias chain is too deep")

    @classmethod
    def _job_conn(cls, conn: sqlite3.Connection, job_id: str) -> dict[str, Any] | None:
        return cls._decode(conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    @classmethod
    def _view_conn(cls, conn: sqlite3.Connection, requested_id: str) -> dict[str, Any] | None:
        requested = cls._job_conn(conn, requested_id)
        if requested is None:
            return None
        canonical_id = cls._resolve_conn(conn, requested_id)
        if canonical_id is None:
            return None
        canonical = cls._job_conn(conn, canonical_id)
        if canonical is None:
            return None
        canonical["id"] = requested_id
        canonical["request_id"] = requested["request_id"]
        canonical["query"] = requested["query"]
        canonical["canonical_id"] = canonical_id
        return canonical

    def create_request_job(
        self,
        request_id: str,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        queue_limit: int = 5,
    ) -> tuple[dict[str, Any], bool]:
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute(
                "SELECT payload_query,job_id FROM api_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if old is not None:
                if old["payload_query"] != query:
                    raise RequestConflict("request_id is already bound to a different payload")
                existing = self._view_conn(conn, str(old["job_id"]))
                if existing is None:
                    raise RuntimeError("request record lost its job")
                return existing, False

            placeholders = ",".join("?" for _ in self._ACTIVE)
            active = conn.execute(
                f"SELECT COUNT(*) FROM jobs WHERE status IN ({placeholders}) "
                "AND NOT EXISTS (SELECT 1 FROM job_aliases WHERE alias_job_id=jobs.id)",
                self._ACTIVE,
            ).fetchone()[0]
            if int(active) >= max(1, int(queue_limit)):
                raise QueueLimitError(queue_limit)

            job_id = uuid.uuid4().hex[:12]
            while conn.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
                job_id = uuid.uuid4().hex[:12]
            conn.execute(
                "INSERT INTO jobs(id,request_id,query,params_json,status,stage,progress,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (job_id, request_id, query, _json(params or {}), "queued", "queued", "Queued", now, now),
            )
            conn.execute(
                "INSERT INTO api_requests(request_id,payload_query,job_id,created_at) VALUES(?,?,?,?)",
                (request_id, query, job_id, now),
            )
            return self._view_conn(conn, job_id), True

    def lookup_request(self, request_id: str, query: str) -> dict[str, Any] | None:
        """Look up a durable idempotency key before readiness gating a new request."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT payload_query,job_id FROM api_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                return None
            if row["payload_query"] != query:
                raise RequestConflict("request_id is already bound to a different payload")
            return self._view_conn(conn, str(row["job_id"]))

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return self._job_conn(conn, job_id)

    def get_job_view(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return self._view_conn(conn, job_id)

    def get_canonical_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            canonical = self._resolve_conn(conn, job_id)
            return self._job_conn(conn, canonical) if canonical else None

    def resolve_job_id(self, job_id: str) -> str | None:
        with self._connect() as conn:
            return self._resolve_conn(conn, job_id)

    def list_jobs(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        limit = min(max(1, int(limit)), 500)
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT id FROM jobs WHERE status=? AND NOT EXISTS "
                    "(SELECT 1 FROM job_aliases WHERE alias_job_id=jobs.id) "
                    "ORDER BY created_at DESC LIMIT ?", (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id FROM jobs WHERE NOT EXISTS "
                    "(SELECT 1 FROM job_aliases WHERE alias_job_id=jobs.id) "
                    "ORDER BY created_at DESC LIMIT ?", (limit,),
                ).fetchall()
            return [self._job_conn(conn, str(row["id"])) for row in rows]

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        song: dict[str, Any] | None = None,
        candidates: list[dict[str, Any]] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {"updated_at": time.time()}
        for key, value in (("status", status), ("stage", stage), ("progress", progress),
                           ("error_code", error_code), ("error_message", error_message)):
            if value is not None:
                values[key] = value
        for key, value in (("song_json", song), ("candidates_json", candidates), ("params_json", params)):
            if value is not None:
                values[key] = _json(value)
        if status in {"failed", "cancelled"}:
            values["dedup_key"] = None
        assignments = ",".join(f"{key}=?" for key in values)
        with self._connect() as conn:
            cur = conn.execute(f"UPDATE jobs SET {assignments} WHERE id=?", (*values.values(), job_id))
            if cur.rowcount == 0:
                raise KeyError(job_id)
            result = self._job_conn(conn, job_id)
            assert result is not None
            return result

    def select_candidate(self, job_id: str, candidate_id: str) -> tuple[dict[str, Any], bool]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            requested = self._job_conn(conn, job_id)
            if requested is None:
                raise KeyError(job_id)
            selected = str(requested.get("selected_candidate_id") or "")
            if selected:
                if selected != candidate_id:
                    raise SelectionConflict("a different candidate was already selected")
                view = self._view_conn(conn, job_id)
                assert view is not None
                return view, False
            canonical_id = self._resolve_conn(conn, job_id)
            if canonical_id is None:
                raise KeyError(job_id)
            row = self._job_conn(conn, canonical_id)
            assert row is not None
            if row["status"] != "selection_required":
                raise SelectionConflict("job is not awaiting a song choice")
            candidate = next(
                (item for item in row["candidates"] if str(item.get("id", "")) == candidate_id), None
            )
            if candidate is None:
                raise SelectionConflict("candidate_id is not a current search result")
            song = {key: candidate.get(key, "") for key in ("id", "title", "artist", "duration_ms")}
            now = time.time()
            conn.execute(
                "UPDATE jobs SET selected_candidate_id=?,song_json=?,candidates_json='[]',status='queued',"
                "stage='queued',progress='Choice accepted',error_code='',error_message='',updated_at=? WHERE id=?",
                (candidate_id, _json(song), now, canonical_id),
            )
            view = self._view_conn(conn, job_id)
            assert view is not None
            return view, True

    def bind_dedup_key(self, job_id: str, dedup_key: str) -> str:
        """Merge only requests with a resolved stem/profile fingerprint."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            source_id = self._resolve_conn(conn, job_id)
            if source_id is None:
                raise KeyError(job_id)
            collision = conn.execute(
                "SELECT id FROM jobs WHERE dedup_key=? AND id<>? AND status NOT IN ('failed','cancelled')",
                (dedup_key, source_id),
            ).fetchone()
            if collision:
                target_id = self._resolve_conn(conn, str(collision["id"]))
                if target_id is None:
                    raise KeyError(collision["id"])
                if target_id != source_id:
                    now = time.time()
                    conn.execute(
                        "INSERT OR IGNORE INTO job_aliases(alias_job_id,canonical_job_id,created_at) VALUES(?,?,?)",
                        (source_id, target_id, now),
                    )
                    conn.execute(
                        "UPDATE job_aliases SET canonical_job_id=? WHERE canonical_job_id=?",
                        (target_id, source_id),
                    )
                    conn.execute(
                        "UPDATE jobs SET status='cancelled',stage='deduplicated',progress='Using matching active or completed job',"
                        "dedup_key=NULL,error_code='',error_message='',updated_at=? WHERE id=?",
                        (now, source_id),
                    )
                    return target_id
                return source_id
            conn.execute("UPDATE jobs SET dedup_key=?,updated_at=? WHERE id=?", (dedup_key, time.time(), source_id))
            return source_id

    def find_dedup_key(self, dedup_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE dedup_key=? AND status NOT IN ('failed','cancelled')",
                (dedup_key,),
            ).fetchone()
            return self._job_conn(conn, str(row["id"])) if row else None

    def invalidate_reusable_job(self, job_id: str, reason: str = "completed artifact is missing") -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='failed',dedup_key=NULL,stage='artifact_invalid',"
                "error_code='artifact_invalid',error_message=?,updated_at=? WHERE id=? AND status='ready'",
                (reason[:300], time.time(), job_id),
            )

    def record_artifact(self, job_id: str, stage: str, fingerprint: str, outputs: dict[str, str]) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT artifacts_json,fingerprints_json FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            artifacts = json.loads(row["artifacts_json"] or "{}")
            fingerprints = json.loads(row["fingerprints_json"] or "{}")
            artifacts[stage] = dict(outputs)
            fingerprints[stage] = fingerprint
            now = time.time()
            conn.execute(
                "UPDATE jobs SET artifacts_json=?,fingerprints_json=?,updated_at=? WHERE id=?",
                (_json(artifacts), _json(fingerprints), now, job_id),
            )
            conn.execute(
                "INSERT INTO stage_cache(stage,fingerprint,outputs_json,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(stage,fingerprint) DO UPDATE SET outputs_json=excluded.outputs_json,updated_at=excluded.updated_at",
                (stage, fingerprint, _json(outputs), now),
            )

    @staticmethod
    def _valid_outputs(value: Any) -> dict[str, str] | None:
        if not isinstance(value, dict) or not value:
            return None
        result = {str(k): str(v) for k, v in value.items()}
        for raw in result.values():
            try:
                if not Path(raw).is_file() or Path(raw).stat().st_size <= 0:
                    return None
            except OSError:
                return None
        return result

    def cached_artifact(self, job_id: str, stage: str, fingerprint: str) -> dict[str, str] | None:
        job = self.get_job(job_id)
        if not job or job["fingerprints"].get(stage) != fingerprint:
            return None
        return self._valid_outputs(job["artifacts"].get(stage))

    def cached_fingerprint(self, stage: str, fingerprint: str) -> dict[str, str] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT outputs_json FROM stage_cache WHERE stage=? AND fingerprint=?", (stage, fingerprint)
            ).fetchone()
        if row is None:
            return None
        try:
            return self._valid_outputs(json.loads(row["outputs_json"]))
        except (TypeError, ValueError):
            return None

    def recover_after_restart(self) -> None:
        """Resume interrupted production stages; artifacts remain for cache validation."""
        placeholders = ",".join("?" for _ in self._RUNNING)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET status='queued',stage='queued',progress='Resuming after restart',updated_at=? "
                f"WHERE status IN ({placeholders})",
                (time.time(), *self._RUNNING),
            )
