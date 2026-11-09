"""Persistent SQLite-backed job storage for the OCR API.

The store keeps job metadata in ``jobs`` and per-page results in ``job_pages``.
Every operation opens a short-lived connection, so the store is safe to use
from multiple threads and from multiple uvicorn worker processes. Jobs are
claimed with ``BEGIN IMMEDIATE`` so two workers cannot process the same job.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    """SQLite storage for OCR jobs and their per-page results."""

    def __init__(self, db_path: str | Path = "results/jobs.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    pages_processed INTEGER NOT NULL DEFAULT 0,
                    pages_total INTEGER,
                    total_numerals INTEGER NOT NULL DEFAULT 0,
                    result_segments INTEGER NOT NULL DEFAULT 0,
                    files_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    worker_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                    ON jobs(status, created_at);

                CREATE TABLE IF NOT EXISTS job_pages (
                    job_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    page_key TEXT,
                    result_json TEXT NOT NULL,
                    PRIMARY KEY (job_id, seq),
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                );
                """
            )

    @staticmethod
    def _row_to_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        job = dict(row)
        job["config"] = json.loads(job.pop("config_json") or "{}")
        job["files"] = json.loads(job.pop("files_json") or "{}")
        return job

    # ------------------------------------------------------------------ jobs
    def create_job(
        self,
        *,
        job_id: str,
        filename: str,
        pdf_path: str,
        output_dir: str,
        config: dict[str, Any],
        pages_total: int | None,
        status: str = "queued",
    ) -> None:
        now = _utcnow()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, filename, pdf_path, output_dir,
                    config_json, pages_total, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    status,
                    filename,
                    str(pdf_path),
                    str(output_dir),
                    json.dumps(config, ensure_ascii=False, default=str),
                    pages_total,
                    now,
                    now,
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._row_to_job(row)

    def list_jobs(
        self, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM jobs"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([max(1, int(limit)), max(0, int(offset))])
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        jobs: list[dict[str, Any]] = []
        for row in rows:
            job = self._row_to_job(row)
            if job is not None:
                jobs.append(job)
        return jobs

    def update_status(
        self, job_id: str, status: str, error: str | None = None
    ) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? "
                "WHERE job_id = ?",
                (status, error, _utcnow(), job_id),
            )

    def set_files(self, job_id: str, files: dict[str, str]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE jobs SET files_json = ?, updated_at = ? WHERE job_id = ?",
                (json.dumps(files, ensure_ascii=False), _utcnow(), job_id),
            )

    def delete_job(self, job_id: str) -> bool:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            return cursor.rowcount == 1

    # ------------------------------------------------------------- execution
    def claim_next_job(self, worker_id: str) -> dict[str, Any] | None:
        """Atomically move the oldest queued job to ``processing``.

        ``BEGIN IMMEDIATE`` plus the conditional ``UPDATE`` guarantees that at
        most one worker can claim a given job, even across processes.
        """
        now = _utcnow()
        with closing(self._connect()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT job_id FROM jobs WHERE status = 'queued' "
                    "ORDER BY created_at LIMIT 1"
                ).fetchone()
                if row is None:
                    conn.rollback()
                    return None
                cursor = conn.execute(
                    "UPDATE jobs SET status = 'processing', worker_id = ?, "
                    "updated_at = ? WHERE job_id = ? AND status = 'queued'",
                    (worker_id, now, row["job_id"]),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    return None
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return self.get_job(row["job_id"])

    def should_stop(self, job_id: str) -> bool:
        """Return True when the job was deleted or cancelled."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return row is None or row["status"] == "cancelled"

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or processing job. Returns False if terminal."""
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = 'cancelled', updated_at = ? "
                "WHERE job_id = ? AND status IN ('queued', 'processing')",
                (_utcnow(), job_id),
            )
            return cursor.rowcount == 1

    def requeue_job(self, job_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE jobs SET status = 'queued', worker_id = NULL, "
                "updated_at = ? WHERE job_id = ? AND status = 'processing'",
                (_utcnow(), job_id),
            )

    def requeue_stale_jobs(self, stale_seconds: int) -> int:
        """Requeue jobs left in ``processing`` by a crashed/stopped worker."""
        cutoff = (
            datetime.now(UTC) - timedelta(seconds=max(1, stale_seconds))
        ).isoformat()
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = 'queued', worker_id = NULL, "
                "updated_at = ? WHERE status = 'processing' AND updated_at < ?",
                (_utcnow(), cutoff),
            )
            return cursor.rowcount

    # --------------------------------------------------------------- results
    def add_page_results(
        self,
        job_id: str,
        results: list[dict[str, Any]],
        pages_done: int,
        pages_total: int | None,
    ) -> bool:
        """Append page results and update progress.

        Returns False when the job no longer exists or was cancelled, which
        tells the worker to stop.
        """
        numerals = sum(len(r.get("numerals_found", [])) for r in results)
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None or row["status"] == "cancelled":
                return False

            start = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM job_pages WHERE job_id = ?",
                (job_id,),
            ).fetchone()[0]
            conn.executemany(
                "INSERT INTO job_pages (job_id, seq, page_key, result_json) "
                "VALUES (?, ?, ?, ?)",
                [
                    (
                        job_id,
                        start + index,
                        str(result.get("page")),
                        json.dumps(result, ensure_ascii=False, default=str),
                    )
                    for index, result in enumerate(results)
                ],
            )
            conn.execute(
                """
                UPDATE jobs SET
                    pages_processed = ?,
                    pages_total = COALESCE(?, pages_total),
                    total_numerals = total_numerals + ?,
                    result_segments = result_segments + ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    pages_done,
                    pages_total,
                    numerals,
                    len(results),
                    _utcnow(),
                    job_id,
                ),
            )
        return True

    def get_page_results(self, job_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT result_json FROM job_pages WHERE job_id = ? ORDER BY seq",
                (job_id,),
            ).fetchall()
        return [json.loads(row["result_json"]) for row in rows]
