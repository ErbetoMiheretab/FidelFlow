"""Background worker that executes queued OCR jobs.

The API only enqueues jobs; a :class:`JobWorker` claims them from the SQLite
store and runs the OCR pipeline. Workers are started by the FastAPI lifespan,
so the service works out of the box, and the atomic claim in
:meth:`JobStore.claim_next_job` keeps multiple uvicorn workers from processing
the same job twice.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from amharic_ocr.config import OCRConfig
from amharic_ocr.pipeline import run_pipeline

from .job_store import JobStore

logger = logging.getLogger(__name__)

STALE_JOB_SECONDS = int(os.getenv("OCR_STALE_JOB_SECONDS", "1800"))
POLL_INTERVAL = float(os.getenv("OCR_WORKER_POLL_INTERVAL", "2.0"))


class JobInterrupted(Exception):
    """Raised inside a job when its worker is stopping or its job is gone."""


def output_file_paths(output_dir: Path | str, pdf_path: Path | str) -> dict[str, str]:
    """Return the output files produced by the pipeline for a job."""
    output_dir = Path(output_dir)
    base_name = Path(pdf_path).stem
    return {
        "text": str(output_dir / f"{base_name}_output.txt"),
        "raw": str(output_dir / f"{base_name}_raw.txt"),
        "json": str(output_dir / f"{base_name}_output.json"),
        "review": str(output_dir / f"{base_name}_review_report.txt"),
    }


def run_job(store: JobStore, job: dict, stop_event: threading.Event) -> None:
    """Run one claimed job and persist its progress and final state."""
    job_id = job["job_id"]
    output_dir = Path(job["output_dir"])
    pdf_path = Path(job["pdf_path"])
    config = OCRConfig(**job["config"])

    def on_page_results(
        page_results: list[dict], pages_done: int, pages_total: int
    ) -> None:
        if stop_event.is_set() or store.should_stop(job_id):
            raise JobInterrupted
        if not store.add_page_results(job_id, page_results, pages_done, pages_total):
            raise JobInterrupted

    try:
        run_pipeline(str(pdf_path), config, result_callback=on_page_results)
    except JobInterrupted:
        # Only requeue jobs that are still marked as processing. A cancelled
        # or deleted job must stay in its terminal state.
        current = store.get_job(job_id)
        if current is not None and current["status"] == "processing":
            store.requeue_job(job_id)
        return
    except Exception as e:
        logger.exception("OCR job %s failed", job_id)
        store.update_status(job_id, "failed", error=str(e))
        return

    store.set_files(job_id, output_file_paths(output_dir, pdf_path))
    store.update_status(job_id, "completed")


class JobWorker:
    """Single-thread job consumer that can be embedded in the API process."""

    def __init__(
        self,
        store: JobStore,
        poll_interval: float = POLL_INTERVAL,
        runner: Callable[[JobStore, dict, threading.Event], None] = run_job,
    ) -> None:
        self.store = store
        self.poll_interval = poll_interval
        self.runner = runner
        self.worker_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_job_id: str | None = None
        self._state_lock = threading.Lock()

    def start(self) -> None:
        self.store.requeue_stale_jobs(STALE_JOB_SECONDS)
        self._thread = threading.Thread(
            target=self._run,
            name=f"ocr-worker-{self.worker_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Started OCR worker %s", self.worker_id)

    def stop(self, timeout: float = 30.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "OCR worker %s did not stop within %.0fs",
                    self.worker_id,
                    timeout,
                )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            job = self.store.claim_next_job(self.worker_id)
            if job is None:
                self._stop_event.wait(self.poll_interval)
                continue

            with self._state_lock:
                self._current_job_id = job["job_id"]
            try:
                self.runner(self.store, job, self._stop_event)
            except Exception:
                logger.exception("Unhandled error in OCR worker")
                self.store.update_status(
                    job["job_id"], "failed", error="Worker crashed"
                )
            finally:
                with self._state_lock:
                    self._current_job_id = None
