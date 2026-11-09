"""
FastAPI server for Amharic OCR with Ethiopic numeral recognition.

This API provides endpoints to:
- Upload PDF files for OCR processing
- Retrieve OCR results (text, JSON, review reports)
- Check processing status
- Cancel and delete jobs
- Get system diagnostics
"""

import os
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from amharic_ocr.config import OCRConfig
from amharic_ocr.diagnostics import get_system_diagnostics
from amharic_ocr.io.pdf_reader import get_page_count

from .job_store import TERMINAL_STATUSES, JobStore
from .worker import JobWorker, run_job

# Directories for storing uploaded PDFs and results
UPLOAD_DIR = Path(os.getenv("OCR_UPLOAD_DIR", "uploads"))
RESULTS_DIR = Path(os.getenv("OCR_RESULTS_DIR", "results"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Persistent job storage shared by the API and the background worker
JOBS_DB = Path(os.getenv("OCR_JOBS_DB", str(RESULTS_DIR / "jobs.db")))
job_store = JobStore(JOBS_DB)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Start the job worker with the API and stop it on shutdown."""
    worker = JobWorker(job_store)
    worker.start()
    application.state.worker = worker
    try:
        yield
    finally:
        worker.stop()


app = FastAPI(
    title="Amharic OCR API",
    description="High-accuracy OCR service for Amharic text and Ethiopic numerals",
    version="0.1.0",
    lifespan=lifespan,
)


def _job_storage_paths(job_id: str, job: dict[str, Any]) -> tuple[Path, Path]:
    """Return the job's PDF and output paths, validated against the storage roots.

    Stored paths are never trusted blindly: a missing or malformed path must not
    resolve to the process working directory.
    """
    upload_root = UPLOAD_DIR.resolve()
    results_root = RESULTS_DIR.resolve()
    pdf_path = Path(job.get("pdf_path", UPLOAD_DIR / f"{job_id}.pdf")).resolve()
    output_dir = Path(job.get("output_dir", RESULTS_DIR / job_id)).resolve()

    if pdf_path.parent != upload_root or output_dir.parent != results_root:
        raise HTTPException(status_code=500, detail="Stored job paths are invalid")
    return pdf_path, output_dir


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Amharic OCR API",
        "version": "0.1.0",
        "description": "OCR service for Amharic text with Ethiopic numeral recognition",
        "endpoints": {
            "POST /ocr": "Upload PDF for OCR processing",
            "GET /ocr/{job_id}/status": "Get processing status",
            "GET /ocr/{job_id}/results": "Get page-level results summary",
            "GET /ocr/{job_id}/text": "Get plain text output",
            "GET /ocr/{job_id}/json": "Get structured JSON output",
            "GET /ocr/{job_id}/review": "Get quality review report",
            "POST /ocr/{job_id}/cancel": "Cancel a queued or running job",
            "DELETE /ocr/{job_id}": "Delete a job and its files",
            "GET /health": "Health check",
            "GET /diagnostics": "System diagnostics",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "amharic-ocr"}


@app.get("/diagnostics")
async def diagnostics():
    """Report environment and OCR engine/model capabilities."""
    try:
        diagnostics_output = get_system_diagnostics()
        return {"status": "success", "diagnostics": diagnostics_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostics failed: {str(e)}")


@app.post("/ocr")
async def process_pdf(
    file: UploadFile = File(...),
    dpi: int = 400,
    split_pages: bool = False,
    use_easyocr: bool = False,
    gpu: bool = False,
    min_char_threshold: int = 50,
    async_mode: bool = True,
):
    """
    Upload a PDF file for OCR processing.

    Args:
        file: PDF file to process
        dpi: Rendering resolution (default: 400)
        split_pages: Split double-page scans horizontally (default: False)
        use_easyocr: Enable EasyOCR as secondary engine (default: False)
        gpu: Enable GPU acceleration for EasyOCR (default: False)
        min_char_threshold: Minimum character count for quality flagging (default: 50)
        async_mode: Queue the job for the background worker (default: True)

    Returns:
        Job ID for tracking processing status
    """
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"
    try:
        content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create output directory for this job
    output_dir = RESULTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the page count up front so progress is meaningful before the first
    # page finishes. A failure here is not fatal; the pipeline will report it.
    try:
        pages_total: int | None = get_page_count(str(pdf_path))
    except Exception:  # noqa: BLE001
        pages_total = None

    config = OCRConfig(
        dpi=dpi,
        split_pages=split_pages,
        use_tesseract=True,
        use_easyocr=use_easyocr,
        easyocr_gpu=gpu,
        min_char_threshold=min_char_threshold,
        output_dir=str(output_dir),
    )

    job_store.create_job(
        job_id=job_id,
        filename=filename,
        pdf_path=str(pdf_path),
        output_dir=str(output_dir),
        config=asdict(config),
        pages_total=pages_total,
        status="queued" if async_mode else "processing",
    )

    if async_mode:
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "PDF uploaded successfully. Queued for processing.",
        }

    # Process synchronously off the event loop so the server stays responsive.
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=500, detail="Failed to create job")
    await run_in_threadpool(run_job, job_store, job, threading.Event())

    job = job_store.get_job(job_id) or {}
    status = job.get("status", "failed")
    return {
        "job_id": job_id,
        "status": status,
        "message": (
            "OCR processing completed successfully"
            if status == "completed"
            else f"Processing failed: {job.get('error')}"
        ),
    }


@app.get("/ocr/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the status and progress of an OCR job."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    pages_processed = job["pages_processed"]
    pages_total = job["pages_total"]
    result_segments = job["result_segments"]

    response: dict[str, Any] = {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
        "pages_processed": pages_processed,
        "pages_total": pages_total,
        "result_segments": result_segments,
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
    if pages_total:
        response["progress_percent"] = round(
            min(pages_processed, pages_total) / pages_total * 100, 1
        )

    if job["status"] == "completed":
        response["files_available"] = list(job["files"].keys())
        response["total_pages"] = pages_total or result_segments
        response["total_numerals"] = job["total_numerals"]
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")

    return response


@app.get("/ocr/{job_id}/text")
async def get_text_output(job_id: str):
    """Get the plain text output of an OCR job."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job status: {job['status']}. Results not available yet.",
        )

    text_file = job["files"].get("text")
    if not text_file or not os.path.exists(text_file):
        raise HTTPException(status_code=404, detail="Text output file not found")

    return FileResponse(
        text_file, media_type="text/plain", filename=f"{job['filename']}_output.txt"
    )


@app.get("/ocr/{job_id}/json")
async def get_json_output(job_id: str):
    """Get the structured JSON output of an OCR job."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job status: {job['status']}. Results not available yet.",
        )

    json_file = job["files"].get("json")
    if not json_file or not os.path.exists(json_file):
        raise HTTPException(status_code=404, detail="JSON output file not found")

    return FileResponse(
        json_file,
        media_type="application/json",
        filename=f"{job['filename']}_output.json",
    )


@app.get("/ocr/{job_id}/review")
async def get_review_report(job_id: str):
    """Get the quality review report of an OCR job."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job status: {job['status']}. Results not available yet.",
        )

    review_file = job["files"].get("review")
    if not review_file or not os.path.exists(review_file):
        raise HTTPException(status_code=404, detail="Review report file not found")

    return FileResponse(
        review_file,
        media_type="text/plain",
        filename=f"{job['filename']}_review_report.txt",
    )


@app.get("/ocr/{job_id}/results")
async def get_results_summary(job_id: str, include_partial: bool = False):
    """
    Get a summary of OCR results with page-level details.

    Args:
        job_id: Job identifier
        include_partial: If True, return partial results even if the job is
            still queued or processing.
    """
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Allow partial results if requested
    if not include_partial and job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Job status: {job['status']}. Results not available yet. "
                "Use ?include_partial=true to get partial results."
            ),
        )

    results = job_store.get_page_results(job_id)
    pages_total = job["pages_total"]
    return JSONResponse(
        content={
            "job_id": job_id,
            "filename": job["filename"],
            "status": job["status"],
            "total_pages": pages_total if pages_total is not None else len(results),
            "pages_processed": job["pages_processed"],
            "result_segments": len(results),
            "pages": [
                {
                    "page": r["page"],
                    "text_length": len(r["text"]),
                    "confidence": r["confidence"],
                    "engine": r["engine_name"],
                    "numerals_found": r["numerals_found"],
                    "flagged": r["flagged"],
                    "review_reasons": r["review_reasons"],
                }
                for r in results
            ],
        }
    )


@app.post("/ocr/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued or running job.

    A running job stops after the page currently being processed.
    """
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409, detail=f"Job is already {job['status']}"
        )
    if not job_store.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job could not be cancelled")

    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancelled. A page already being processed may finish first.",
    }


@app.delete("/ocr/{job_id}")
async def delete_job(job_id: str, force: bool = False):
    """
    Delete a job, its uploaded PDF and its output directory.

    A job that is queued or still processing is rejected with 409 unless
    ``force=true`` is passed. Forced deletion removes the job from the store
    immediately while the worker finishes; its late result is discarded.
    """
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] in {"queued", "processing"} and not force:
        raise HTTPException(
            status_code=409,
            detail=(
                "Job is still queued or processing. Wait for it to finish, or "
                "pass ?force=true to delete it while it runs."
            ),
        )

    pdf_path, output_dir = _job_storage_paths(job_id, job)

    try:
        if pdf_path.is_file():
            pdf_path.unlink()
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        elif output_dir.exists():
            output_dir.unlink()
    except OSError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete job files: {e}"
        ) from e

    job_store.delete_job(job_id)

    return {"message": f"Job {job_id} deleted successfully"}
