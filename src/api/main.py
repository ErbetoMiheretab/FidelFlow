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
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import cv2
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from amharic_ocr.config import OCRConfig
from amharic_ocr.diagnostics import get_system_diagnostics
from amharic_ocr.io.pdf_reader import get_page_count, render_page_to_numpy

from .job_store import TERMINAL_STATUSES, JobStore
from .worker import JobWorker, output_file_paths, run_job

# Directories for storing uploaded PDFs and results
UPLOAD_DIR = Path(os.getenv("OCR_UPLOAD_DIR", "uploads"))
RESULTS_DIR = Path(os.getenv("OCR_RESULTS_DIR", "results"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Persistent job storage shared by the API and the background worker
JOBS_DB = Path(os.getenv("OCR_JOBS_DB", str(RESULTS_DIR / "jobs.db")))
job_store = JobStore(JOBS_DB)

# Upload limits. Both are configurable for deployments with larger books.
MAX_UPLOAD_BYTES = int(os.getenv("OCR_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "2000"))
UPLOAD_CHUNK_SIZE = 1024 * 1024

JobStatus = Literal["queued", "processing", "completed", "failed", "cancelled"]


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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a consistent error shape for invalid requests."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": jsonable_encoder(exc.errors()),
        },
    )


def _results_not_ready(job: dict[str, Any]) -> HTTPException:
    """Build a 409 error for output requested before a job is completed."""
    detail = f"Job status: {job['status']}. Results not available yet."
    if job.get("error"):
        detail += f" Error: {job['error']}"
    return HTTPException(status_code=409, detail=detail)


def _page_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Return the compact representation used by result listings."""
    return {
        "page": result["page"],
        "text_length": len(result.get("text", "")),
        "raw_text_length": len(result.get("raw_text", "")),
        "confidence": result.get("confidence", 0.0),
        "engine": result.get("engine_name", "unknown"),
        "numeral_heavy": bool(result.get("numeral_heavy", False)),
        "numerals_found": result.get("numerals_found", []),
        "flagged": bool(result.get("flagged", False)),
        "review_reasons": result.get("review_reasons", []),
        "warnings": result.get("warnings", []),
    }


def _page_number(page_id: str) -> int | None:
    """Extract the PDF page number from a page id such as ``12`` or ``12a``."""
    match = re.match(r"^(\d+)", page_id)
    return int(match.group(1)) if match else None


def _build_archive(output_dir: Path) -> Path:
    """Create a temporary ZIP containing every file in an output directory."""
    fd, name = tempfile.mkstemp(prefix="ocr-", suffix=".zip")
    os.close(fd)
    archive_path = Path(name)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir))
    return archive_path


def _safe_filename(filename: str | None) -> str:
    """Return a display filename with path components and control chars removed."""
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = "".join(char for char in name if char >= " " and char != "\x7f")
    return name[:255] or "upload.pdf"


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
            "GET /ocr": "List jobs with filtering and pagination",
            "GET /ocr/{job_id}/status": "Get processing status",
            "GET /ocr/{job_id}/results": "Get page-level results summary",
            "GET /ocr/{job_id}/pages": "List page results",
            "GET /ocr/{job_id}/pages/{page_id}": "Get one full page result",
            "GET /ocr/{job_id}/pages/{page_id}/image": "Render one page image",
            "GET /ocr/{job_id}/text": "Get plain text output",
            "GET /ocr/{job_id}/raw": "Get raw pre-correction text",
            "GET /ocr/{job_id}/json": "Get structured JSON output",
            "GET /ocr/{job_id}/review": "Get quality review report",
            "GET /ocr/{job_id}/archive": "Download all outputs as a ZIP",
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
        raise HTTPException(status_code=500, detail=f"Diagnostics failed: {e!s}")


@app.post(
    "/ocr",
    responses={
        202: {"description": "Job queued for background processing"},
        400: {"description": "Invalid or unreadable PDF"},
        413: {"description": "Uploaded file too large"},
    },
)
async def process_pdf(
    file: UploadFile = File(...),
    dpi: int = Form(400, ge=72, le=1200),
    split_pages: bool = Form(False),
    use_easyocr: bool = Form(False),
    gpu: bool = Form(False),
    min_char_threshold: int = Form(50, ge=0, le=1_000_000),
    async_mode: bool = Form(True),
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
    filename = _safe_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Stream the upload to disk with a hard size limit so a large request
    # cannot exhaust memory. Keep the first kilobyte for a signature check.
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"
    size = 0
    header = b""
    try:
        with open(pdf_path, "wb") as f:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Uploaded file exceeds the {MAX_UPLOAD_BYTES} byte "
                            "limit (OCR_MAX_UPLOAD_BYTES)."
                        ),
                    )
                if len(header) < 1024:
                    header += chunk[: 1024 - len(header)]
                f.write(chunk)
    except HTTPException:
        pdf_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to save file: {e!s}"
        ) from e

    if size == 0:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if b"%PDF-" not in header:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    # Read the page count up front so progress is meaningful before the first
    # page finishes. Reject unreadable or unreasonably large documents now
    # instead of queuing a job that cannot succeed.
    try:
        pages_total = get_page_count(str(pdf_path))
    except Exception as e:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail=f"Could not read PDF: {e!s}"
        ) from e

    if pages_total < 1:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="PDF contains no pages")
    if pages_total > MAX_PAGES:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"PDF has {pages_total} pages; the maximum is {MAX_PAGES} "
                "(OCR_MAX_PAGES)."
            ),
        )

    # Create output directory for this job
    output_dir = RESULTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

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
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "queued",
                "message": "PDF uploaded successfully. Queued for processing.",
            },
        )

    # Process synchronously off the event loop so the server stays responsive.
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=500, detail="Failed to create job")
    await run_in_threadpool(run_job, job_store, job, threading.Event())

    job = job_store.get_job(job_id) or {}
    status = job.get("status", "failed")
    if status == "cancelled":
        raise HTTPException(
            status_code=409, detail=f"Job {job_id} was cancelled"
        )
    if status != "completed":
        raise HTTPException(
            status_code=500,
            detail=(
                f"OCR processing failed for job {job_id}: "
                f"{job.get('error', 'unknown error')}"
            ),
        )
    return {
        "job_id": job_id,
        "status": status,
        "message": "OCR processing completed successfully",
    }


@app.get("/ocr")
async def list_jobs(
    status: JobStatus | None = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List jobs, newest first, with progress and result counts."""
    jobs = job_store.list_jobs(status=status, limit=limit, offset=offset)
    total = job_store.count_jobs(status=status)

    items: list[dict[str, Any]] = []
    for job in jobs:
        item: dict[str, Any] = {
            "job_id": job["job_id"],
            "status": job["status"],
            "filename": job["filename"],
            "pages_processed": job["pages_processed"],
            "pages_total": job["pages_total"],
            "result_segments": job["result_segments"],
            "total_numerals": job["total_numerals"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
        }
        if job["pages_total"]:
            item["progress_percent"] = round(
                min(job["pages_processed"], job["pages_total"])
                / job["pages_total"]
                * 100,
                1,
            )
        if job.get("error"):
            item["error"] = job["error"]
        items.append(item)

    return {
        "jobs": items,
        "total": total,
        "limit": limit,
        "offset": offset,
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
        raise _results_not_ready(job)

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
        raise _results_not_ready(job)

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
        raise _results_not_ready(job)

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
        error = _results_not_ready(job)
        error.detail += " Use ?include_partial=true to get partial results."
        raise error

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
            "pages": [_page_summary(result) for result in results],
        }
    )


@app.get("/ocr/{job_id}/raw")
async def get_raw_output(job_id: str):
    """Get the raw OCR text before corrections."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise _results_not_ready(job)

    raw_file = job["files"].get("raw") or output_file_paths(
        job["output_dir"], job["pdf_path"]
    )["raw"]
    if not os.path.exists(raw_file):
        raise HTTPException(status_code=404, detail="Raw output file not found")

    return FileResponse(
        raw_file, media_type="text/plain", filename=f"{job['filename']}_raw.txt"
    )


@app.get("/ocr/{job_id}/pages")
async def list_pages(
    job_id: str,
    include_partial: bool = False,
    flagged: bool | None = None,
    numeral_heavy: bool | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List page-level results with optional filtering and pagination."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not include_partial and job["status"] != "completed":
        raise _results_not_ready(job)

    results = job_store.get_page_results(job_id)
    filtered = [
        result
        for result in results
        if (flagged is None or bool(result.get("flagged")) == flagged)
        and (
            numeral_heavy is None
            or bool(result.get("numeral_heavy")) == numeral_heavy
        )
    ]
    window = filtered[offset : offset + limit]

    return {
        "job_id": job_id,
        "status": job["status"],
        "total": len(filtered),
        "result_segments": len(results),
        "limit": limit,
        "offset": offset,
        "pages": [_page_summary(result) for result in window],
    }


@app.get("/ocr/{job_id}/pages/{page_id}")
async def get_page(job_id: str, page_id: str):
    """Get the full result for a single page, including diagnostics."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job_store.get_page_result(job_id, page_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Page {page_id!r} not found for this job"
        )

    return {"job_id": job_id, "status": job["status"], "page": result}


@app.get("/ocr/{job_id}/pages/{page_id}/image")
async def get_page_image(job_id: str, page_id: str):
    """Render the source PDF page for a page id (``12`` or split ``12a``/``12b``)."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    page_number = _page_number(page_id)
    if page_number is None or page_number < 1:
        raise HTTPException(
            status_code=404, detail=f"Page {page_id!r} has no PDF page number"
        )
    if job["pages_total"] and page_number > job["pages_total"]:
        raise HTTPException(status_code=404, detail="Page out of range")

    pdf_path = Path(job["pdf_path"])
    if not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="Source PDF not found")

    dpi = int(job["config"].get("dpi", 400))
    try:
        image = await run_in_threadpool(
            render_page_to_numpy, str(pdf_path), page_number, dpi
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Could not render page image: {e!s}"
        ) from e

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode page image")

    return Response(content=encoded.tobytes(), media_type="image/png")


@app.get("/ocr/{job_id}/archive")
async def get_archive(job_id: str):
    """Download every output file for a completed job as a ZIP archive."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise _results_not_ready(job)

    _, output_dir = _job_storage_paths(job_id, job)
    if not output_dir.is_dir():
        raise HTTPException(status_code=404, detail="Output directory not found")

    try:
        archive_path = await run_in_threadpool(_build_archive, output_dir)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Could not create archive: {e!s}"
        ) from e

    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=f"{Path(job['filename']).stem}_ocr.zip",
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
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

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "cancelled",
            "message": (
                "Job cancelled. A page already being processed may finish first."
            ),
        },
    )


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
