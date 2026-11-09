"""
FastAPI server for Amharic OCR with Ethiopic numeral recognition.

This API provides endpoints to:
- Upload PDF files for OCR processing
- Retrieve OCR results (text, JSON, review reports)
- Check processing status
- Get system diagnostics
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from amharic_ocr.config import OCRConfig
from amharic_ocr.diagnostics import get_system_diagnostics
from amharic_ocr.io.pdf_reader import get_page_count
from amharic_ocr.pipeline import run_pipeline

app = FastAPI(
    title="Amharic OCR API",
    description="High-accuracy OCR service for Amharic text and Ethiopic numerals",
    version="0.1.0",
)

# Directory for storing uploaded PDFs and results
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory storage for job status (could be replaced with Redis/database)
job_store: dict[str, dict[str, Any]] = {}


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
            "GET /ocr/{job_id}/text": "Get plain text output",
            "GET /ocr/{job_id}/json": "Get structured JSON output",
            "GET /ocr/{job_id}/review": "Get quality review report",
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
    background_tasks: BackgroundTasks,
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
        async_mode: Process asynchronously in background (default: True)

    Returns:
        Job ID for tracking processing status
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"
    try:
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create output directory for this job
    output_dir = RESULTS_DIR / job_id
    output_dir.mkdir(exist_ok=True)

    # Read the page count up front so progress is meaningful before the first
    # page finishes. A failure here is not fatal; the pipeline will report it.
    try:
        pages_total: int | None = get_page_count(str(pdf_path))
    except Exception:  # noqa: BLE001
        pages_total = None

    # Initialize job status
    job_store[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "pdf_path": str(pdf_path),
        "output_dir": str(output_dir),
        "results": [],  # Filled incrementally so partial results are available
        "pages_processed": 0,
        "pages_total": pages_total,
        "config": {
            "dpi": dpi,
            "split_pages": split_pages,
            "use_easyocr": use_easyocr,
            "gpu": gpu,
            "min_char_threshold": min_char_threshold,
        },
    }

    # Configure OCR
    config = OCRConfig(
        dpi=dpi,
        split_pages=split_pages,
        use_tesseract=True,
        use_easyocr=use_easyocr,
        easyocr_gpu=gpu,
        min_char_threshold=min_char_threshold,
        output_dir=str(output_dir),
    )

    def on_page_results(
        page_results: list[dict[str, Any]], pages_done: int, pages_available: int
    ):
        """Publish page results as soon as that page has been processed."""
        job = job_store.get(job_id)
        if job is None:
            return
        job["results"].extend(page_results)
        job["pages_processed"] = pages_done
        job["pages_total"] = pages_available

    def process_job():
        """Background task to process the PDF."""
        try:
            # Run OCR pipeline
            results = run_pipeline(
                str(pdf_path), config, result_callback=on_page_results
            )
        except Exception as e:
            # The job may have been force-deleted while it was processing.
            if job_id in job_store:
                job_store[job_id]["status"] = "failed"
                job_store[job_id]["error"] = str(e)
            return

        if job_id not in job_store:
            return

        # Update job status
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["results"] = results
        job_store[job_id]["pages_processed"] = job_store[job_id].get(
            "pages_total"
        ) or len(results)

        # Get output file paths - pipeline uses the PDF basename (job_id) not original filename
        pdf_base_name = Path(pdf_path).stem  # This will be the job_id
        job_store[job_id]["files"] = {
            "text": str(output_dir / f"{pdf_base_name}_output.txt"),
            "json": str(output_dir / f"{pdf_base_name}_output.json"),
            "review": str(output_dir / f"{pdf_base_name}_review_report.txt"),
        }

    if async_mode:
        # Process in background and return immediately
        background_tasks.add_task(process_job)
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "PDF uploaded successfully. Processing in background.",
        }
    else:
        # Process synchronously (blocking)
        process_job()
        return {
            "job_id": job_id,
            "status": job_store[job_id]["status"],
            "message": "OCR processing completed successfully" if job_store[job_id]["status"] == "completed" else f"Processing failed: {job_store[job_id].get('error')}",
        }


@app.get("/ocr/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the status of an OCR job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]
    results = job.get("results", [])
    pages_processed = job.get("pages_processed", 0)
    pages_total = job.get("pages_total")

    response: dict[str, Any] = {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
        "pages_processed": pages_processed,
        "pages_total": pages_total,
        "result_segments": len(results),
    }
    if pages_total:
        response["progress_percent"] = round(
            min(pages_processed, pages_total) / pages_total * 100, 1
        )

    if job["status"] == "completed":
        response["files_available"] = list(job.get("files", {}).keys())
        response["total_pages"] = pages_total or len(results)
        response["total_numerals"] = sum(
            len(r.get("numerals_found", [])) for r in results
        )
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")

    return response


@app.get("/ocr/{job_id}/text")
async def get_text_output(job_id: str):
    """Get the plain text output of an OCR job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Job status: {job['status']}. Results not available yet."
        )

    text_file = job["files"]["text"]
    if not os.path.exists(text_file):
        raise HTTPException(status_code=404, detail="Text output file not found")

    return FileResponse(
        text_file, media_type="text/plain", filename=f"{job['filename']}_output.txt"
    )


@app.get("/ocr/{job_id}/json")
async def get_json_output(job_id: str):
    """Get the structured JSON output of an OCR job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Job status: {job['status']}. Results not available yet."
        )

    json_file = job["files"]["json"]
    if not os.path.exists(json_file):
        raise HTTPException(status_code=404, detail="JSON output file not found")

    return FileResponse(
        json_file,
        media_type="application/json",
        filename=f"{job['filename']}_output.json",
    )


@app.get("/ocr/{job_id}/review")
async def get_review_report(job_id: str):
    """Get the quality review report of an OCR job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Job status: {job['status']}. Results not available yet."
        )

    review_file = job["files"]["review"]
    if not os.path.exists(review_file):
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
        include_partial: If True, return partial results even if job is still processing
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]
    
    # Allow partial results if requested
    if not include_partial and job["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Job status: {job['status']}. Results not available yet. Use ?include_partial=true to get partial results."
        )

    results = job.get("results", [])
    pages_total = job.get("pages_total")
    pages_processed = job.get("pages_processed", 0)
    return JSONResponse(
        content={
            "job_id": job_id,
            "filename": job["filename"],
            "status": job["status"],
            "total_pages": pages_total if pages_total is not None else len(results),
            "pages_processed": pages_processed,
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


@app.delete("/ocr/{job_id}")
async def delete_job(job_id: str, force: bool = False):
    """
    Delete a job, its uploaded PDF and its output directory.

    A job that is still processing is rejected with 409 unless ``force=true`` is
    passed. Forced deletion removes the job from the store immediately while the
    background task finishes; its late result is discarded.
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]
    if job.get("status") == "processing" and not force:
        raise HTTPException(
            status_code=409,
            detail=(
                "Job is still processing. Wait for it to finish, or pass "
                "?force=true to delete it while it runs."
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

    # Remove from job store
    del job_store[job_id]

    return {"message": f"Job {job_id} deleted successfully"}
