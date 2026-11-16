# Amharic OCR API

FastAPI server providing REST endpoints for Amharic OCR with Ethiopic numeral recognition.

## Features

- **PDF Upload & Processing**: Upload PDF files for OCR processing
- **Multiple Output Formats**: Get results as plain text, structured JSON, or quality review reports
- **Job Tracking**: Persistent job state, live progress and partial results
- **Job Control**: Cancel, retry-safe inspection and delete jobs
- **System Diagnostics**: Verify OCR dependencies and system setup
- **Configurable Processing**: Adjust DPI, page splitting, and engine options

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .
```

### Running the Server

#### Option 1: Using the command-line script
```bash
amharic-ocr-server
```

#### Option 2: Using Python directly
```bash
python -m api.server
```

#### Option 3: Using uvicorn directly
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8023 --reload
```

The API will be available at `http://localhost:8023`

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8023/docs
- **ReDoc**: http://localhost:8023/redoc

## API Endpoints

### 1. Root Endpoint
```http
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```http
GET /health
```
Check if the service is running.

### 3. System Diagnostics
```http
GET /diagnostics
```
Run system diagnostics to verify OCR dependencies (Tesseract, language packs, etc.).

### 4. Upload PDF for OCR
```http
POST /ocr
```
Upload a PDF file for processing.

**Parameters (multipart form fields):**
- `file` (required): PDF file to process
- `dpi` (optional): Rendering resolution, 72–1200 (default: 400)
- `split_pages` (optional): Split double-page scans (default: false)
- `use_easyocr` (optional): Enable EasyOCR engine (default: false)
- `gpu` (optional): Enable GPU for EasyOCR (default: false)
- `min_char_threshold` (optional): Quality threshold, 0–1,000,000 (default: 50)
- `async_mode` (optional): Queue the job for the worker (default: true)

**Response:** `202 Accepted`
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "message": "PDF uploaded successfully. Queued for processing."
}
```

With `async_mode=false` the request processes the PDF inline and returns `200` on
success or `500` if OCR fails.

### 5. Get Job Status
```http
GET /ocr/{job_id}/status
```
Check the status and live progress of a job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "filename": "example.pdf",
  "pages_processed": 10,
  "pages_total": 10,
  "result_segments": 10,
  "progress_percent": 100.0,
  "created_at": "2026-01-01T00:00:00+00:00",
  "updated_at": "2026-01-01T00:05:00+00:00",
  "files_available": ["text", "raw", "json", "review"],
  "total_pages": 10,
  "total_numerals": 42
}
```

For failed jobs the response includes an `error` field instead of
`files_available` / `total_pages` / `total_numerals`.

### 6. Get Text Output
```http
GET /ocr/{job_id}/text
```
Download the plain text output file. Returns `409` until the job is completed.

### 7. Get JSON Output
```http
GET /ocr/{job_id}/json
```
Download the structured JSON output with full metadata. Returns `409` until the
job is completed.

### 8. Get Review Report
```http
GET /ocr/{job_id}/review
```
Download the quality review report. Returns `409` until the job is completed.

### 9. Get Results Summary
```http
GET /ocr/{job_id}/results
```
Get a JSON summary of results with page-level details. Returns `409` while the
job is queued or running; pass `?include_partial=true` to read pages that have
already finished.

**Response:**
```json
{
  "job_id": "uuid-string",
  "filename": "example.pdf",
  "total_pages": 10,
  "pages": [
    {
      "page": "1",
      "text_length": 1024,
      "raw_text_length": 1010,
      "confidence": 0.95,
      "engine": "tesseract_amh+eng_psm6",
      "numeral_heavy": false,
      "numerals_found": ["፩", "፪", "፫"],
      "flagged": false,
      "review_reasons": [],
      "warnings": []
    }
  ]
}
```

### 10. Get Raw Text Output
```http
GET /ocr/{job_id}/raw
```
Download the raw OCR text before corrections are applied. Returns `409` until
the job is completed.

### 11. List Page Results
```http
GET /ocr/{job_id}/pages?include_partial=false&flagged=true&numeral_heavy=false&limit=50&offset=0
```
List page-level results with optional filters and pagination.

**Query parameters:**
- `include_partial` (optional): Include pages that have finished while the job is
  still running (default: false)
- `flagged` (optional): Only pages flagged for review when `true`, only unflagged
  pages when `false`
- `numeral_heavy` (optional): Filter by numeral-heavy pages
- `limit` (optional): 1–500 (default: 50)
- `offset` (optional): >= 0 (default: 0)

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "total": 10,
  "result_segments": 10,
  "limit": 50,
  "offset": 0,
  "pages": [
    {
      "page": "1",
      "text_length": 1024,
      "raw_text_length": 1010,
      "confidence": 0.95,
      "engine": "tesseract_amh+eng_psm6",
      "numeral_heavy": false,
      "numerals_found": ["፩", "፪", "፫"],
      "flagged": false,
      "review_reasons": [],
      "warnings": []
    }
  ]
}
```

### 12. Get Page Result
```http
GET /ocr/{job_id}/pages/{page_id}
```
Get the full result for one page, including raw text, candidates, corrections,
regions and warnings. `page_id` is the id shown in the page list, such as `12`
or `12a` for split pages. Returns `404` if the page does not exist for the job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "page": {
    "page": "12a",
    "text": "corrected text",
    "raw_text": "raw OCR text",
    "confidence": 0.93,
    "engine_name": "tesseract_amh+eng_psm6",
    "numeral_heavy": false,
    "numerals_found": ["፲"],
    "flagged": true,
    "review_reasons": ["low confidence"],
    "warnings": [],
    "candidates": [],
    "corrections": [],
    "regions": []
  }
}
```

### 13. Get Page Image
```http
GET /ocr/{job_id}/pages/{page_id}/image
```
Render the source PDF page as a PNG on demand. For split page ids such as `12a`
or `12b`, the underlying PDF page (`12`) is rendered. The rendering DPI is the
`dpi` stored with the job. Returns `404` if the page or source PDF is missing.

### 14. Download ZIP Archive
```http
GET /ocr/{job_id}/archive
```
Download all output files for a completed job as a ZIP archive. Returns `409`
until the job is completed.

### 15. Cancel Job
```http
POST /ocr/{job_id}/cancel
```
Cancel a queued or running job. Returns `202 Accepted`; a running job stops
after the page currently being processed. Returns `409` if the job has already
completed, failed or been cancelled.

### 16. Delete Job
```http
DELETE /ocr/{job_id}
```
Delete a job, its PDF and its output directory. Queued or running jobs require
`?force=true`; otherwise the request returns `409`.

### 17. List Jobs
```http
GET /ocr?status=completed&limit=50&offset=0
```
List jobs newest first. `status` accepts `queued`, `processing`, `completed`,
`failed` or `cancelled`. `limit` is 1–200, `offset` is >= 0.

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "uuid-string",
      "status": "completed",
      "filename": "example.pdf",
      "pages_processed": 10,
      "pages_total": 10,
      "result_segments": 10,
      "total_numerals": 42,
      "progress_percent": 100.0,
      "created_at": "2026-01-01T00:00:00+00:00",
      "updated_at": "2026-01-01T00:05:00+00:00"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

## Usage Examples

### Using cURL

#### Upload a PDF
```bash
curl -X POST "http://localhost:8023/ocr" \
  -F "file=@books/sample.pdf" \
  -F "dpi=400" \
  -F "split_pages=false"
```

#### Check Status
```bash
curl "http://localhost:8023/ocr/{job_id}/status"
```

#### Download Text Output
```bash
curl "http://localhost:8023/ocr/{job_id}/text" -o output.txt
```

### Using Python

```python
import requests

# Upload PDF
with open("books/sample.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8023/ocr",
        files={"file": f},
        data={"dpi": 400, "split_pages": False}
    )
    
job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")

# Check status
status_response = requests.get(f"http://localhost:8023/ocr/{job_id}/status")
print(status_response.json())

# Get results summary
results = requests.get(f"http://localhost:8023/ocr/{job_id}/results")
print(results.json())

# Download text output
text_response = requests.get(f"http://localhost:8023/ocr/{job_id}/text")
with open("output.txt", "wb") as f:
    f.write(text_response.content)
```

## Configuration

### Server Configuration

The server can be configured via environment variables or command-line arguments:

- `HOST`: Host address (default: 0.0.0.0)
- `PORT`: Port number (default: 8023)
- `RELOAD`: Enable auto-reload for development (default: False)
- `OCR_JOBS_DB`: SQLite job database (default: `results/jobs.db`)
- `OCR_UPLOAD_DIR`: Upload directory (default: `uploads`)
- `OCR_RESULTS_DIR`: Results directory (default: `results`)
- `OCR_MAX_UPLOAD_BYTES`: Maximum upload size (default: 200 MB)
- `OCR_MAX_PAGES`: Maximum PDF page count (default: 2000)
- `OCR_STALE_JOB_SECONDS`: Requeue processing jobs older than this (default: 1800)
- `OCR_WORKER_POLL_INTERVAL`: Worker queue poll interval in seconds (default: 2)

### OCR Configuration

OCR processing options can be specified when uploading PDFs:

- `dpi`: Image resolution (higher = better quality, slower processing)
- `split_pages`: Enable for double-page book scans
- `use_easyocr`: Add EasyOCR as a secondary engine (requires more memory)
- `gpu`: Enable CUDA acceleration for EasyOCR
- `min_char_threshold`: Pages with fewer characters are flagged for review

## Storage

- **Uploads**: Stored in `uploads/` directory
- **Results**: Stored in `results/{job_id}/` directory
- **Job metadata**: Stored in the SQLite database `results/jobs.db` (configurable
  with `OCR_JOBS_DB`). Jobs survive API restarts and are shared by multiple
  uvicorn workers.

Note: In a production environment, consider implementing:
- External queue/storage (Redis/Postgres) for multi-host deployments
- File cleanup policies
- Authentication and authorization
- Rate limiting
- Request validation and sanitization

## Development

### Running in Development Mode

```bash
# With auto-reload
python -m api.server
```

### Running Tests

```bash
# Test the API endpoints
curl http://localhost:8023/health
curl http://localhost:8023/diagnostics
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `202`: Accepted (job queued or cancellation requested)
- `400`: Bad request (invalid or unreadable PDF)
- `404`: Resource not found (job ID, file)
- `409`: Conflict (results not ready yet, job already completed/cancelled, or delete without force)
- `413`: Uploaded file too large
- `422`: Request validation failed
- `500`: Server error (processing failure)

Error responses include detail messages:
```json
{
  "detail": "Error description"
}
```

## Performance Considerations

- Processing time depends on PDF size, DPI, and page count
- Single-page PDFs are processed without multiprocessing overhead
- Multi-page PDFs use parallel processing (4-8 workers)
- EasyOCR requires significantly more memory and GPU resources
- Consider implementing async job processing for production use

## Security Notes

This is a basic implementation. For production deployment:

1. Add authentication (API keys, OAuth, etc.)
2. Implement rate limiting
3. Sanitize uploaded files
4. Add CORS middleware if needed
5. Use HTTPS
6. Implement file size limits
7. Add request validation
8. Set up proper logging and monitoring
