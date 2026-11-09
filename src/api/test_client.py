"""
Simple test client for the Amharic OCR API.
"""

import sys
import time
from pathlib import Path

import requests


def test_api(pdf_path: str, base_url: str = "http://localhost:8023"):
    """Test the OCR API with a sample PDF."""
    print(f"Testing Amharic OCR API at {base_url}")
    print("=" * 50)

    # 1. Health check
    print("\n1. Health Check...")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    try:
        print(f"   Response: {response.json()}")
    except Exception:
        print(f"   Response: {response.text}")

    # 2. System diagnostics
    print("\n2. System Diagnostics...")
    response = requests.get(f"{base_url}/diagnostics")
    if response.status_code == 200:
        print(f"   Status: {response.status_code}")
        print(f"   Diagnostics passed: {response.json().get('status')}")
    else:
        print(f"   Warning: Diagnostics returned {response.status_code}")

    # 3. Upload PDF for OCR
    print(f"\n3. Uploading PDF: {pdf_path}")
    if not Path(pdf_path).exists():
        print(f"   Error: File not found: {pdf_path}")
        return

    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f, "application/pdf")}
        data = {
            "dpi": 400,
            "split_pages": False,
            "use_easyocr": False,
            "min_char_threshold": 50,
        }
        response = requests.post(f"{base_url}/ocr", files=files, data=data)

    if response.status_code not in (200, 202):
        print(f"   Error: Upload failed with status {response.status_code}")
        print(f"   Response: {response.json()}")
        return

    result = response.json()
    job_id = result["job_id"]
    print(f"   Job ID: {job_id}")
    print(f"   Status: {result['status']}")

    # 4. Wait for the job to finish
    print("\n4. Waiting for Job...")
    deadline = time.time() + 600
    status = {}
    while time.time() < deadline:
        response = requests.get(f"{base_url}/ocr/{job_id}/status")
        status = response.json()
        if status.get("status") in {"completed", "failed", "cancelled"}:
            break
        print(
            f"   Status: {status.get('status')} "
            f"({status.get('pages_processed', 0)}/"
            f"{status.get('pages_total', '?')} pages)"
        )
        time.sleep(2)

    print(f"   Final status: {status.get('status')}")
    print(f"   Filename: {status.get('filename')}")
    if status.get("status") not in {"completed", "failed", "cancelled"}:
        print("   Timed out waiting for the job. Check /status later.")
        return
    if status.get("status") == "completed":
        print(f"   Total Pages: {status.get('total_pages', 'N/A')}")
        print(f"   Total Numerals: {status.get('total_numerals', 'N/A')}")
    elif status.get("status") in {"failed", "cancelled"}:
        print(f"   Error: {status.get('error', status.get('status'))}")
        return

    # 5. Get results summary
    print("\n5. Getting Results Summary...")
    response = requests.get(f"{base_url}/ocr/{job_id}/results")
    if response.status_code == 200:
        results = response.json()
        print(f"   Total Pages: {results['total_pages']}")
        for page in results["pages"][:3]:  # Show first 3 pages
            print(f"   - Page {page['page']}: {page['text_length']} chars, "
                  f"confidence: {page['confidence']:.2f}, "
                  f"numerals: {page['numerals_found']}")
        if len(results["pages"]) > 3:
            print(f"   ... and {len(results['pages']) - 3} more pages")

    # 6. Download text output
    print("\n6. Downloading Text Output...")
    response = requests.get(f"{base_url}/ocr/{job_id}/text")
    if response.status_code == 200:
        output_file = f"test_output_{job_id}.txt"
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"   Saved to: {output_file}")
        # Show first 200 characters
        with open(output_file, "r", encoding="utf-8") as f:
            preview = f.read(200)
        print(f"   Preview: {preview}...")

    # 7. Download JSON output
    print("\n7. Downloading JSON Output...")
    response = requests.get(f"{base_url}/ocr/{job_id}/json")
    if response.status_code == 200:
        output_file = f"test_output_{job_id}.json"
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"   Saved to: {output_file}")

    print("\n" + "=" * 50)
    print("✅ API Test Complete!")
    print(f"Job ID: {job_id}")
    print("\nTo delete this job, run:")
    print(f"  curl -X DELETE {base_url}/ocr/{job_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <pdf_path> [base_url]")
        print("Example: python test_client.py books/sample.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8023"

    test_api(pdf_path, base_url)
