"""
Diagnostics and system verification for Amharic OCR & Ethiopic numeral
recognition.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .engines.tesseract import TesseractEngine


def check_engine_capabilities() -> dict:
    """Report which installed engines actually have an Amharic model."""
    status: dict = {"tesseract": {}, "easyocr": {}}
    try:
        engine = TesseractEngine()
        status["tesseract"] = engine.model_status()
    except Exception as e:  # noqa: BLE001
        status["tesseract"] = {"error": str(e), "tesseract_amh_available": False}

    try:
        import importlib.util

        if importlib.util.find_spec("easyocr") is None:
            status["easyocr"] = {
                "installed": False,
                "easyocr_amharic_available": False,
            }
        else:
            from .engines.easyocr_engine import easyocr_supports_amharic

            status["easyocr"] = {
                "installed": True,
                "easyocr_amharic_available": easyocr_supports_amharic(),
            }
    except Exception as e:  # noqa: BLE001
        status["easyocr"] = {
            "installed": False,
            "easyocr_amharic_available": False,
            "error": str(e),
        }
    return status


def test_numeral_recognition() -> str:
    """
    Test Ethiopic numeral recognition with an in-memory **synthetic** image.

    This checks that Fidel numeral glyphs can be rendered and recognized; it
    does not predict accuracy on scanned book pages.
    """
    img = Image.new("L", (400, 100), color=255)
    draw = ImageDraw.Draw(img)

    test_text = "፩ ፪ ፫ ፬ ፭ ፮ ፯ ፰ ፱ ፲"

    font = None
    try:
        from .evaluation import find_ethiopic_font

        font_path = find_ethiopic_font()
        if font_path:
            font = ImageFont.truetype(font_path, 32)
    except OSError:
        font = None

    if font is None:
        font = ImageFont.load_default()

    draw.text((10, 30), test_text, fill=0, font=font)

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)

    print("Testing Ethiopic numeral recognition (SYNTHETIC image)...")
    print(f"Expected: {test_text}")

    engine = TesseractEngine()
    result = engine.ocr_numerals_only(cv_img)
    print(f"Got:      {result.text}")
    print("Note: synthetic results do not predict scanned-book accuracy.")

    return result.text


def check_system_setup():
    """Run full environment diagnostics."""
    print("=" * 60)
    print("AMHARIC OCR ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)

    print(f"Python Version: {sys.version.splitlines()[0]}")
    print(f"OpenCV Version: {cv2.__version__}")

    try:
        import fitz

        print(f"PyMuPDF (fitz) Version: {getattr(fitz, '__version__', 'Installed')}")
    except ImportError:
        print("❌ PyMuPDF (fitz) is NOT installed!")

    capabilities = check_engine_capabilities()

    print("\n--- ENGINE / AMHARIC MODEL CAPABILITY ---")
    tesseract = capabilities.get("tesseract", {})
    if tesseract.get("tesseract_amh_available"):
        print("  ✅ Tesseract 'amh' model is installed.")
    else:
        print("  ❌ Tesseract 'amh' model is NOT installed.")
        print("     Install with: sudo apt-get install tesseract-ocr-amh")
    print(
        f"  Available Tesseract languages: "
        f"{tesseract.get('available_languages', [])}"
    )

    easyocr = capabilities.get("easyocr", {})
    if not easyocr.get("installed"):
        print("  ⚠️  EasyOCR is not installed (optional; no Amharic model anyway).")
    elif easyocr.get("easyocr_amharic_available"):
        print("  ✅ EasyOCR reports an Amharic model.")
    else:
        print(
            "  ⚠️  EasyOCR has NO Amharic model in this installation. "
            "It is not an Amharic accuracy improvement."
        )

    tesseract_path = os.environ.get("TESSDATA_PREFIX", "Not set")
    print(f"\nTESSDATA_PREFIX env: {tesseract_path}")

    print("\n--- ETHIOPIC NUMERAL TEST (SYNTHETIC) ---")
    test_numeral_recognition()

    print("\n--- EVALUATION ---")
    print(
        "Run `amharic-ocr --evaluate <dir>` with your own image/transcription "
        "pairs for real CER/WER. Use `--synthetic-eval` only for a labelled "
        "synthetic regression check."
    )
    print("=" * 60)


if __name__ == "__main__":
    check_system_setup()
