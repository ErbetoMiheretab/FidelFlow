"""
Main execution pipeline for Amharic OCR.

The default workflow is designed for unknown books: it keeps an untouched page
and raw OCR, estimates observable image properties, detects a facing-page
gutter conservatively, compares a few preprocessing variants, and records every
choice per page.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import pytesseract

from .config import OCRConfig
from .constants import ALL_ETHIOPIC_NUMERALS
from .engines.tesseract import TesseractEngine
from .fusion.combiner import (
    OCRResultCombiner,
    score_ocr_result_detailed,
)
from .io.output_writer import generate_review_report, save_json_output, save_text_output
from .io.pdf_reader import get_page_count, render_page_to_numpy
from .postprocessing.corrections import (
    extract_ethiopic_numerals,
    post_process_text_with_report,
)
from .preprocessing.image_prep import (
    deskew_with_info,
    detect_if_numeral_heavy,
)
from .preprocessing.page_analysis import estimate_page_properties
from .preprocessing.page_split import detect_page_gutter
from .preprocessing.variants import build_preprocessing_variants


def _read_image_file(path: str) -> np.ndarray:
    """Read an image file, including non-ASCII paths on Windows/Linux."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _write_image_file(path: str, image: np.ndarray) -> None:
    """Write an image to a path that may contain non-ASCII characters."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    extension = os.path.splitext(path)[1] or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Could not encode image for {path}")
    encoded.tofile(path)


def _make_segments(image: np.ndarray, config: OCRConfig) -> tuple[list[dict], dict]:
    """Return page segments and a diagnostic description of the split choice."""
    w = image.shape[1]
    gutter = detect_page_gutter(image)
    split_x: int | None = None
    mode = "none"
    warnings: list[str] = []

    if config.split_at is not None:
        fraction = min(max(float(config.split_at), 0.05), 0.95)
        split_x = round(w * fraction)
        mode = "manual_fraction"
    elif config.split_pages:
        detected = gutter.get("split_x")
        split_x = int(detected) if detected else w // 2
        mode = "forced"
        if not detected:
            warnings.append(
                "Forced split at the image centre because no gutter was found; "
                "this may cut through text."
            )
    elif config.auto_split and gutter.get("possible_facing_pages"):
        split_x = int(gutter["split_x"])
        mode = "auto"
        if float(gutter.get("confidence", 0.0)) < 0.55:
            warnings.append(
                "Automatic facing-page split had low confidence; inspect both halves."
            )

    if split_x is None:
        segments = [{"suffix": "", "image": image, "offset": [0, 0]}]
    else:
        split_x = max(1, min(split_x, w - 1))
        segments = [
            {"suffix": "a", "image": image[:, :split_x], "offset": [0, 0]},
            {"suffix": "b", "image": image[:, split_x:], "offset": [split_x, 0]},
        ]

    segmentation = {
        "mode": mode,
        "split_x": split_x,
        "warnings": warnings + list(gutter.get("warnings", [])),
        "gutter": gutter,
        "segments": [
            {
                "suffix": seg["suffix"],
                "offset": seg["offset"],
                "shape": list(seg["image"].shape),
            }
            for seg in segments
        ],
    }
    return segments, segmentation


def _offset_regions(regions: list[dict], dx: int, dy: int) -> list[dict]:
    """Convert segment-local region boxes to page-image coordinates."""
    shifted = []
    for region in regions:
        item = dict(region)
        bbox = list(item.get("bbox", []))
        if len(bbox) == 4:
            item["bbox"] = [int(bbox[0] + dx), int(bbox[1] + dy), bbox[2], bbox[3]]
        item["page_offset"] = [dx, dy]
        shifted.append(item)
    return shifted


def _candidate_summary(result) -> dict:
    details = score_ocr_result_detailed(result)
    metadata = result.metadata or {}
    return {
        "engine": result.engine_name,
        "variant": metadata.get("variant"),
        "scope": metadata.get("scope", "page"),
        "confidence": float(result.confidence),
        "score": details["score"],
        "score_components": details["components"],
        "character_count": len(result.text),
        "text": result.text,
        "description": metadata.get("description"),
        "transformations": metadata.get("transformations", []),
        "warning": metadata.get("warning"),
        "amharic_model_available": metadata.get("amharic_model_available"),
    }


def _process_segment(
    segment_image: np.ndarray,
    page_id: str,
    config: OCRConfig,
    segmentation: dict,
    segment_info: dict,
) -> dict:
    original_segment = segment_image
    properties = estimate_page_properties(original_segment)
    numeral_heavy = detect_if_numeral_heavy(original_segment) if config.detect_numerals else False

    working_image, skew_info = deskew_with_info(
        original_segment,
        max_angle=config.max_deskew_angle,
        enabled=config.auto_deskew,
    )
    geometry_transformations: list[dict] = [
        {
            "name": "deskew",
            "requested": config.auto_deskew,
            "applied": bool(skew_info.get("applied")),
            "angle_deg": skew_info.get("angle_deg", 0.0),
            "reliable": bool(skew_info.get("reliable")),
            "notes": skew_info.get("notes", []),
        }
    ]

    variants = build_preprocessing_variants(
        original_segment,
        config,
        working_image=working_image,
        geometry_transformations=geometry_transformations,
    )

    candidates = []
    warnings: list[str] = []
    model_status: dict[str, Any] = {}
    numeral_candidates = []

    if config.use_tesseract:
        engine = TesseractEngine()
        model_status["tesseract"] = engine.model_status()
        language_notes = engine._resolve_language(config)[1]
        warnings.extend(language_notes)
        candidates.extend(engine.recognize_variants(variants, config))

        if (
            config.include_numeral_candidate
            and config.detect_numerals
            and numeral_heavy
        ):
            numeral_candidates.append(
                engine.ocr_numerals_only(
                    working_image, use_sauvola=bool(config.use_sauvola)
                )
            )

    if config.use_easyocr:
        try:
            from .engines.easyocr_engine import EasyOCREngine

            easy = EasyOCREngine()
            model_status["easyocr"] = easy.model_status()
            easy_result = easy.recognize(working_image, config)
            easy_result.metadata["variant"] = "easyocr"
            if not easy.amharic_supported:
                easy_result.metadata["scope"] = "supplementary"
                warnings.append(
                    "EasyOCR has no Amharic model. Its Latin-model output is kept "
                    "for diagnostics but is not used to outvote Amharic OCR."
                )
            candidates.append(easy_result)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"EasyOCR could not run: {e}")

    candidates.extend(numeral_candidates)
    selectable = [
        r
        for r in candidates
        if (r.metadata or {}).get("scope", "page") == "page"
    ]
    if not selectable:
        selectable = candidates

    selected = OCRResultCombiner.combine(selectable)
    selected_metadata = selected.metadata or {}
    raw_text = selected.text

    post = post_process_text_with_report(
        raw_text,
        is_numeral_heavy=numeral_heavy,
        apply_numeral_corrections=config.apply_numeral_corrections,
        convert_latin_digits=config.convert_latin_digits,
        deduplicate_numeral_runs=config.deduplicate_numeral_runs,
        clean_margin_artifacts_enabled=config.clean_margin_artifacts,
        strip_margin_noise_edges=config.strip_margin_noise_edges,
    )
    final_text = post["text"]
    numerals_found = extract_ethiopic_numerals(final_text)
    numerals_raw = extract_ethiopic_numerals(raw_text)

    numeral_pass = None
    if numeral_candidates:
        best_numeral = OCRResultCombiner.combine(numeral_candidates)
        numeral_pass = {
            "text": best_numeral.text,
            "confidence": float(best_numeral.confidence),
            "numerals_found": extract_ethiopic_numerals(best_numeral.text),
            "candidates": [_candidate_summary(r) for r in numeral_candidates],
        }

    flagged = False
    reasons: list[str] = []

    if selected.confidence != -1.0 and selected.confidence < 0.6:
        flagged = True
        reasons.append(f"Low confidence ({selected.confidence:.2f})")
    if len(final_text.strip()) < config.min_char_threshold:
        flagged = True
        reasons.append(f"Low character count ({len(final_text.strip())} chars)")
    if (
        not final_text.strip()
        or "[ERROR" in final_text
        or "[Tesseract Error" in final_text
        or "[EasyOCR Error" in final_text
    ):
        flagged = True
        reasons.append("Empty output or engine error")
    if post["review_flags"]:
        flagged = True
        reasons.extend(post["review_flags"])

    # Destructive transformations are always visible, whether or not they
    # changed the image, because the user asked for them.
    for transformation in selected_metadata.get("transformations", []) or []:
        if transformation.get("requested"):
            warnings.append(
                f"{transformation.get('name')} was requested "
                f"(applied={transformation.get('applied')})."
                + (
                    " " + transformation["warning"]
                    if transformation.get("warning")
                    else ""
                )
            )

    regions = _offset_regions(
        selected_metadata.get("regions", []) or [],
        segment_info["offset"][0],
        segment_info["offset"][1],
    )

    return {
        "page": page_id,
        "text": final_text,
        "raw_text": raw_text,
        "confidence": float(selected.confidence),
        "engine_name": selected.engine_name,
        "numeral_heavy": bool(numeral_heavy),
        "numerals_found": numerals_found,
        "numerals_found_raw": numerals_raw,
        "numeral_pass": numeral_pass,
        "flagged": flagged,
        "review_reasons": reasons,
        "warnings": warnings,
        "image_properties": properties,
        "skew": skew_info,
        "segmentation": segmentation,
        "segment": {
            "suffix": segment_info["suffix"],
            "offset": segment_info["offset"],
            "shape": list(original_segment.shape),
        },
        "preprocessing": {
            "selected_variant": selected_metadata.get("variant"),
            "description": selected_metadata.get("description"),
            "transformations": selected_metadata.get("transformations", []),
            "variants_compared": [variant.summary() for variant in variants],
        },
        "candidates": [_candidate_summary(r) for r in candidates]
        if config.save_candidates
        else [],
        "corrections": post["changes"],
        "regions": regions,
        "model_status": model_status,
    }


def process_page_image(
    image: np.ndarray,
    page_id: str,
    config: OCRConfig | None = None,
    original_image_path: str | None = None,
) -> list[dict[str, Any]]:
    """Process an already-rendered page image (used by PDF and evaluation paths)."""
    if config is None:
        config = OCRConfig()

    if original_image_path:
        _write_image_file(original_image_path, image)

    segments, segmentation = _make_segments(image, config)
    results: list[dict[str, Any]] = []

    for segment in segments:
        segment_id = f"{page_id}{segment['suffix']}"
        try:
            results.append(
                _process_segment(
                    segment["image"],
                    segment_id,
                    config,
                    segmentation,
                    segment,
                )
            )
        except Exception as e:  # noqa: BLE001
            results.append(
                {
                    "page": segment_id,
                    "text": f"[ERROR: Page processing failed: {e}]",
                    "raw_text": "",
                    "confidence": 0.0,
                    "engine_name": "pipeline",
                    "numeral_heavy": False,
                    "numerals_found": [],
                    "numerals_found_raw": [],
                    "numeral_pass": None,
                    "flagged": True,
                    "review_reasons": [f"Fatal processing error: {e}"],
                    "warnings": [],
                    "image_properties": {},
                    "skew": {},
                    "segmentation": segmentation,
                    "segment": {
                        "suffix": segment["suffix"],
                        "offset": segment["offset"],
                        "shape": list(segment["image"].shape),
                    },
                    "preprocessing": {},
                    "candidates": [],
                    "corrections": [],
                    "regions": [],
                    "model_status": {},
                }
            )
    return results


def process_image_file(
    image_path: str,
    config: OCRConfig | None = None,
    page_id: str = "1",
) -> list[dict[str, Any]]:
    """Convenience wrapper for evaluating a single image file."""
    image = _read_image_file(image_path)
    return process_page_image(image, page_id, config)


def process_single_page(args) -> list[dict[str, Any]]:
    """
    Worker function to process a single page of a PDF.

    Returns a list of page result dicts (two if the page is split).
    """
    if len(args) == 4:
        page_num, pdf_path, config, image_dir = args
    else:
        page_num, pdf_path, config = args
        image_dir = None

    try:
        image = render_page_to_numpy(pdf_path, page_num, dpi=config.dpi)
    except Exception as e:  # noqa: BLE001
        return [
            {
                "page": str(page_num),
                "text": f"[ERROR: Page rendering failed: {e}]",
                "raw_text": "",
                "confidence": 0.0,
                "engine_name": "pdf_reader",
                "numeral_heavy": False,
                "numerals_found": [],
                "numerals_found_raw": [],
                "numeral_pass": None,
                "flagged": True,
                "review_reasons": [f"PDF rendering error: {e}"],
                "warnings": [],
                "image_properties": {},
                "skew": {},
                "segmentation": {},
                "segment": {},
                "preprocessing": {},
                "candidates": [],
                "corrections": [],
                "regions": [],
                "model_status": {},
            }
        ]

    original_path = None
    if image_dir:
        original_path = os.path.join(image_dir, f"page_{page_num:04d}_original.png")

    return process_page_image(
        image,
        str(page_num),
        config,
        original_image_path=original_path,
    )


def run_pipeline(
    pdf_path: str,
    config: OCRConfig | None = None,
    status_callback: Callable[[int, int], None] | None = None,
    result_callback: Callable[[list[dict[str, Any]], int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Main entry point for running the extraction pipeline on a PDF file.

    ``status_callback(pages_done, pages_total)`` is called after each PDF page.
    ``result_callback(page_results, pages_done, pages_total)`` additionally
    receives the results for that page, so callers can expose partial output
    while the job is still running.
    """
    if config is None:
        config = OCRConfig()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    os.makedirs(config.output_dir, exist_ok=True)

    try:
        langs = pytesseract.get_languages()
        print(f"Available Tesseract languages: {langs}")
        if "amh" not in langs:
            print("⚠️  Warning: 'amh' language pack not found!")
            print("   Install with: sudo apt-get install tesseract-ocr-amh")
    except Exception as e:  # noqa: BLE001
        print(f"Could not check Tesseract languages: {e}")

    if config.use_easyocr:
        from .engines.easyocr_engine import easyocr_supports_amharic

        if not easyocr_supports_amharic():
            print(
                "⚠️  EasyOCR has no Amharic model. It will run as a Latin/digit "
                "diagnostic only and will not be treated as an Amharic improvement."
            )

    num_pages = get_page_count(pdf_path)

    print(f"\nProcessing '{os.path.basename(pdf_path)}' ({num_pages} pages)...")
    print(f"Ethiopic numerals considered: {ALL_ETHIOPIC_NUMERALS}")
    print("-" * 50)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_dir = None
    if config.save_page_images:
        image_dir = os.path.join(config.output_dir, f"{base_name}_pages")
        os.makedirs(image_dir, exist_ok=True)

    tasks = [(i + 1, pdf_path, config, image_dir) for i in range(num_pages)]
    all_results = []

    max_workers = min(mp.cpu_count(), 4) if config.use_easyocr else min(mp.cpu_count(), 8)

    if num_pages == 1:
        page_results = process_single_page(tasks[0])
        all_results.extend(page_results)
        for r in page_results:
            print(
                f"  Page {r['page']}: Text length: {len(r['text'])} chars | "
                f"Raw length: {len(r.get('raw_text', ''))} | "
                f"Numerals found: {r['numerals_found']}"
            )
        if result_callback:
            result_callback(page_results, 1, num_pages)
        if status_callback:
            status_callback(1, num_pages)
    else:
        with mp.Pool(processes=max_workers) as pool:
            for idx, page_results in enumerate(pool.imap(process_single_page, tasks)):
                all_results.extend(page_results)
                for r in page_results:
                    print(
                        f"  Page {r['page']}/{num_pages}: Text length: "
                        f"{len(r['text'])} chars | Raw length: "
                        f"{len(r.get('raw_text', ''))} | "
                        f"Numerals found: {r['numerals_found']}"
                    )
                if result_callback:
                    result_callback(page_results, idx + 1, num_pages)
                if status_callback:
                    status_callback(idx + 1, num_pages)

    txt_out = os.path.join(config.output_dir, f"{base_name}_output.txt")
    raw_out = os.path.join(config.output_dir, f"{base_name}_raw.txt")
    json_out = os.path.join(config.output_dir, f"{base_name}_output.json")
    report_out = os.path.join(config.output_dir, f"{base_name}_review_report.txt")

    save_text_output(all_results, txt_out)
    save_text_output(
        [
            {
                "page": r["page"],
                "text": r.get("raw_text", r.get("text", "")),
                "numerals_found": r.get("numerals_found_raw", []),
            }
            for r in all_results
        ],
        raw_out,
        header="RAW OCR (before corrections)",
    )
    save_json_output(all_results, json_out)
    generate_review_report(all_results, report_out)

    total_numerals = sum(len(r["numerals_found"]) for r in all_results)
    flagged = sum(1 for r in all_results if r.get("flagged"))

    print("\n" + "=" * 50)
    print(f"✅ OCR complete: {txt_out}")
    print(f"🧾 Raw OCR: {raw_out}")
    print(f"📊 Total numerals detected: {total_numerals}")
    print(f"🚩 Pages flagged for review: {flagged}")
    print(f"📄 Structured JSON: {json_out}")
    print(f"📋 Review Report: {report_out}")
    if image_dir:
        print(f"🖼  Original page images: {image_dir}")
    print("=" * 50)

    return all_results
