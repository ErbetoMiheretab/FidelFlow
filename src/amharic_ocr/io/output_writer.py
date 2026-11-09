"""
Output generation for OCR results (plain text, raw text, JSON metadata, and
quality/diagnostic review reports).
"""

from __future__ import annotations

import json
import re
from typing import Any


def _page_sort_key(page: Any):
    text = str(page)
    match = re.match(r"^(\d+)(.*)$", text)
    if match:
        return (int(match.group(1)), match.group(2))
    return (10**9, text)


def save_text_output(
    results: list[dict[str, Any]],
    output_path: str,
    header: str | None = None,
) -> None:
    """Save page text in the familiar ``=== Page X ===`` format."""
    with open(output_path, "w", encoding="utf-8") as f:
        if header:
            f.write(f"# {header}\n\n")
        sorted_results = sorted(results, key=lambda x: _page_sort_key(x["page"]))
        for result in sorted_results:
            f.write(f"=== Page {result['page']} ===\n")
            f.write(result.get("text", "") + "\n")
            if result.get("numerals_found"):
                f.write(f"[Numerals: {', '.join(result['numerals_found'])}]\n")
            f.write("\n")


def _serialise_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "page": result.get("page"),
        "text": result.get("text", ""),
        "raw_text": result.get("raw_text", ""),
        "confidence": float(result.get("confidence", 0.0)),
        "engine_name": result.get("engine_name", "unknown"),
        "numeral_heavy": bool(result.get("numeral_heavy", False)),
        "numerals_found": result.get("numerals_found", []),
        "numerals_found_raw": result.get("numerals_found_raw", []),
        "numeral_pass": result.get("numeral_pass"),
        "character_count": len(result.get("text", "")),
        "raw_character_count": len(result.get("raw_text", "")),
        "is_flagged_for_review": bool(result.get("flagged", False)),
        "review_reasons": result.get("review_reasons", []),
        "warnings": result.get("warnings", []),
        "image_properties": result.get("image_properties", {}),
        "skew": result.get("skew", {}),
        "segmentation": result.get("segmentation", {}),
        "segment": result.get("segment", {}),
        "preprocessing": result.get("preprocessing", {}),
        "candidates": result.get("candidates", []),
        "corrections": result.get("corrections", []),
        "regions": result.get("regions", []),
        "model_status": result.get("model_status", {}),
    }


def save_json_output(results: list[dict[str, Any]], output_path: str) -> None:
    """Save structured JSON output with page text, diagnostics and candidates."""
    serializable = [_serialise_result(r) for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)


def generate_review_report(results: list[dict[str, Any]], output_path: str) -> None:
    """Generate a human-readable quality and diagnostics report."""
    flagged_pages = [r for r in results if r.get("flagged", False)]
    total_numerals = sum(len(r.get("numerals_found", [])) for r in results)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AMHARIC OCR PIPELINE - QUALITY AND DIAGNOSTICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total processed pages: {len(results)}\n")
        f.write(f"Total Ethiopic numerals detected: {total_numerals}\n")
        f.write(f"Pages flagged for review: {len(flagged_pages)}\n\n")
        f.write(
            "Scores and flags are ranking aids, not proof of correctness. "
            "Compare the raw and cleaned text on flagged pages.\n\n"
        )

        f.write("PER-PAGE DIAGNOSTICS:\n")
        f.write("-" * 80 + "\n")
        for r in sorted(results, key=lambda x: _page_sort_key(x["page"])):
            props = r.get("image_properties", {}) or {}
            segmentation = r.get("segmentation", {}) or {}
            preprocessing = r.get("preprocessing", {}) or {}
            f.write(
                f"Page {r.get('page')}: engine={r.get('engine_name')} "
                f"conf={r.get('confidence', 0.0):.2f} "
                f"chars={len(r.get('text', ''))} "
                f"raw_chars={len(r.get('raw_text', ''))}\n"
            )
            f.write(
                f"  layout: split_mode={segmentation.get('mode')} "
                f"split_x={segmentation.get('split_x')} "
                f"kinds={segmentation.get('gutter', {}).get('method')} "
                f"text_density={props.get('ink_ratio')} "
                f"skew={props.get('skew', {}).get('angle_deg')}deg "
                f"variant={preprocessing.get('selected_variant')}\n"
            )
            f.writelines(
                f"  warning: {warning}\n"
                for warning in r.get("warnings", []) or []
            )
            f.writelines(
                f"  correction: {correction.get('type')} "
                f"count={correction.get('count')} "
                f"({correction.get('detail')})\n"
                for correction in r.get("corrections", []) or []
            )
            if r.get("flagged"):
                reasons = ", ".join(r.get("review_reasons", []))
                f.write(f"  FLAGGED: {reasons}\n")
                raw_snippet = r.get("raw_text", "")[:160].replace("\n", " ")
                final_snippet = r.get("text", "")[:160].replace("\n", " ")
                f.write(f"  raw snippet:   {raw_snippet}...\n")
                f.write(f"  final snippet: {final_snippet}...\n")
            candidates = r.get("candidates", []) or []
            if candidates:
                f.write("  candidates:\n")
                f.writelines(
                    f"    - {candidate.get('engine')} "
                    f"[{candidate.get('variant')}] "
                    f"scope={candidate.get('scope')} "
                    f"score={candidate.get('score')} "
                    f"conf={candidate.get('confidence'):.2f} "
                    f"chars={candidate.get('character_count')}\n"
                    for candidate in candidates
                )
            f.write("-" * 80 + "\n")

        if flagged_pages:
            f.write("\nFLAGGED PAGE SUMMARY:\n")
            f.write("-" * 80 + "\n")
            for r in flagged_pages:
                reasons = ", ".join(r.get("review_reasons", []))
                f.write(f"Page {r['page']}: {reasons}\n")
        else:
            f.write(
                "\nNo pages crossed the review thresholds. This does not prove "
                "that the OCR is correct; inspect samples of the output.\n"
            )


def save_evaluation_output(report: dict[str, Any], output_path: str) -> None:
    """Save an evaluation report as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
