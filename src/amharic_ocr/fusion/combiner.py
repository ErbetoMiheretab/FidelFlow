"""
Candidate scoring and engine fusion.

Scoring is a ranking heuristic, not a correctness measure. Tesseract confidence
can be high on repeated garbage, and Ethiopic character density can be high on
hallucinated Fidel text. The score therefore combines several weak signals and
explicitly penalises unknown symbols, garbage characters and numeral-only
candidates competing as full-page text.
"""

from __future__ import annotations

import math

from ..engines.base import OCRResult

_GARBAGE_CHARS = set("|~^\\_`<>{}[]")


def _ethiopic_ratio(text: str) -> float:
    """Fraction of characters in the Ethiopic Unicode block (U+1200-U+137F)."""
    if not text:
        return 0.0
    ethiopic = sum(1 for c in text if "\u1200" <= c <= "\u137f")
    return ethiopic / len(text)


def _unknown_ratio(text: str) -> float:
    """Fraction of characters outside common Ethiopic/Latin/punctuation sets."""
    if not text:
        return 0.0
    unknown = 0
    for c in text:
        if c.isspace():
            continue
        if "\u1200" <= c <= "\u137f":
            continue
        if c.isascii() and (c.isalnum() or c in ".,;:!?()[]{}<>«»'\"\\/-—–"):
            continue
        unknown += 1
    return unknown / len(text)


def score_ocr_result_detailed(result: OCRResult) -> dict:
    """Return a candidate score plus the components used to compute it."""
    text = result.text.strip()
    if not text:
        return {
            "score": -1.0,
            "components": {"empty": True},
        }

    confidence = (
        max(result.confidence, 0.0) if result.confidence != -1.0 else 0.5
    )
    eth_ratio = _ethiopic_ratio(text)
    unknown_ratio = _unknown_ratio(text)
    garbage_ratio = sum(1 for c in text if c in _GARBAGE_CHARS) / len(text)

    # Logarithmic length factor: more complete text is useful, but a long
    # hallucinated run must not automatically beat a shorter clean result.
    length_factor = min(math.log1p(len(text)) / math.log1p(1000), 1.0)

    # Numeral-only candidates may contain real numerals but are not evidence
    # for the rest of the page. Keep them available but do not let them win as
    # full-page text by default.
    scope_penalty = 0.15 if result.metadata.get("scope") == "numerals" else 0.0

    score = (
        (confidence * 0.30)
        + (eth_ratio * 0.20)
        + (length_factor * 0.20)
        - (garbage_ratio * 0.45)
        - (unknown_ratio * 0.25)
        - scope_penalty
    )
    return {
        "score": round(float(score), 6),
        "components": {
            "confidence": round(float(confidence), 4),
            "ethiopic_ratio": round(float(eth_ratio), 4),
            "length_factor": round(float(length_factor), 4),
            "garbage_ratio": round(float(garbage_ratio), 5),
            "unknown_ratio": round(float(unknown_ratio), 5),
            "scope_penalty": scope_penalty,
            "chars": len(text),
        },
    }


def score_ocr_result(result: OCRResult) -> float:
    """Composite ranking score for an OCR candidate (higher is better)."""
    return float(score_ocr_result_detailed(result)["score"])


class OCRResultCombiner:
    @staticmethod
    def combine(results: list[OCRResult], strategy: str = "confidence") -> OCRResult:
        """
        Combine OCR results from multiple engines.

        Supported strategies:
        - "confidence"/"score": composite ranking (default)
        - "longest": longest non-error text
        """
        if not results:
            return OCRResult(text="", confidence=0.0, engine_name="combiner")

        valid_results = [
            r
            for r in results
            if r.text.strip()
            and not r.text.startswith("[ERROR")
            and not r.text.startswith("[Tesseract Error")
        ]

        if not valid_results:
            return results[0]

        if strategy == "longest":
            return max(valid_results, key=lambda r: len(r.text))
        return max(valid_results, key=score_ocr_result)
