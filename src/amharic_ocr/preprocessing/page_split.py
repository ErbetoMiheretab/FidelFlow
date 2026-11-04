"""
Facing-page detection and page splitting.

The old implementation always cut a two-page image at ``width // 2``. Real
scans often have an off-centre gutter, so that could cut through text. The
functions here estimate the gutter from the image content and only split when
there is evidence of a facing-page layout. A manual split point remains
available as an override.
"""

from __future__ import annotations

import cv2
import numpy as np


def _column_ink_profile(image: np.ndarray) -> np.ndarray:
    """Return the fraction of foreground ink in each image column."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return np.count_nonzero(binary, axis=0) / max(gray.shape[0], 1)


def _smooth_profile(profile: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)  # odd kernel size
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(profile, kernel, mode="same")


def detect_page_gutter(
    image: np.ndarray,
    search_min: float = 0.25,
    search_max: float = 0.75,
    min_band_ratio: float = 0.004,
    min_side_ink: float = 0.005,
    min_facing_aspect: float = 1.05,
) -> dict:
    """
    Estimate whether an image is a facing-page spread and where its gutter is.

    The result is deliberately conservative. ``possible_facing_pages`` is only
    true when a low-ink vertical band near the middle has content on both
    sides **and** the page has a landscape-ish aspect ratio. A single portrait
    page with a wide centre margin is therefore not silently cut in half.

    Returns a JSON-serialisable dictionary with the split coordinate,
    confidence, and reasons so callers can expose the decision to users.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    result: dict = {
        "possible_facing_pages": False,
        "split_x": None,
        "confidence": 0.0,
        "method": "content_gutter",
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(w / max(h, 1), 4),
        "warnings": [],
    }

    if w < 40 or h < 40:
        result["warnings"].append("Image is too small for reliable gutter detection.")
        return result

    profile = _column_ink_profile(gray)
    smooth = _smooth_profile(profile, max(3, w // 200))

    x0 = max(1, int(w * search_min))
    x1 = min(w - 1, int(w * search_max))
    if x1 <= x0:
        result["warnings"].append("Search interval for gutter is empty.")
        return result

    region = smooth[x0:x1]
    # A gutter is a column band with very little foreground ink. Use a low
    # percentile of the search region rather than a hard global threshold.
    low_threshold = max(0.003, float(np.percentile(region, 20)) * 0.6)
    low = region <= low_threshold

    runs: list[tuple[int, int]] = []
    run_start = None
    for i, is_low in enumerate(low):
        if is_low and run_start is None:
            run_start = i
        elif not is_low and run_start is not None:
            runs.append((run_start + x0, i + x0))
            run_start = None
    if run_start is not None:
        runs.append((run_start + x0, len(low) + x0))

    min_band = max(3, int(w * min_band_ratio))
    candidates = []
    for start, end in runs:
        band_width = end - start
        if band_width < min_band:
            continue

        left_ink = float(np.mean(smooth[:start])) if start > 0 else 0.0
        right_ink = float(np.mean(smooth[end:])) if end < w else 0.0
        if left_ink < min_side_ink or right_ink < min_side_ink:
            continue

        split_x = (start + end) // 2
        # Prefer a wide, clean band close to the centre, but do not force the
        # centre: off-centre gutters are common in photographed books.
        distance = abs(split_x - (w / 2.0)) / (w / 2.0)
        width_score = min(band_width / max(w * 0.02, 1), 1.0)
        balance = min(left_ink, right_ink) / max(left_ink, right_ink, 1e-6)
        centre_score = max(0.0, 1.0 - distance)
        confidence = min(
            1.0,
            0.45 * width_score + 0.30 * balance + 0.25 * centre_score,
        )
        candidates.append(
            {
                "split_x": split_x,
                "band": [int(start), int(end)],
                "band_width": int(band_width),
                "left_ink": round(left_ink, 5),
                "right_ink": round(right_ink, 5),
                "confidence": round(confidence, 4),
            }
        )

    if not candidates:
        result["warnings"].append(
            "No low-ink vertical gutter with text on both sides was found."
        )
        return result

    best = max(candidates, key=lambda c: c["confidence"])
    result.update(
        {
            "split_x": int(best["split_x"]),
            "confidence": float(best["confidence"]),
            "band": best["band"],
            "band_width": best["band_width"],
            "left_ink": best["left_ink"],
            "right_ink": best["right_ink"],
            "candidates": candidates,
        }
    )

    aspect_ok = result["aspect_ratio"] >= min_facing_aspect
    confident = best["confidence"] >= 0.45
    if aspect_ok and confident:
        result["possible_facing_pages"] = True
    else:
        if not aspect_ok:
            result["warnings"].append(
                "Gutter found, but image looks portrait-sized; not auto-splitting."
            )
        if not confident:
            result["warnings"].append(
                "Gutter candidate confidence is too low for automatic splitting."
            )

    return result


def split_image_horizontally(
    image: np.ndarray,
    split_x: int | None = None,
    overlap: int = 0,
) -> list[np.ndarray]:
    """
    Split an image into left and right pages.

    Args:
        image: Source image.
        split_x: Optional explicit x coordinate. When omitted, the detected
            gutter is used and then the image centre as a last resort.
        overlap: Optional pixels duplicated on both sides of the split. Keep
            at 0 for normal use; a small overlap can help manual inspection.
    """
    h, w = image.shape[:2]
    if split_x is None:
        info = detect_page_gutter(image)
        split_x = info.get("split_x") or (w // 2)
    split_x = max(0, min(int(split_x), w))
    overlap = max(0, min(int(overlap), split_x, w - split_x))

    left_page = image[0:h, 0:min(w, split_x + overlap)]
    right_page = image[0:h, max(0, split_x - overlap):w]
    return [left_page, right_page]
