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


def _runs(mask: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    """Return ``(start, end)`` runs of true values, merging small gaps."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - previous > max_gap + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def _content_bounds(binary: np.ndarray) -> tuple[int, int, int, int]:
    """
    Estimate the ink-bearing content rectangle.

    Photographed pages often have large blank margins and a small scanner
    watermark at the bottom. A plain bounding box would count those, making a
    landscape spread look portrait. Keeping only the largest contiguous row
    band drops isolated watermarks, while the full column span preserves both
    halves of a spread.
    """
    h, w = binary.shape[:2]
    row_ink = np.count_nonzero(binary, axis=1) / max(w, 1)
    col_ink = np.count_nonzero(binary, axis=0) / max(h, 1)

    row_threshold = max(0.002, float(np.percentile(row_ink, 95)) * 0.02)
    row_runs = _runs(row_ink > row_threshold, max_gap=max(3, h // 100))
    if row_runs:
        y0, y1 = max(row_runs, key=lambda run: (run[1] - run[0], sum(row_ink[run[0] : run[1] + 1])))
    else:
        y0, y1 = 0, h - 1

    col_threshold = max(0.002, float(np.percentile(col_ink, 95)) * 0.02)
    col_runs = _runs(col_ink > col_threshold, max_gap=max(3, w // 100))
    if col_runs:
        # Use the union of substantial column runs: the gutter itself may be
        # blank in a clean scan, separating the two page columns.
        substantial = [
            run for run in col_runs if run[1] - run[0] >= max(2, int(w * 0.02))
        ]
        if not substantial:
            substantial = col_runs
        x0 = min(run[0] for run in substantial)
        x1 = max(run[1] for run in substantial)
    else:
        x0, x1 = 0, w - 1

    return x0, x1, y0, y1


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
    true when a gutter or dark binding shadow near the middle has content on
    both sides **and** the ink-bearing content region has a landscape-ish
    aspect ratio. A single portrait page with a wide centre margin is therefore
    not silently cut in half, while a photographed spread inside a portrait
    scan can still be recognised.

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
        "image_aspect_ratio": round(w / max(h, 1), 4),
        "aspect_ratio": round(w / max(h, 1), 4),
        "warnings": [],
    }

    if w < 40 or h < 40:
        result["warnings"].append("Image is too small for reliable gutter detection.")
        return result

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    content_x0, content_x1, content_y0, content_y1 = _content_bounds(binary)
    content_w = max(1, content_x1 - content_x0 + 1)
    content_h = max(1, content_y1 - content_y0 + 1)
    content_aspect = content_w / content_h
    result.update(
        {
            "content_box": [
                int(content_x0),
                int(content_y0),
                int(content_x1),
                int(content_y1),
            ],
            "aspect_ratio": round(content_aspect, 4),
        }
    )

    profile = _column_ink_profile(gray)
    smooth = _smooth_profile(profile, max(3, w // 200))

    # Search inside the content rectangle rather than the whole scan: blank
    # scanner margins should not influence the gutter estimate.
    x0 = max(content_x0, min(content_x1 - 1, content_x0 + int(content_w * search_min)))
    x1 = min(content_x1, max(x0 + 1, content_x0 + int(content_w * search_max)))
    if x1 <= x0:
        result["warnings"].append("Search interval for gutter is empty.")
        return result

    content_center = (content_x0 + content_x1) / 2.0
    half_content = max(content_w / 2.0, 1.0)
    min_band = max(3, int(content_w * min_band_ratio))
    candidates: list[dict] = []

    def _side_ink(start: int, end: int) -> tuple[float, float]:
        left = float(np.mean(smooth[content_x0:start])) if start > content_x0 else 0.0
        right = (
            float(np.mean(smooth[end + 1 : content_x1 + 1]))
            if end < content_x1
            else 0.0
        )
        return left, right

    def _centre_score(split_x: int) -> float:
        distance = abs(split_x - content_center) / half_content
        return max(0.0, 1.0 - distance)

    # 1) A clean, low-ink gutter (the usual case for flatbed scans).
    region = smooth[x0:x1]
    low_threshold = max(0.003, float(np.percentile(region, 20)) * 0.6)
    for start, end in _runs(region <= low_threshold):
        start += x0
        end += x0
        band_width = end - start + 1
        if band_width < min_band:
            continue

        left_ink, right_ink = _side_ink(start, end)
        if left_ink < min_side_ink or right_ink < min_side_ink:
            continue

        split_x = (start + end) // 2
        width_score = min(band_width / max(content_w * 0.02, 1), 1.0)
        balance = min(left_ink, right_ink) / max(left_ink, right_ink, 1e-6)
        centre_score = _centre_score(split_x)
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
                "method": "content_gutter",
            }
        )

    # 2) A dark binding shadow. Photographed open books often show the gutter
    # as a near-continuous dark band rather than a blank one. Measure vertical
    # continuity within the content rows so sparse text columns do not look
    # like a binding.
    content_rows = binary[content_y0 : content_y1 + 1]
    dark_fraction = np.count_nonzero(content_rows, axis=0) / content_h
    dark_smooth = _smooth_profile(dark_fraction, max(3, w // 200))
    dark_region = dark_smooth[x0:x1]
    high_threshold = max(0.28, float(np.percentile(dark_region, 75)) * 1.6)
    shadow_min_band = max(min_band, int(content_w * 0.008))
    for start, end in _runs(dark_region >= high_threshold):
        start += x0
        end += x0
        band_width = end - start + 1
        if band_width < shadow_min_band:
            continue

        peak = float(np.max(dark_smooth[start : end + 1]))
        split_x = (start + end) // 2
        # Binding shadows are expected near the centre of the content; a
        # dense text column elsewhere must not be mistaken for one.
        centre_distance = abs(split_x - content_center) / half_content
        if centre_distance > 0.25:
            continue

        left_ink, right_ink = _side_ink(start, end)
        if left_ink < min_side_ink or right_ink < min_side_ink:
            continue

        width_score = min(band_width / max(content_w * 0.015, 1), 1.0)
        strength = min(1.0, max(0.0, (peak - 0.20) / 0.35))
        centre_score = _centre_score(split_x)
        confidence = min(
            1.0,
            0.40 * width_score + 0.30 * strength + 0.30 * centre_score,
        )
        candidates.append(
            {
                "split_x": split_x,
                "band": [int(start), int(end)],
                "band_width": int(band_width),
                "left_ink": round(left_ink, 5),
                "right_ink": round(right_ink, 5),
                "dark_fraction": round(peak, 4),
                "confidence": round(confidence, 4),
                "method": "binding_shadow",
            }
        )

    if not candidates:
        result["warnings"].append(
            "No low-ink gutter or binding shadow with content on both sides was found."
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
            "method": best["method"],
            "candidates": candidates,
        }
    )
    if "dark_fraction" in best:
        result["dark_fraction"] = best["dark_fraction"]

    aspect_ok = content_aspect >= min_facing_aspect
    confident = best["confidence"] >= 0.45
    if aspect_ok and confident:
        result["possible_facing_pages"] = True
    else:
        if not aspect_ok:
            result["warnings"].append(
                "Gutter or binding found, but the content region looks "
                "portrait-sized; not auto-splitting."
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
