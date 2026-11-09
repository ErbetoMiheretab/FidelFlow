"""
Estimates of observable page properties used for diagnostics and safe defaults.

Nothing in this module is a book/font classifier. It only reports properties
that can be measured from the image itself (skew, contrast, ink density) plus
a conservative facing-page hint from :mod:`page_split`.
"""

from __future__ import annotations

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def estimate_skew(image: np.ndarray, max_angle: float = 20.0) -> dict:
    """
    Estimate text skew with a small downscaled copy.

    Returns an angle in degrees where positive values mean the page should be
    rotated counter-clockwise by ``cv2.getRotationMatrix2D``. ``reliable`` is
    false for near-zero angles, very sparse pages, or implausibly large angles.
    """
    gray = _to_gray(image)
    h, w = gray.shape[:2]
    info: dict = {
        "angle_deg": 0.0,
        "reliable": False,
        "method": "min_area_rect",
        "notes": [],
    }
    if h < 20 or w < 20:
        info["notes"].append("Image too small for skew estimation.")
        return info

    # Downscale large 400 DPI pages; this keeps the estimate fast without
    # changing the rotation that is ultimately applied to the full image.
    scale = min(1.0, 1600.0 / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(
        small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Prefer near-horizontal Hough line segments. Text baselines are a much
    # better skew signal than the minimum-area rectangle of the whole page,
    # which often locks onto the page border instead of the text.
    edges = cv2.Canny(binary, 50, 150)
    min_length = max(30, int(small.shape[1] * 0.08))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(30, int(small.shape[1] * 0.03)),
        minLineLength=min_length,
        maxLineGap=max(8, int(small.shape[1] * 0.01)),
    )
    angles: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) <= max_angle:
                angles.append(angle)

    if len(angles) >= 3:
        angle = float(np.median(angles))
        info["angle_deg"] = float(round(angle, 3))
        info["method"] = "hough_lines"
        info["line_count"] = len(angles)
        if abs(angle) < 0.15:
            info["notes"].append("Skew is negligible.")
            return info
        info["reliable"] = True
        return info

    points = cv2.findNonZero(binary)
    if points is None or len(points) < 50:
        info["notes"].append("Too few foreground pixels for skew estimation.")
        return info

    (_, _), (rw, rh), angle = cv2.minAreaRect(points)
    # Normalise OpenCV's angle convention (which differs between versions) to
    # a small rotation around horizontal text.
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    info["angle_deg"] = float(round(angle, 3))
    info["method"] = "min_area_rect"
    if abs(angle) < 0.15:
        info["notes"].append("Skew is negligible.")
        return info
    if abs(angle) > max_angle:
        info["notes"].append(
            f"Estimated skew {angle:.2f}° exceeds safe limit {max_angle:.2f}°."
        )
        return info
    # A minimum-area rectangle can lock onto page borders rather than text.
    # Reject boxes that are implausibly elongated for a text-bearing page.
    long_side, short_side = max(rw, rh), max(min(rw, rh), 1.0)
    if long_side / short_side > 20:
        info["notes"].append("Skew estimate looks like a border/edge artifact.")
        return info

    info["reliable"] = True
    return info


def estimate_page_properties(
    image: np.ndarray,
    gutter_info: dict | None = None,
) -> dict:
    """Return observable image statistics for per-page diagnostics."""
    gray = _to_gray(image)
    h, w = gray.shape[:2]
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    ink_ratio = float(np.count_nonzero(binary) / max(h * w, 1))

    # Use robust percentiles so a single black scanner edge does not dominate
    # the contrast estimate.
    p05, p95 = np.percentile(gray, [5, 95])
    contrast = float(p95 - p05)
    mean_brightness = float(np.mean(gray))

    # Simple row-profile text density: fraction of rows with meaningful ink.
    row_ink = np.count_nonzero(binary, axis=1) / max(w, 1)
    text_rows = int(np.count_nonzero(row_ink > 0.01))

    skew = estimate_skew(image)
    props = {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(w / max(h, 1), 4),
        "mean_brightness": round(mean_brightness, 2),
        "contrast_p95_minus_p05": round(contrast, 2),
        "ink_ratio": round(ink_ratio, 5),
        "text_row_ratio": round(text_rows / max(h, 1), 4),
        "skew": skew,
        "possible_facing_pages": bool(
            gutter_info.get("possible_facing_pages", False)
        )
        if gutter_info
        else None,
    }
    if gutter_info:
        props["gutter"] = {
            "split_x": gutter_info.get("split_x"),
            "confidence": gutter_info.get("confidence"),
            "warnings": gutter_info.get("warnings", []),
        }
    return props
