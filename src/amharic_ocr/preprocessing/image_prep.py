"""
Image preprocessing routines for Amharic text and Ethiopic numeral OCR.
"""

import cv2
import numpy as np

from .page_analysis import estimate_skew


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Helper to convert BGR/RGB to Grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def normalize_background(image: np.ndarray) -> np.ndarray:
    """
    Flat-field background illumination normalization.
    Compensates for uneven lighting, shadows across book spine, and yellowed paper.
    Uses morphological dilation to estimate background and normalizes against it.
    """
    gray = to_grayscale(image)
    # 51x51 kernel captures slow background illumination changes without capturing text glyphs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    norm = cv2.divide(gray, dilated, scale=255)
    return norm


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Enhances faint Fidel characters on faded or aged paper.
    """
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def crop_to_text_content(image: np.ndarray, padding_ratio: float = 0.02) -> np.ndarray:
    """
    Content-aware text block detection and margin cropping.

    .. warning::
       Cropping can delete marginalia, footnotes, page numbers, or text that
       touches an edge. It is opt-in (``--crop``) and every crop is recorded
       in the per-page diagnostics. The untouched page is still retained.
    """
    gray = to_grayscale(image)
    h, w = gray.shape[:2]

    # Otsu thresholding inverted (text foreground is white)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dynamic kernel size proportional to image dimensions
    k_w = max(int(w * 0.02), 15)
    k_h = max(int(h * 0.01), 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    min_x, min_y = w, h
    max_x, max_y = 0, 0
    min_area = (w * h) * 0.0005  # Filter out tiny speckles/dust

    found_content = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Skip outer boundary frames that span almost the entire image
            if cw > w * 0.98 and ch > h * 0.98:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + cw)
            max_y = max(max_y, y + ch)
            found_content = True

    if not found_content or min_x >= max_x or min_y >= max_y:
        return image

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, min_x - pad_x)
    y1 = max(0, min_y - pad_y)
    x2 = min(w, max_x + pad_x)
    y2 = min(h, max_y + pad_y)

    return image[y1:y2, x1:x2]


def remove_binding_shadow(image: np.ndarray) -> np.ndarray:
    """
    Detect and crop the dark vertical band produced by book spine binding.

    .. warning::
       This is destructive and is **not** used by default. It assumes the
       darkest outer columns are binding shadow, which is not always true for
       dark covers, edge annotations, or text that reaches the page edge.
       Enable with ``--remove-shadow`` and inspect the recorded crop.
    """
    gray = to_grayscale(image)
    w = gray.shape[1]

    # Per-column mean brightness
    col_means = np.mean(gray, axis=0)

    # --- Left binding shadow ---
    dark_threshold = 80
    left_limit = int(w * 0.20)  # only inspect the leftmost 20%
    shadow_end_left = 0
    for i in range(left_limit):
        if col_means[i] < dark_threshold:
            shadow_end_left = i

    # --- Right binding shadow ---
    right_start = int(w * 0.80)
    shadow_start_right = w
    for i in range(w - 1, right_start - 1, -1):
        if col_means[i] < dark_threshold:
            shadow_start_right = i

    # Add a small padding (0.5% of width) past the shadow boundary
    pad = max(int(w * 0.005), 5)
    crop_left = min(shadow_end_left + pad, left_limit)
    crop_right = max(shadow_start_right - pad, right_start)

    # Only crop if a meaningful shadow was actually detected
    if crop_left > pad or crop_right < w - pad:
        image = image[:, crop_left:crop_right]

    return image


def remove_scan_artifacts(image: np.ndarray) -> np.ndarray:
    """
    Remove common scanner artifacts from the image margins.

    .. warning::
       Trimming the bottom 8% can delete footnotes or page numbers. This is
       opt-in (``--trim-edges``) and is recorded in diagnostics; it is never
       applied silently by the default workflow.
    """
    gray = to_grayscale(image)
    h = gray.shape[0]

    # Inspect the bottom 8% for watermark / whitespace
    bottom_start = int(h * 0.92)
    bottom_strip = gray[bottom_start:, :]
    strip_mean = np.mean(bottom_strip)

    # Watermark regions are mostly white (mean > 220) with a few dark chars
    if strip_mean > 210:
        image = image[:bottom_start, :]

    return image


def deskew_with_info(
    image: np.ndarray,
    max_angle: float = 10.0,
    enabled: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Deskew an image and return both the result and the decision.

    The rotation is skipped when the angle is negligible, unreliable, or
    larger than ``max_angle``. This avoids silently applying a bad estimate
    to a page that may simply be rotated or have a strong border.
    """
    info = estimate_skew(image, max_angle=max_angle)
    info["enabled"] = enabled
    info["applied"] = False

    if not enabled:
        info["notes"].append("Deskew disabled by configuration.")
        return image, info
    if not info["reliable"]:
        info["notes"].append("Deskew skipped because the estimate was not reliable.")
        return image, info

    angle = float(info["angle_deg"])
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    info["applied"] = True
    info["applied_angle_deg"] = angle
    return rotated, info


def deskew(image: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
    """
    Fix image skew/rotation for better OCR alignment.

    Backwards-compatible wrapper around :func:`deskew_with_info`. New code
    should use ``deskew_with_info`` so the decision can be recorded.
    """
    rotated, _ = deskew_with_info(image, max_angle=max_angle)
    return rotated

def sauvola_threshold(image: np.ndarray, window_size: int = 15, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    """
    Applies Sauvola local thresholding to a grayscale image.
    T = m * (1 + k * (s / R - 1))
    """
    gray = to_grayscale(image)
    
    # Calculate local mean (m)
    mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
    
    # Calculate local mean of squared image (E[X^2])
    sq_gray = np.square(gray.astype(np.float32))
    sq_mean = cv2.boxFilter(sq_gray, cv2.CV_32F, (window_size, window_size))
    
    # Local variance = E[X^2] - (E[X])^2
    var = sq_mean - np.square(mean)
    var = np.maximum(var, 0)  # Ensure variance is non-negative
    std = np.sqrt(var)
    
    # Compute Sauvola threshold
    threshold = mean * (1.0 + k * (std / R - 1.0))
    
    # Binarize: standard text is black-on-white (foreground is 0, background is 255)
    binary = np.zeros_like(gray)
    binary[gray > threshold] = 255
    
    return binary

def preprocess_for_numerals(image: np.ndarray, use_sauvola: bool = False) -> np.ndarray:
    """
    Specialized preprocessing for Ethiopic numeral recognition.
    Numerals need gentler processing than text:
    - Resize small images if min dimension < 50
    - Non-local means denoising
    - Adaptive Gaussian thresholding or Sauvola local thresholding
    - Morphological closing to connect broken numeral strokes
    """
    gray = to_grayscale(image)
    
    # Resize small images (numerals need sufficient resolution)
    height, width = gray.shape
    min_dim = min(height, width)
    if min_dim < 50:
        scale = 100 / min_dim
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Mild denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    if use_sauvola:
        binary = sauvola_threshold(denoised)
    else:
        # Adaptive thresholding preserves numeral details better than Otsu
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            15, 10
        )
    
    # Morphological closing to connect broken numeral strokes
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def preprocess_for_text(image: np.ndarray, use_sauvola: bool = False, enhance_contrast: bool = True) -> np.ndarray:
    """
    Standard preprocessing for Amharic text:
    - Grayscale conversion
    - Flat-field illumination normalization (removes page yellowing / dark shadows)
    - Median blur denoising (kernel=3 to preserve fine Fidel stroke variations)
    - Contrast enhancement (CLAHE)
    - Otsu thresholding or Sauvola local thresholding
    """
    gray = to_grayscale(image)

    # Upscale very small images so Otsu doesn't produce a blank binary
    if min(gray.shape) < 50:
        scale = 100 / min(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Normalize background illumination surface to level lighting gradients
    normalized = normalize_background(gray)

    # Use a 3×3 kernel instead of 5×5: Fidel order-mark hooks (e.g. ሰ→ሶ) differ
    # by just a few pixels; a 5-pixel blur erases them and collapses entire
    # character columns into the same glyph.
    denoised = cv2.medianBlur(normalized, 3)

    if enhance_contrast:
        denoised = apply_clahe(denoised, clip_limit=2.0)

    if use_sauvola:
        binary = sauvola_threshold(denoised)
    else:
        # Otsu thresholding for text
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

def detect_if_numeral_heavy(image: np.ndarray) -> bool:
    """
    Heuristic to detect if image contains mostly numerals/sparse structures.
    Based on the ratio of small isolated components to total components.

    The old threshold of 20 small components triggered on virtually every
    400 DPI text page (Fidel characters produce hundreds of small connected
    components), causing false positives that then corrupted body text
    through numeral-zone Latin-to-Ethiopic conversions.

    Now requires that >70% of all components are small AND the total text
    ink coverage is low (< 5% of image area), which genuinely indicates
    a numeral table or sparse numeral list rather than body text.
    """
    gray = to_grayscale(image)
    h, w = gray.shape[:2]
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary)
    
    if num_labels <= 1:
        return False

    # Count components excluding the background (label 0)
    total_components = num_labels - 1
    small_components = sum(1 for i in range(1, num_labels) 
                           if stats[i, cv2.CC_STAT_WIDTH] < w * 0.05
                           and stats[i, cv2.CC_STAT_HEIGHT] < h * 0.05)

    # Ink coverage: fraction of the image that is foreground
    ink_pixels = np.count_nonzero(binary)
    ink_ratio = ink_pixels / (h * w)

    small_ratio = small_components / total_components

    # Numeral-heavy = mostly tiny isolated glyphs AND low ink coverage
    return small_ratio > 0.70 and ink_ratio < 0.05 and total_components < 200
