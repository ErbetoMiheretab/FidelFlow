"""
Image preprocessing routines.
"""

from .image_prep import (
    deskew,
    deskew_with_info,
    detect_if_numeral_heavy,
    normalize_background,
    preprocess_for_numerals,
    preprocess_for_text,
    to_grayscale,
)
from .page_analysis import estimate_page_properties, estimate_skew
from .page_split import detect_page_gutter, split_image_horizontally
from .variants import PreprocessedVariant, build_preprocessing_variants

__all__ = [
    "PreprocessedVariant",
    "build_preprocessing_variants",
    "deskew",
    "deskew_with_info",
    "detect_if_numeral_heavy",
    "detect_page_gutter",
    "estimate_page_properties",
    "estimate_skew",
    "normalize_background",
    "preprocess_for_numerals",
    "preprocess_for_text",
    "split_image_horizontally",
    "to_grayscale",
]
