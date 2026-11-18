"""
Books rendering and template module for converting scanned OCR text into digital books.
"""

from .renderer import BookRenderer, main, parse_input_source, render_book, safe_slug
from .template import (
    PAGE_MARGINS,
    PAGE_SIZES,
    PAPER_COLORS,
    BookTemplate,
    derive_metadata_from_filename,
    find_default_ethiopic_font,
    validate_font_coverage,
)

__all__ = [
    "BookRenderer",
    "BookTemplate",
    "PAGE_MARGINS",
    "PAGE_SIZES",
    "PAPER_COLORS",
    "derive_metadata_from_filename",
    "find_default_ethiopic_font",
    "main",
    "parse_input_source",
    "render_book",
    "safe_slug",
    "validate_font_coverage",
]
