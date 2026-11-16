"""
Books rendering and template module for converting scanned OCR text into digital books.
"""

from .renderer import BookRenderer, main, parse_input_source, render_book
from .template import (
    PAGE_SIZES,
    BookTemplate,
    derive_metadata_from_filename,
    find_default_ethiopic_font,
)

__all__ = [
    "BookRenderer",
    "BookTemplate",
    "PAGE_SIZES",
    "derive_metadata_from_filename",
    "find_default_ethiopic_font",
    "main",
    "parse_input_source",
    "render_book",
]
