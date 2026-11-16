"""
Book template configurations, sizing presets, and metadata parsing for scanned books.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Standard paper size dimensions in PostScript points (72 points = 1 inch)
PAGE_SIZES: dict[str, tuple[float, float]] = {
    # 6" x 9" - standard trade paperback / novel size
    "novel_6x9": (432.0, 648.0),
    # 5.5" x 8.5" - standard digest novel size
    "novel_digest": (396.0, 612.0),
    # Standard ISO sizes
    "a5": (419.53, 595.28),
    "a4": (595.28, 841.89),
    "letter": (612.0, 792.0),
}

# System candidate font paths for Ethiopic Fidel
ETHIOPIC_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/fonts-senamirmir-washra/wookianos.ttf",
    "/usr/share/fonts/truetype/fonts-senamirmir-washra/jiret.ttf",
    "/usr/share/fonts/truetype/fonts-senamirmir-washra/washrab.ttf",
    "/usr/share/fonts/truetype/fonts-senamirmir-washra/tint.ttf",
    "/usr/share/fonts/truetype/fonts-senamirmir-washra/yebse.ttf",
    "/usr/share/fonts/truetype/fonts-senamirmir-washra/fantuwua.ttf",
    "/usr/share/fonts/truetype/fonts-senamirmir-washra/hiwua.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf",
    "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf",
]


def find_default_ethiopic_font() -> str:
    """Find the best available Ethiopic font on the system."""
    for path in ETHIOPIC_FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    # If none found, return the first candidate
    return ETHIOPIC_FONT_CANDIDATES[0]


def derive_metadata_from_filename(filename_or_path: str) -> tuple[str, str]:
    """
    Derive a clean book title and author from a filename or path.

    Handles common patterns:
      - 'ሞገድ - ኸርቪንግ ዋላስ.pdf' -> ('ሞገድ', 'ኸርቪንግ ዋላስ')
      - 'ራማቶሓራ 2 @ETHIO_PDF_BOOKS.pdf' -> ('ራማቶሓራ 2', '')
      - 'የቀንድ አውጣ ኑሮ - @ETHIO_PDF_BOOKS.pdf' -> ('የቀንድ አውጣ ኑሮ', '')
      - 'ፍቅር እስከ መቃብር.pdf' -> ('ፍቅር እስከ መቃብር', '')
      - 'page_002_output.json' -> ('Page 002', '')
    """
    base = os.path.basename(filename_or_path)
    # Remove extensions
    name, _ = os.path.splitext(base)
    while "." in name:
        name, _ = os.path.splitext(name)

    # Remove known OCR/split suffixes
    for suffix in ["_output", "_raw", "_review_report", "_rendered"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    # Strip telegram tags and handles (e.g., @ETHIO_PDF_BOOKS)
    name = re.sub(r"@[A-Za-z0-9_]+", "", name)

    # Normalize whitespace and separators
    name = re.sub(r"_+", " ", name).strip()

    title = name
    author = ""

    # Split by hyphen or em-dash if present
    for sep in [" - ", " – ", " — ", "-", "–"]:
        if sep in name:
            parts = [p.strip() for p in name.split(sep, 1) if p.strip()]
            if len(parts) == 2:
                # Discard tag-like parts
                if not parts[1].startswith("@"):
                    title = parts[0]
                    author = parts[1]
                else:
                    title = parts[0]
            elif len(parts) == 1:
                title = parts[0]
            break

    # Clean up title and author
    title = re.sub(r"\s+", " ", title).strip()
    author = re.sub(r"\s+", " ", author).strip()

    if not title:
        title = "Untitled Book"

    return title, author


@dataclass
class BookTemplate:
    """
    Template configuration for formatting OCR book pages into PDF or EPUB.
    Defaults to standard 6" x 9" trade novel sizing.
    """

    title: str = "Untitled Book"
    author: str = ""
    language: str = "am"  # ISO 639-1 code

    # Sizing & Layout (novel_6x9 = 432pt x 648pt)
    page_size: str = "novel_6x9"
    page_width_pt: float = 432.0
    page_height_pt: float = 648.0
    margin_left_pt: float = 45.0
    margin_right_pt: float = 45.0
    margin_top_pt: float = 45.0
    margin_bottom_pt: float = 45.0

    # Typography
    font_path: str = field(default_factory=find_default_ethiopic_font)
    font_name: str = "EthiopicWookianos"
    font_size: float = 11.0
    line_spacing: float = 1.35  # Line height multiplier

    # Header & Footer
    include_page_numbers: bool = True
    include_running_header: bool = False
    footer_font_size: float = 9.0
    header_font_size: float = 8.5

    # Structure & Chapter handling
    break_pages_at_markers: bool = True
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_size in PAGE_SIZES:
            dims = PAGE_SIZES[self.page_size]
            self.page_width_pt = dims[0]
            self.page_height_pt = dims[1]

        # Extract font name from path if default
        if self.font_path and os.path.exists(self.font_path):
            stem = Path(self.font_path).stem
            self.font_name = stem.capitalize()

    @property
    def content_width(self) -> float:
        return max(50.0, self.page_width_pt - self.margin_left_pt - self.margin_right_pt)

    @property
    def content_height(self) -> float:
        return max(50.0, self.page_height_pt - self.margin_top_pt - self.margin_bottom_pt)

    @property
    def text_rect(self) -> tuple[float, float, float, float]:
        """Returns (x0, y0, x1, y1) rectangle for page body text."""
        return (
            self.margin_left_pt,
            self.margin_top_pt,
            self.page_width_pt - self.margin_right_pt,
            self.page_height_pt - self.margin_bottom_pt,
        )

    @classmethod
    def from_filename(
        cls,
        filename_or_path: str,
        page_size: str = "novel_6x9",
        **kwargs: Any,
    ) -> BookTemplate:
        """Create a BookTemplate automatically populating title/author from filename."""
        title, author = derive_metadata_from_filename(filename_or_path)
        return cls(title=title, author=author, page_size=page_size, **kwargs)
