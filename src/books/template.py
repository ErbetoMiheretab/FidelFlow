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

# Per-preset margins (left, right, top, bottom) in points.
# Naively reusing the 6x9 margins on A4/Letter leaves ~0.62" gutters,
# which is too tight for print.
PAGE_MARGINS: dict[str, tuple[float, float, float, float]] = {
    "novel_6x9": (54.0, 54.0, 54.0, 60.0),  # 0.75" sides
    "novel_digest": (48.0, 48.0, 48.0, 54.0),
    "a5": (50.0, 50.0, 50.0, 56.0),
    "a4": (72.0, 72.0, 72.0, 80.0),  # 1" sides
    "letter": (72.0, 72.0, 72.0, 80.0),
}

# Fonts shipped with the package. Bundling makes rendering reproducible;
# distro font availability varies (many images ship none of the system paths).
ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"

# Preferred order: full Ethiopic coverage, real hinting, and a bold companion.
# Senamirmir WashRa is kept only as a last resort: Wookianos covers 342/384
# codepoints of the base Ethiopic block and none of Ethiopic Supplement /
# Extended, so common fidel (e.g. ሇ ዯ ቬ ፇ) render as .notdef.
ETHIOPIC_FONT_CANDIDATES: list[str] = [
    str(ASSETS_DIR / "NotoSerifEthiopic-Regular.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSerifEthiopic-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSerifEthiopic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansEthiopic-Regular.ttf",
    "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf",
    "/usr/share/fonts/truetype/fonts-senamirmir-washra/jiret.ttf",
    "/usr/share/fonts/truetype/fonts-senamirmir-washra/wookianos.ttf",
]

ETHIOPIC_BOLD_FONT_CANDIDATES: list[str] = [
    str(ASSETS_DIR / "NotoSerifEthiopic-Bold.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSerifEthiopic-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSerifEthiopic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Bold.ttf",
    "/usr/share/fonts/truetype/fonts-senamirmir-washra/washrab.ttf",
]

# Ethiopic fonts carry no Latin letters or Arabic digits. Amharic books still
# contain years, page numbers, codes, and "©", so every text run needs a Latin
# companion with matching vertical metrics (Noto Serif: 1.069 / -0.293, same
# as Noto Serif Ethiopic).
LATIN_FONT_CANDIDATES: list[str] = [
    str(ASSETS_DIR / "NotoSerif-Regular.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
]

LATIN_BOLD_FONT_CANDIDATES: list[str] = [
    str(ASSETS_DIR / "NotoSerif-Bold.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
]


def _first_existing(candidates: list[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def find_default_ethiopic_font() -> str:
    """Find the best available Ethiopic font, preferring the bundled Noto Serif."""
    path = _first_existing(ETHIOPIC_FONT_CANDIDATES)
    if not path:
        raise FileNotFoundError(
            "No Ethiopic font found. Install one (e.g. fonts-noto-core) or pass "
            "--font /path/to/Ethiopic.ttf."
        )
    return path


def find_default_ethiopic_bold_font() -> str:
    """Find a bold Ethiopic companion font, or '' when none is available."""
    return _first_existing(ETHIOPIC_BOLD_FONT_CANDIDATES)


def find_default_latin_font() -> str:
    """Find a Latin serif companion for digits and Latin text."""
    path = _first_existing(LATIN_FONT_CANDIDATES)
    if not path:
        raise FileNotFoundError(
            "No Latin serif font found for digits/Latin text. Install fonts-noto-core "
            "or pass --latin-font /path/to/serif.ttf."
        )
    return path


def find_default_latin_bold_font() -> str:
    """Find a bold Latin serif companion, or '' when none is available."""
    return _first_existing(LATIN_BOLD_FONT_CANDIDATES)


# Ethiopic script blocks. Anything outside these ranges is rendered with the
# Latin companion font.
ETHIOPIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x1200, 0x137F),  # Ethiopic
    (0x1380, 0x139F),  # Ethiopic Supplement
    (0x2D80, 0x2DDF),  # Ethiopic Extended
    (0xAB00, 0xAB2F),  # Ethiopic Extended-A
)


def is_ethiopic(char: str) -> bool:
    """True when *char* belongs to an Ethiopic script block."""
    code = ord(char)
    return any(lo <= code <= hi for lo, hi in ETHIOPIC_RANGES)


# A cross-section of fidel plus the codepoints the scanned corpus actually
# contains. Used to surface .notdef holes before a book is rendered.
COVERAGE_SAMPLE = "ሀለሐመሠረሰሸቀበተኀነአከወዐዘዠየደጀገጠጨፀፈፐሇዯቬፇ"


def validate_font_coverage(font_path: str, text: str = "") -> list[str]:
    """
    Return the distinct characters in *text* that the font cannot render.

    Scanning a real corpus is the only reliable way to catch holes in legacy
    Ethiopic fonts; an empty list means every character has a glyph.
    """
    if not font_path or not os.path.exists(font_path):
        return sorted(set(text))

    import fitz  # Local import so metadata helpers stay dependency-free.

    font = fitz.Font(fontfile=font_path)
    sample = COVERAGE_SAMPLE + text
    missing = sorted({ch for ch in sample if ch.strip() and not font.has_glyph(ord(ch))})
    return missing


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    """Convert '#RRGGBB' (or '#RGB') to a PyMuPDF/CSS friendly RGB tuple."""
    raw = value.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        raise ValueError(f"Invalid hex color: {value!r}")
    try:
        return tuple(int(raw[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"Invalid hex color: {value!r}") from exc


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

    Defaults target a 6" x 9" trade paperback with warm paper, 11.5 pt Noto
    Serif Ethiopic, 1.5 line spacing, and a 1.2 em first-line indent.
    """

    title: str = "Untitled Book"
    author: str = ""
    language: str = "am"  # ISO 639-1 code

    # Sizing & Layout (0 means "use the page_size preset")
    page_size: str = "novel_6x9"
    page_width_pt: float = 0.0
    page_height_pt: float = 0.0
    margin_left_pt: float = 0.0
    margin_right_pt: float = 0.0
    margin_top_pt: float = 0.0
    margin_bottom_pt: float = 0.0

    # Typography
    font_path: str = field(default_factory=find_default_ethiopic_font)
    font_bold_path: str = field(default_factory=find_default_ethiopic_bold_font)
    font_latin_path: str = field(default_factory=find_default_latin_font)
    font_latin_bold_path: str = field(default_factory=find_default_latin_bold_font)
    font_name: str = "NotoSerifEthiopic"
    font_size: float = 11.5
    chapter_font_size: float = 18.0
    title_font_size: float = 26.0
    line_spacing: float = 1.5  # Line height multiplier (Ethiopic needs >= 1.45)
    paragraph_indent_em: float = 1.2
    body_align: str = "justify"  # "justify" or "left"

    # Colors (screen-friendly warm paper; use #FFFFFF for print)
    page_bg_hex: str = "#FBF7EF"
    text_color_hex: str = "#1C1A17"
    secondary_color_hex: str = "#5E5A54"

    # Header & Footer
    include_page_numbers: bool = True
    include_running_header: bool = False
    footer_font_size: float = 9.0
    header_font_size: float = 9.0

    # EPUB
    epub_line_height: float = 1.65
    epub_chapter_pages: int = 15  # Group this many OCR pages per XHTML file

    # Structure & Chapter handling
    break_pages_at_markers: bool = True
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_size in PAGE_SIZES:
            dims = PAGE_SIZES[self.page_size]
            if self.page_width_pt <= 0:
                self.page_width_pt = dims[0]
            if self.page_height_pt <= 0:
                self.page_height_pt = dims[1]

            margins = PAGE_MARGINS.get(self.page_size, (45.0, 45.0, 45.0, 45.0))
            if self.margin_left_pt <= 0:
                self.margin_left_pt = margins[0]
            if self.margin_right_pt <= 0:
                self.margin_right_pt = margins[1]
            if self.margin_top_pt <= 0:
                self.margin_top_pt = margins[2]
            if self.margin_bottom_pt <= 0:
                self.margin_bottom_pt = margins[3]

        # Fall back to 6" x 9" for unknown presets with unset dimensions.
        self.page_width_pt = self.page_width_pt or 432.0
        self.page_height_pt = self.page_height_pt or 648.0
        self.margin_left_pt = self.margin_left_pt or 54.0
        self.margin_right_pt = self.margin_right_pt or 54.0
        self.margin_top_pt = self.margin_top_pt or 54.0
        self.margin_bottom_pt = self.margin_bottom_pt or 60.0

        # Validate colors early so a typo fails at config time, not mid-render.
        for attr in ("page_bg_hex", "text_color_hex", "secondary_color_hex"):
            hex_to_rgb(getattr(self, attr))

        # Extract font name from path if default
        if self.font_path and os.path.exists(self.font_path):
            self.font_name = Path(self.font_path).stem

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

    @property
    def page_bg_rgb(self) -> tuple[float, float, float]:
        return hex_to_rgb(self.page_bg_hex)

    @property
    def text_color_rgb(self) -> tuple[float, float, float]:
        return hex_to_rgb(self.text_color_hex)

    @property
    def secondary_color_rgb(self) -> tuple[float, float, float]:
        return hex_to_rgb(self.secondary_color_hex)

    @property
    def is_print_paper(self) -> bool:
        return self.page_bg_hex.strip().upper() in {"#FFF", "#FFFFFF"}

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
