"""
Configuration options for the Amharic OCR pipeline.

The defaults are intentionally conservative: they are designed for unknown
scanned books and do not require the user to identify a font or book family.
Destructive geometry changes (content cropping, edge trimming, shadow cropping)
are opt-in so that footnotes, marginal text and page numbers are not lost.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class OCRConfig:
    # PDF rendering DPI. 400 DPI keeps fine Fidel strokes legible; lower values
    # are faster but reduce accuracy on small Ethiopic glyphs.
    dpi: int = 400

    # Page splitting. ``split_pages`` forces a split, ``auto_split`` detects a
    # facing-page gutter when the scan actually looks like a two-page spread.
    split_pages: bool = False
    auto_split: bool = True
    # Optional manual split point as a fraction of image width (0.0-1.0).
    split_at: float | None = None

    # Engines to enable. Tesseract is the only engine with a bundled Amharic
    # language model in a standard install. EasyOCR currently has no Amharic
    # model, so it is disabled by default and is never described as an
    # Amharic-accuracy improvement.
    use_tesseract: bool = True
    use_easyocr: bool = False

    # Languages to load.
    languages: list[str] = field(default_factory=lambda: ["amh", "eng"])

    # Tesseract configuration. PSM 3 (fully automatic page segmentation) is
    # safer than PSM 6 for unknown books with headers, footnotes or columns.
    tesseract_psm: int = 3
    tesseract_oem: int = 1
    layout_mode: Literal["auto", "page", "block", "line"] = "auto"

    # EasyOCR configuration.
    easyocr_gpu: bool = False

    # Preprocessing variants to compare. All variants are OCR'd from the same
    # (possibly deskewed) page, and the untouched page is always kept.
    # Supported: "raw", "normalized", "local", "otsu".
    preprocessing_variants: list[str] = field(
        default_factory=lambda: ["raw", "normalized", "local"]
    )

    # Geometry correction. Deskewing is skipped when the estimated angle is
    # unreliable or large; the decision is recorded in diagnostics.
    auto_deskew: bool = True
    max_deskew_angle: float = 10.0

    # Opt-in destructive transformations. These are off by default because
    # they can remove footnotes, marginalia, page numbers or text near a
    # gutter. When enabled, every transformation is recorded per page.
    crop_to_content: bool = False
    remove_binding_shadow: bool = False
    remove_scan_artifacts: bool = False

    # Error detection: flag page if character count is below this.
    min_char_threshold: int = 50

    # Paths.
    output_dir: str = "output_texts"
    save_page_images: bool = False
    save_candidates: bool = True

    # Numeral handling. Numeral candidates are kept separate from body text.
    detect_numerals: bool = True
    include_numeral_candidate: bool = True
    # Contextual Latin-lookalike repair is only attempted on numeral-heavy
    # regions. Automatic Arabic-digit conversion and run de-duplication are
    # opt-in because both can change genuine numbers.
    apply_numeral_corrections: bool = True
    convert_latin_digits: bool = False
    deduplicate_numeral_runs: bool = False

    # Only lines that contain nothing but scanner noise are removed. Stripping
    # leading/trailing dashes or punctuation is opt-in via this flag.
    clean_margin_artifacts: bool = True
    strip_margin_noise_edges: bool = False

    # Kept for backwards compatibility. When true, the "local" variant is
    # guaranteed to be included even if the user supplied a custom list.
    use_sauvola: bool = False
