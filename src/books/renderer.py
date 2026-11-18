"""
Book renderer module: converts OCR results into formatted text-layer PDF or EPUB.
Supports standard 6x9 trade novel dimensions, Ethiopic font embedding, and clean pagination.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import fitz  # PyMuPDF
from ebooklib import epub

from books.template import (
    BookTemplate,
    derive_metadata_from_filename,
    find_default_ethiopic_font,
    is_ethiopic,
    validate_font_coverage,
)

# Chapter markers as they appear in the scanned Amharic corpus. "ምእራፍ" is a
# common OCR variant of "ምዕራፍ".
_HEADING_RE = re.compile(r"^ም[ዕእ]ራፍ\b")


def _page_sort_key(page: Any) -> tuple[int, str]:
    text = str(page)
    match = re.match(r"^(\d+)(.*)$", text)
    if match:
        return (int(match.group(1)), match.group(2))
    return (10**9, text)


def _clean_ocr_text(text: str) -> str:
    """Clean OCR output: strip review markers, numeral annotations, etc."""
    cleaned_lines = []
    for line in text.splitlines():
        trimmed = line.strip()
        # Filter out metadata lines like [Numerals: ...] or === Page X ===
        if trimmed.startswith("[Numerals:") and trimmed.endswith("]"):
            continue
        if re.match(r"^===\s*Page\s+.*\s*===$", trimmed, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


@dataclass(frozen=True)
class _Block:
    """A reflowed paragraph or chapter heading."""

    text: str
    heading: bool = False


def _reflow_blocks(text: str) -> list[_Block]:
    """
    Turn OCR lines into paragraphs.

    OCR emits one line per visual line; treating each as a paragraph inserts a
    blank line between every line of prose (and nearly doubles the page count).
    Blank lines are the only paragraph separators; chapter markers start a new
    block of their own.
    """
    blocks: list[_Block] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraph = " ".join(part.strip() for part in buffer if part.strip())
            if paragraph:
                blocks.append(_Block(text=re.sub(r"\s+", " ", paragraph)))
            buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if _HEADING_RE.match(line):
            flush()
            blocks.append(_Block(text=line, heading=True))
            continue
        buffer.append(line)

    flush()
    return blocks


@dataclass(frozen=True)
class _FontPair:
    """An Ethiopic font plus its Latin companion (digits, punctuation)."""

    ethiopic: fitz.Font
    latin: fitz.Font
    ascender: float
    descender: float


@dataclass(frozen=True)
class _FontSet:
    regular: _FontPair
    bold: _FontPair


def _load_font_pair(ethiopic_path: str, latin_path: str) -> _FontPair:
    ethiopic = fitz.Font(fontfile=ethiopic_path)
    latin = fitz.Font(fontfile=latin_path) if latin_path else ethiopic
    return _FontPair(
        ethiopic=ethiopic,
        latin=latin,
        ascender=max(ethiopic.ascender, latin.ascender),
        descender=min(ethiopic.descender, latin.descender),
    )


def _load_fonts(template: BookTemplate) -> _FontSet:
    regular = _load_font_pair(template.font_path, template.font_latin_path)
    if template.font_bold_path:
        bold_ethiopic = fitz.Font(fontfile=template.font_bold_path)
        bold_latin_path = template.font_latin_bold_path
        bold_latin = (
            fitz.Font(fontfile=bold_latin_path) if bold_latin_path else bold_ethiopic
        )
        bold = _FontPair(
            ethiopic=bold_ethiopic,
            latin=bold_latin,
            ascender=max(bold_ethiopic.ascender, bold_latin.ascender),
            descender=min(bold_ethiopic.descender, bold_latin.descender),
        )
    else:
        bold = regular
    return _FontSet(regular=regular, bold=bold)


def _iter_runs(text: str, pair: _FontPair) -> list[tuple[str, fitz.Font]]:
    """Split text into contiguous runs that share one font."""
    runs: list[list[Any]] = []
    for char in text:
        font = pair.ethiopic if is_ethiopic(char) else pair.latin
        if runs and runs[-1][1] is font:
            runs[-1][0] += char
        else:
            runs.append([char, font])
    return [(str(chunk), font) for chunk, font in runs]


def _text_width(text: str, pair: _FontPair, size: float) -> float:
    return sum(font.text_length(chunk, size) for chunk, font in _iter_runs(text, pair))


def _append_text(
    writer: fitz.TextWriter,
    x: float,
    baseline: float,
    text: str,
    pair: _FontPair,
    size: float,
) -> float:
    """Append a mixed-script string and return the new x position."""
    cx = x
    for chunk, font in _iter_runs(text, pair):
        writer.append((cx, baseline), chunk, font, size)
        cx += font.text_length(chunk, size)
    return cx


@dataclass
class _Line:
    words: list[str]
    width: float
    indent: float


def _break_word(
    word: str, pair: _FontPair, size: float, max_width: float
) -> list[str]:
    """Split a word that is wider than the column, character by character."""
    if _text_width(word, pair, size) <= max_width or max_width <= 0:
        return [word]
    chunks: list[str] = []
    current = ""
    for char in word:
        candidate = current + char
        if current and _text_width(candidate, pair, size) > max_width:
            chunks.append(current)
            current = char
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [word]


def _wrap_paragraph(
    text: str,
    pair: _FontPair,
    size: float,
    width: float,
    first_indent: float,
) -> list[_Line]:
    """Greedy line breaking with a first-line indent."""
    words = text.split()
    if not words:
        return []

    space_width = _text_width(" ", pair, size)
    lines: list[_Line] = []
    current: list[str] = []
    current_width = 0.0
    indent = first_indent
    avail = max(width - indent, size * 2)

    for word in words:
        word_width = _text_width(word, pair, size)
        needed = word_width if not current else current_width + space_width + word_width

        if needed <= avail:
            current.append(word)
            current_width = needed
            continue

        if current:
            lines.append(_Line(current, current_width, indent))
            current, current_width = [], 0.0
            indent, avail = 0.0, width

        if word_width > avail:
            chunks = _break_word(word, pair, size, avail)
            for chunk in chunks[:-1]:
                lines.append(_Line([chunk], _text_width(chunk, pair, size), indent))
                indent = 0.0
            current = [chunks[-1]]
            current_width = _text_width(chunks[-1], pair, size)
        else:
            current, current_width = [word], word_width

    if current:
        lines.append(_Line(current, current_width, indent))
    return lines


def _line_baseline(pair: _FontPair, size: float, line_spacing: float, top: float) -> float:
    """Baseline for a line box whose top edge is *top*."""
    natural = (pair.ascender - pair.descender) * size
    leading = max(0.0, size * line_spacing - natural)
    return top + leading / 2 + pair.ascender * size


class _PdfLayout:
    """Paginates reflowed blocks into a warm-paper text PDF."""

    def __init__(self, doc: fitz.Document, template: BookTemplate, fonts: _FontSet):
        self.doc = doc
        self.t = template
        self.fonts = fonts
        self.page: fitz.Page | None = None
        self.writer: fitz.TextWriter | None = None
        self.y = 0.0
        self.toc: list[list[Any]] = []

    # -- page lifecycle ---------------------------------------------------
    def new_page(self, furniture: bool = True) -> None:
        self.flush()
        self.page = self.doc.new_page(
            width=self.t.page_width_pt, height=self.t.page_height_pt
        )
        self.page.draw_rect(
            self.page.rect, color=None, fill=self.t.page_bg_rgb, overlay=False
        )
        self.writer = fitz.TextWriter(self.page.rect)
        self.y = self.t.margin_top_pt
        if furniture:
            self._draw_footer()
            if self.t.include_running_header and len(self.doc) > 1:
                self._draw_header()

    def flush(self) -> None:
        if self.writer is not None and self.page is not None:
            self.writer.write_text(self.page, color=self.t.text_color_rgb)
            self.writer = None

    def _draw_furniture_line(self, text: str, baseline: float, size: float) -> None:
        assert self.page is not None
        writer = fitz.TextWriter(self.page.rect)
        width = _text_width(text, self.fonts.regular, size)
        x = self.t.margin_left_pt + (self.t.content_width - width) / 2
        _append_text(writer, x, baseline, text, self.fonts.regular, size)
        writer.write_text(self.page, color=self.t.secondary_color_rgb)

    def _draw_footer(self) -> None:
        if not self.t.include_page_numbers:
            return
        baseline = self.t.page_height_pt - self.t.margin_bottom_pt * 0.5
        self._draw_furniture_line(str(len(self.doc)), baseline, self.t.footer_font_size)

    def _draw_header(self) -> None:
        baseline = self.t.margin_top_pt * 0.55
        self._draw_furniture_line(self.t.title, baseline, self.t.header_font_size)

    # -- content ----------------------------------------------------------
    def _ensure_space(self, height: float) -> None:
        if self.page is None or self.writer is None:
            self.new_page()
            return
        bottom = self.t.page_height_pt - self.t.margin_bottom_pt
        if self.y + height > bottom:
            self.new_page()

    def draw_heading(self, text: str) -> None:
        size = self.t.chapter_font_size
        line_height = size * 1.25
        # Keep a heading with at least a couple of body lines.
        self._ensure_space(line_height + self.t.font_size * self.t.line_spacing * 2)
        self.y += size * 0.4

        for index, line in enumerate(
            _wrap_paragraph(text, self.fonts.bold, size, self.t.content_width, 0.0)
        ):
            self._ensure_space(line_height)
            assert self.writer is not None
            baseline = _line_baseline(self.fonts.bold, size, 1.25, self.y)
            x = self.t.margin_left_pt + (self.t.content_width - line.width) / 2
            _append_text(self.writer, x, baseline, " ".join(line.words), self.fonts.bold, size)
            self.y += line_height
            if index == 0:
                self._record_toc(text)

        self.y += size * 0.5

    def _record_toc(self, text: str) -> None:
        page_no = len(self.doc)
        if self.toc and self.toc[-1][2] == page_no and self.toc[-1][1] == text:
            return
        self.toc.append([1, text, page_no])

    def draw_paragraph(self, text: str, indent: bool = True) -> None:
        size = self.t.font_size
        line_height = size * self.t.line_spacing
        first_indent = self.t.paragraph_indent_em * size if indent else 0.0
        lines = _wrap_paragraph(
            text, self.fonts.regular, size, self.t.content_width, first_indent
        )
        justify = self.t.body_align == "justify"

        for index, line in enumerate(lines):
            self._ensure_space(line_height)
            assert self.writer is not None
            baseline = _line_baseline(self.fonts.regular, size, self.t.line_spacing, self.y)

            extra = 0.0
            if justify and index < len(lines) - 1 and len(line.words) > 1:
                extra = max(
                    0.0, (self.t.content_width - line.indent - line.width) / (len(line.words) - 1)
                )

            x = self.t.margin_left_pt + line.indent
            for word_index, word in enumerate(line.words):
                # Include the trailing space glyph: positional gaps alone are
                # invisible to text extraction, which breaks copy/paste and
                # full-text search.
                last = word_index == len(line.words) - 1
                chunk = word if last else f"{word} "
                x = _append_text(
                    self.writer, x, baseline, chunk, self.fonts.regular, size
                )
                if not last:
                    x += extra

            self.y += line_height


def _atomic_save(doc: fitz.Document, output_path: str) -> None:
    """Write via a sibling temp file so a failed render never truncates output."""
    tmp_path = f"{output_path}.tmp"
    doc.save(tmp_path, deflate=True, garbage=4, clean=True)
    os.replace(tmp_path, output_path)


def parse_input_source(
    source: list[dict[str, Any]] | str | Path,
) -> tuple[list[dict[str, Any]], str]:
    """
    Parse results from list of dicts, a path to JSON output, or a path to TXT output.
    Returns (page_results, suggested_title).
    """
    if isinstance(source, list):
        return source, ""

    path = str(source)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    suggested_title = Path(path).stem

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data, suggested_title
        if isinstance(data, dict) and "pages" in data:
            return data["pages"], suggested_title
        raise ValueError(f"Unexpected JSON structure in {path}")

    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse === Page X === blocks
        page_blocks = re.split(r"===\s*Page\s+([^\n=]+)\s*===", content, flags=re.IGNORECASE)
        results = []
        if len(page_blocks) > 1:
            # page_blocks[0] is text before first marker
            for i in range(1, len(page_blocks), 2):
                page_id = page_blocks[i].strip()
                page_text = page_blocks[i + 1] if (i + 1) < len(page_blocks) else ""
                results.append({"page": page_id, "text": page_text})
        else:
            # Single text file without page markers
            results.append({"page": "1", "text": content})

        return results, suggested_title

    raise ValueError(f"Unsupported file extension for book rendering: {path}")


class BookRenderer:
    """
    Renders structured OCR pages into text-layer PDF or EPUB.
    """

    def __init__(self, template: BookTemplate | None = None) -> None:
        self.template = template or BookTemplate()

    def _warn_on_missing_glyphs(self, text: str) -> None:
        missing = validate_font_coverage(
            self.template.font_path, text, self.template.font_latin_path
        )
        if missing:
            warnings.warn(
                f"{os.path.basename(self.template.font_path)} has no glyph for "
                f"{len(missing)} character(s): {''.join(missing[:20])}. "
                "They will render as .notdef.",
                RuntimeWarning,
                stacklevel=3,
            )

    def render_pdf(
        self,
        results: list[dict[str, Any]] | str | Path,
        output_path: str,
    ) -> str:
        """
        Render OCR text to a standard text-layer PDF with embedded Ethiopic font.
        """
        pages_data, _ = parse_input_source(results)
        sorted_results = sorted(pages_data, key=lambda x: _page_sort_key(x.get("page", "")))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        if not os.path.exists(self.template.font_path):
            self.template.font_path = find_default_ethiopic_font()

        corpus = "\n".join(str(item.get("text", "")) for item in sorted_results)
        self._warn_on_missing_glyphs(corpus)

        fonts = _load_fonts(self.template)
        doc = fitz.open()
        try:
            layout = _PdfLayout(doc, self.template, fonts)

            for item in sorted_results:
                cleaned = _clean_ocr_text(str(item.get("text", "")))
                if not cleaned:
                    continue

                for block in _reflow_blocks(cleaned):
                    if block.heading:
                        has_content = layout.y > self.template.margin_top_pt + 1
                        if self.template.break_pages_at_markers and has_content:
                            layout.new_page()
                        layout.draw_heading(block.text)
                    else:
                        layout.draw_paragraph(block.text)
                    # Paragraphs keep flowing; only headings may break pages.

            if len(doc) == 0:
                layout.new_page()
                layout.draw_heading(self.template.title)

            layout.flush()

            doc.set_metadata(
                {
                    "title": self.template.title,
                    "author": self.template.author or "",
                    "creator": "Amharic OCR Book Renderer",
                    "producer": "PyMuPDF",
                    "subject": self.template.language,
                }
            )
            if layout.toc:
                doc.set_toc(layout.toc)
            if hasattr(doc, "set_language"):
                doc.set_language(self.template.language)

            _atomic_save(doc, output_path)
            return output_path
        finally:
            doc.close()

    def render_epub(
        self,
        results: list[dict[str, Any]] | str | Path,
        output_path: str,
    ) -> str:
        """
        Render OCR text to an EPUB 3 document with embedded Ethiopic font and novel styling.
        """
        pages_data, _ = parse_input_source(results)
        sorted_results = sorted(pages_data, key=lambda x: _page_sort_key(x.get("page", "")))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        book = epub.EpubBook()
        book.set_identifier(f"urn:uuid:{uuid.uuid4()}")
        book.set_title(self.template.title)
        book.set_language(self.template.language)

        if self.template.author:
            book.add_author(self.template.author)

        # Embed font if available
        font_file = self.template.font_path
        font_css_face = ""
        font_filename = ""
        if os.path.exists(font_file):
            font_filename = os.path.basename(font_file)
            with open(font_file, "rb") as ff:
                font_data = ff.read()
            font_item = epub.EpubItem(
                uid="font_ethiopic",
                file_name=f"fonts/{font_filename}",
                media_type="font/ttf",
                content=font_data,
            )
            book.add_item(font_item)
            font_css_face = f"""
@font-face {{
    font-family: 'EthiopicNovelFont';
    src: url('../fonts/{font_filename}');
}}
"""

        css_content = f"""
{font_css_face}
body {{
    font-family: 'EthiopicNovelFont', 'Noto Sans Ethiopic', 'Abyssinica SIL', serif;
    font-size: 1.05em;
    line-height: {self.template.line_spacing};
    margin: 6%;
    text-align: justify;
}}
h1, h2, h3 {{
    font-family: 'EthiopicNovelFont', 'Noto Sans Ethiopic', serif;
    text-align: center;
    margin-top: 1.6em;
    margin-bottom: 0.8em;
}}
p {{
    margin-top: 0;
    margin-bottom: 0.6em;
    text-indent: 1.5em;
}}
.page-header {{
    font-size: 0.8em;
    color: #666;
    text-align: center;
    border-bottom: 1px solid #ddd;
    padding-bottom: 4px;
    margin-bottom: 1em;
}}
"""
        default_css = epub.EpubItem(
            uid="style_default",
            file_name="style/default.css",
            media_type="text/css",
            content=css_content,
        )
        book.add_item(default_css)

        chapters = []
        for i, item in enumerate(sorted_results, start=1):
            raw_text = item.get("text", "")
            cleaned = _clean_ocr_text(raw_text)
            if not cleaned:
                continue

            paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]
            p_html = "".join(f"<p>{html.escape(p)}</p>\n" for p in paragraphs)

            page_id = item.get("page", str(i))
            chapter_title = f"Page {page_id}"

            c = epub.EpubHtml(
                title=chapter_title,
                file_name=f"page_{i:04d}.xhtml",
                lang=self.template.language,
            )
            c.content = f"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{html.escape(chapter_title)}</title>
    <link rel="stylesheet" href="style/default.css" type="text/css"/>
</head>
<body>
    <div class="page-header">{html.escape(self.template.title)} - {html.escape(chapter_title)}</div>
    {p_html}
</body>
</html>"""
            c.add_item(default_css)
            book.add_item(c)
            chapters.append(c)

        if not chapters:
            # Fallback if no pages
            c = epub.EpubHtml(
                title="Title", file_name="title.xhtml", lang=self.template.language
            )
            c.content = f"<h1>{html.escape(self.template.title)}</h1>"
            c.add_item(default_css)
            book.add_item(c)
            chapters.append(c)

        book.toc = tuple(chapters)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        book.spine = ["nav"] + chapters
        epub.write_epub(output_path, book)
        return output_path


def _safe_slug(title: str, max_bytes: int = 120) -> str:
    """Filesystem-safe slug, truncated by UTF-8 byte length (Amharic is 3B/char)."""
    slug = re.sub(r"[^\w\s-]", "", title).strip()
    slug = re.sub(r"[\s-]+", "_", slug) or "book"
    encoded = slug.encode("utf-8")
    if len(encoded) > max_bytes:
        slug = encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip("_") or "book"
    return slug


def render_book(
    source: list[dict[str, Any]] | str | Path,
    output_dir: str = "rendered",
    format: Literal["pdf", "epub", "both"] = "both",
    template: BookTemplate | None = None,
) -> dict[str, str]:
    """
    Convenience function to render book pages to PDF and/or EPUB.
    Returns a dict mapping format -> output path.
    """
    data, suggested_title = parse_input_source(source)

    if template is None:
        template = (
            BookTemplate.from_filename(suggested_title)
            if suggested_title
            else BookTemplate()
        )
    elif not template.title or template.title == "Untitled Book":
        if suggested_title:
            template.title, auto_author = derive_metadata_from_filename(suggested_title)
            if auto_author and not template.author:
                template.author = auto_author

    renderer = BookRenderer(template)
    os.makedirs(output_dir, exist_ok=True)

    slug = _safe_slug(template.title)

    outputs: dict[str, str] = {}

    if format in ("pdf", "both"):
        pdf_path = os.path.join(output_dir, f"{slug}.pdf")
        renderer.render_pdf(data, pdf_path)
        outputs["pdf"] = pdf_path

    if format in ("epub", "both"):
        epub_path = os.path.join(output_dir, f"{slug}.epub")
        renderer.render_epub(data, epub_path)
        outputs["epub"] = epub_path

    return outputs


def main() -> None:
    """CLI entry point for rendering OCR outputs directly into PDF or EPUB."""
    parser = argparse.ArgumentParser(
        description="Render scanned Amharic OCR text into standard novel PDF and EPUB e-books."
    )
    parser.add_argument(
        "input",
        help="Path to OCR JSON output (*_output.json) or TXT file (*_output.txt).",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default="rendered",
        help="Directory to store rendered PDF/EPUB books (default: rendered)",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "epub", "both"],
        default="both",
        help="Target output format: pdf, epub, or both (default: both)",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Book title (defaults to auto-derived from input filename)",
    )
    parser.add_argument(
        "--author",
        default="",
        help="Book author (defaults to auto-derived from input filename)",
    )
    parser.add_argument(
        "--font",
        default="",
        help="Path to Ethiopic TTF font file (defaults to bundled Noto Serif Ethiopic)",
    )
    parser.add_argument(
        "--page-size",
        choices=sorted(["novel_6x9", "novel_digest", "a5", "a4", "letter"]),
        default="novel_6x9",
        help="Page size preset (default: novel_6x9)",
    )
    parser.add_argument(
        "--paper",
        choices=["cream", "ivory", "sepia", "white"],
        default="cream",
        help="Page background (default: cream; use white for print)",
    )
    parser.add_argument("--font-size", type=float, default=0.0, help="Body font size in pt")
    parser.add_argument(
        "--line-spacing", type=float, default=0.0, help="Line height multiplier"
    )

    args = parser.parse_args()

    tmpl = BookTemplate.from_filename(
        args.input,
        page_size=args.page_size,
    )
    if args.title:
        tmpl.title = args.title
    if args.author:
        tmpl.author = args.author
    if args.font:
        tmpl.font_path = args.font
    if args.font_size > 0:
        tmpl.font_size = args.font_size
    if args.line_spacing > 0:
        tmpl.line_spacing = args.line_spacing

    from books.template import PAPER_COLORS

    tmpl.page_bg_hex = PAPER_COLORS[args.paper]

    print(f"📖 Rendering book: '{tmpl.title}'" + (f" by {tmpl.author}" if tmpl.author else ""))
    print(f"📐 Page size: {tmpl.page_size} ({tmpl.page_width_pt}x{tmpl.page_height_pt} pt)")
    print(f"🔤 Font: {tmpl.font_path} @ {tmpl.font_size} pt / {tmpl.line_spacing} lh")

    outputs = render_book(
        args.input,
        output_dir=args.out_dir,
        format=args.format,
        template=tmpl,
    )

    for fmt, path in outputs.items():
        print(f"✅ Generated {fmt.upper()}: {path}")


if __name__ == "__main__":
    main()
