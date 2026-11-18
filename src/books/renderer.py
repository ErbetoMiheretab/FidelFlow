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
from pathlib import Path
from typing import Any, Literal

import fitz  # PyMuPDF
from ebooklib import epub

from books.template import (
    BookTemplate,
    derive_metadata_from_filename,
    find_default_ethiopic_font,
)


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
        doc = fitz.open()

        font_file = self.template.font_path
        if not os.path.exists(font_file):
            font_file = find_default_ethiopic_font()

        font_registered_name = "ethiopic_body"
        # Ethiopic fonts have no Arabic digits, so page numbers use the Latin
        # companion face.
        latin_font_file = self.template.font_latin_path or font_file

        text_rect = fitz.Rect(*self.template.text_rect)
        line_height = self.template.line_spacing

        # Collect page content
        for item in sorted_results:
            raw_text = item.get("text", "")
            cleaned = _clean_ocr_text(raw_text)
            if not cleaned:
                continue

            paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]
            if not paragraphs:
                continue

            p_idx = 0
            while p_idx < len(paragraphs):
                page = doc.new_page(
                    width=self.template.page_width_pt,
                    height=self.template.page_height_pt,
                )
                page.insert_font(fontname=font_registered_name, fontfile=font_file)

                # Try putting all remaining paragraphs of this book page
                candidate_text = "\n\n".join(paragraphs[p_idx:])
                rc = page.insert_textbox(
                    text_rect,
                    candidate_text,
                    fontname=font_registered_name,
                    fontsize=self.template.font_size,
                    lineheight=line_height,
                )

                if rc >= 0:
                    # Fits on this single page
                    break

                # Overflow: remove page and binary search to find maximum fitting paragraphs
                doc.delete_page(len(doc) - 1)

                low = 1
                high = len(paragraphs) - p_idx
                best = 1
                while low <= high:
                    mid = (low + high) // 2
                    test_page = doc.new_page(
                        width=self.template.page_width_pt,
                        height=self.template.page_height_pt,
                    )
                    test_page.insert_font(
                        fontname=font_registered_name, fontfile=font_file
                    )
                    test_text = "\n\n".join(paragraphs[p_idx : p_idx + mid])
                    rc = test_page.insert_textbox(
                        text_rect,
                        test_text,
                        fontname=font_registered_name,
                        fontsize=self.template.font_size,
                        lineheight=line_height,
                    )
                    doc.delete_page(len(doc) - 1)
                    if rc >= 0:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1

                # Insert the best fitting slice
                fit_page = doc.new_page(
                    width=self.template.page_width_pt,
                    height=self.template.page_height_pt,
                )
                fit_page.insert_font(
                    fontname=font_registered_name, fontfile=font_file
                )
                fit_text = "\n\n".join(paragraphs[p_idx : p_idx + best])
                fit_page.insert_textbox(
                    text_rect,
                    fit_text,
                    fontname=font_registered_name,
                    fontsize=self.template.font_size,
                    lineheight=line_height,
                )
                p_idx += best

        # If document is empty (no text extracted), create at least one placeholder page
        if len(doc) == 0:
            p = doc.new_page(
                width=self.template.page_width_pt,
                height=self.template.page_height_pt,
            )
            p.insert_font(fontname=font_registered_name, fontfile=font_file)
            p.insert_textbox(
                text_rect,
                self.template.title,
                fontname=font_registered_name,
                fontsize=self.template.font_size,
            )

        # Apply running headers and footer page numbers
        footer_top = self.template.page_height_pt - self.template.margin_bottom_pt + 8
        footer_bottom = self.template.page_height_pt - 12
        footer_rect = fitz.Rect(
            self.template.margin_left_pt,
            footer_top,
            self.template.page_width_pt - self.template.margin_right_pt,
            footer_bottom,
        )

        header_top = 15
        header_bottom = self.template.margin_top_pt - 8
        header_rect = fitz.Rect(
            self.template.margin_left_pt,
            header_top,
            self.template.page_width_pt - self.template.margin_right_pt,
            header_bottom,
        )

        total_pages = len(doc)
        for page_idx, page in enumerate(doc, start=1):
            page.insert_font(fontname=font_registered_name, fontfile=font_file)
            page.insert_font(fontname="latin_body", fontfile=latin_font_file)

            if self.template.include_page_numbers:
                num_str = str(page_idx)
                page.insert_textbox(
                    footer_rect,
                    num_str,
                    fontname="latin_body",
                    fontsize=self.template.footer_font_size,
                    color=self.template.secondary_color_rgb,
                    align=fitz.TEXT_ALIGN_CENTER,
                )

            if self.template.include_running_header and page_idx > 1:
                page.insert_textbox(
                    header_rect,
                    self.template.title,
                    fontname=font_registered_name,
                    fontsize=self.template.header_font_size,
                    align=fitz.TEXT_ALIGN_CENTER,
                )

        # Set document metadata
        doc.set_metadata(
            {
                "title": self.template.title,
                "author": self.template.author or "Unknown",
                "creator": "Amharic OCR Book Renderer",
                "producer": "PyMuPDF",
            }
        )

        doc.save(output_path, deflate=True)
        doc.close()
        return output_path

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

    slug = re.sub(r"[^\w\s-]", "", template.title).strip()
    slug = re.sub(r"[\s-]+", "_", slug) or "book"

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
        help="Path to Ethiopic TTF font file (defaults to auto-detected system font)",
    )
    parser.add_argument(
        "--page-size",
        choices=["novel_6x9", "novel_digest", "a5", "a4"],
        default="novel_6x9",
        help="Page size preset (default: novel_6x9)",
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

    print(f"📖 Rendering book: '{tmpl.title}'" + (f" by {tmpl.author}" if tmpl.author else ""))
    print(f"📐 Page size: {tmpl.page_size} ({tmpl.page_width_pt}x{tmpl.page_height_pt} pt)")
    print(f"🔤 Font: {tmpl.font_path}")

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
