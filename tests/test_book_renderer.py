"""Tests for the BookTemplate and BookRenderer modules."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile

import fitz  # PyMuPDF
from ebooklib import epub

from books.renderer import (
    BookRenderer,
    _reflow_blocks,
    parse_input_source,
    render_book,
    safe_slug,
)
from books.template import (
    PAPER_COLORS,
    BookTemplate,
    derive_metadata_from_filename,
    find_default_ethiopic_font,
    validate_font_coverage,
)


class TemplateMetadataTests(unittest.TestCase):
    def test_derive_metadata_with_author(self):
        title, author = derive_metadata_from_filename("ሞገድ - ኸርቪንግ ዋላስ.pdf")
        self.assertEqual(title, "ሞገድ")
        self.assertEqual(author, "ኸርቪንግ ዋላስ")

    def test_derive_metadata_strips_telegram_tag(self):
        title, author = derive_metadata_from_filename("ራማቶሓራ 2 @ETHIO_PDF_BOOKS.pdf")
        self.assertEqual(title, "ራማቶሓራ 2")
        self.assertEqual(author, "")

    def test_derive_metadata_hyphen_and_tag(self):
        title, author = derive_metadata_from_filename("የቀንድ አውጣ ኑሮ - @ETHIO_PDF_BOOKS.pdf")
        self.assertEqual(title, "የቀንድ አውጣ ኑሮ")
        self.assertEqual(author, "")

    def test_derive_metadata_plain_title(self):
        title, author = derive_metadata_from_filename("ፍቅር እስከ መቃብር.pdf")
        self.assertEqual(title, "ፍቅር እስከ መቃብር")
        self.assertEqual(author, "")

    def test_template_novel_dimensions(self):
        template = BookTemplate(page_size="novel_6x9")
        self.assertEqual(template.page_width_pt, 432.0)
        self.assertEqual(template.page_height_pt, 648.0)
        self.assertGreater(template.content_width, 300.0)
        self.assertGreater(template.content_height, 500.0)

    def test_template_font_detected(self):
        font_path = find_default_ethiopic_font()
        self.assertTrue(os.path.exists(font_path))


class BookRendererTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sample_results = [
            {
                "page": "1",
                "text": "ምዕራፍ ፩\n\nዝናና መታነቅ\nከቤቴ ፊት ለፊት፤ ረጅም ዛፍ አለ፤\nአንድ ሰው አይጠፋም፤ የተንጠለጠለ፡፡",
            },
            {
                "page": "2",
                "text": "ምዕራፍ ፪\n\nበጠባብ መስኮቴ ብቅ ብሏል አንገቴ፤ ይህን ምክር ላፈልቅ፡፡\nዝና ነጭ ሻሽ ነው በስህተት ይጎድፋል፡፡",
            },
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_render_pdf(self):
        out_pdf = os.path.join(self.temp_dir.name, "test_novel.pdf")
        tmpl = BookTemplate(title="የፈተና መጽሐፍ", author="ፈታኝ", page_size="novel_6x9")
        renderer = BookRenderer(tmpl)

        result_path = renderer.render_pdf(self.sample_results, out_pdf)
        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 1000)

        # Inspect generated PDF
        doc = fitz.open(result_path)
        self.assertGreaterEqual(len(doc), 2)
        # Verify page dimensions are 6x9 (432x648)
        self.assertEqual(doc[0].rect.width, 432.0)
        self.assertEqual(doc[0].rect.height, 648.0)

        # First page is the title page; body starts on page 2.
        self.assertIn("የፈተና መጽሐፍ", doc[0].get_text())

        # Verify extracted text contains Amharic Fidel
        extracted = doc[1].get_text()
        self.assertIn("ምዕራፍ", extracted)
        self.assertIn("1", extracted)  # Body numbering starts at 1

    def test_render_epub(self):
        out_epub = os.path.join(self.temp_dir.name, "test_novel.epub")
        tmpl = BookTemplate(title="የፈተና መጽሐፍ", author="ፈታኝ")
        renderer = BookRenderer(tmpl)

        result_path = renderer.render_epub(self.sample_results, out_epub)
        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 1000)

        # Inspect generated EPUB
        book = epub.read_epub(result_path)
        metadata_titles = book.get_metadata("DC", "title")
        self.assertTrue(any("የፈተና መጽሐፍ" in t[0] for t in metadata_titles))

    def test_render_from_json_file(self):
        json_path = os.path.join(self.temp_dir.name, "sample_output.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.sample_results, f, ensure_ascii=False)

        out_dir = os.path.join(self.temp_dir.name, "rendered_out")
        outputs = render_book(json_path, output_dir=out_dir, format="both")
        self.assertIn("pdf", outputs)
        self.assertIn("epub", outputs)
        self.assertTrue(os.path.exists(outputs["pdf"]))
        self.assertTrue(os.path.exists(outputs["epub"]))

    def test_render_from_txt_file(self):
        txt_path = os.path.join(self.temp_dir.name, "sample_output.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== Page 1 ===\nሰላም ለሁሉ ይሁን፡፡\n\n=== Page 2 ===\nመልካም ንባብ፡፡\n")

        pages, title = parse_input_source(txt_path)
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["page"], "1")
        self.assertIn("ሰላም", pages[0]["text"])


class ReflowTests(unittest.TestCase):
    """OCR emits one line per visual line; those must become paragraphs."""

    def test_lines_join_into_one_paragraph(self):
        blocks = _reflow_blocks("አንድ መስመር\nሁለተኛ መስመር\nሶስተኛ")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "አንድ መስመር ሁለተኛ መስመር ሶስተኛ")

    def test_blank_line_starts_a_new_paragraph(self):
        blocks = _reflow_blocks("one\ntwo\n\nthree")
        self.assertEqual([block.text for block in blocks], ["one two", "three"])

    def test_chapter_heading_is_its_own_block(self):
        blocks = _reflow_blocks("prose\n\nምዕራፍ ፩\n\nmore prose")
        self.assertEqual(
            [(block.text, block.heading) for block in blocks],
            [("prose", False), ("ምዕራፍ ፩", True), ("more prose", False)],
        )

    def test_ocr_variant_heading_is_recognised(self):
        blocks = _reflow_blocks("ምእራፍ ፪\nbody")
        self.assertTrue(blocks[0].heading)


class TemplateTypographyTests(unittest.TestCase):
    def test_preset_margins_are_proportional(self):
        six_by_nine = BookTemplate(page_size="novel_6x9")
        a4 = BookTemplate(page_size="a4")
        self.assertEqual(six_by_nine.margin_left_pt, 54.0)
        self.assertEqual(a4.margin_left_pt, 72.0)
        self.assertGreater(a4.content_width, six_by_nine.content_width)

    def test_explicit_dimensions_win_over_preset(self):
        template = BookTemplate(
            page_size="novel_6x9",
            page_width_pt=500.0,
            page_height_pt=700.0,
            margin_left_pt=80.0,
        )
        self.assertEqual(template.page_width_pt, 500.0)
        self.assertEqual(template.page_height_pt, 700.0)
        self.assertEqual(template.margin_left_pt, 80.0)

    def test_typography_defaults_are_ethiopic_friendly(self):
        template = BookTemplate()
        self.assertGreaterEqual(template.font_size, 11.5)
        self.assertGreaterEqual(template.line_spacing, 1.45)
        self.assertEqual(template.page_bg_hex, PAPER_COLORS["cream"])

    def test_bundled_font_covers_corpus_characters(self):
        template = BookTemplate()
        missing = validate_font_coverage(
            template.font_path, "ሇዯቬፇ abc 123 ።", template.font_latin_path
        )
        self.assertEqual(missing, [])

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            BookTemplate(page_size="novel_7x10")
        with self.assertRaises(ValueError):
            BookTemplate(body_align="center")
        with self.assertRaises(ValueError):
            BookTemplate(line_spacing=0.8)
        with self.assertRaises(ValueError):
            BookTemplate(font_size=0)


class PdfProductionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _render(self, pages, name="book.pdf", **template_kwargs):
        path = os.path.join(self.temp_dir.name, name)
        template = BookTemplate(title="የፈተና መጽሐፍ", author="ፈታኝ", **template_kwargs)
        BookRenderer(template).render_pdf(pages, path)
        return fitz.open(path)

    def test_ocr_lines_do_not_inflate_the_page_count(self):
        text = "\n".join(["ይህ አንድ የሙከራ መስመር ነው።"] * 30)
        doc = self._render([{"page": "1", "text": text}])
        # One title page + one body page. The old renderer produced ~2 pages
        # for every source page because each OCR line became a paragraph.
        self.assertEqual(len(doc), 2)
        self.assertEqual(doc[0].get_text().strip(), "የፈተና መጽሐፍ\nፈታኝ")

    def test_long_paragraph_is_never_silently_dropped(self):
        paragraph = "የሙከራ ጽሑፍ በዚህ ቦታ ይጻፋል። " * 900
        doc = self._render([{"page": "1", "text": paragraph}])
        self.assertGreater(len(doc), 2)  # genuinely spans several pages

        source = {ch for ch in paragraph if not ch.isspace()}
        rendered = "".join(page.get_text() for page in list(doc)[1:])
        rendered_chars = {ch for ch in rendered if not ch.isspace()}
        # Every distinct character (and therefore every word) survived.
        self.assertTrue(source <= rendered_chars, source - rendered_chars)
        self.assertGreater(len(rendered), len(paragraph) * 0.95)

    def test_no_blank_body_pages(self):
        doc = self._render(
            [{"page": str(i), "text": "ሙሉ ጽሑፍ በየገጹ። " * 40} for i in range(1, 12)]
        )
        blank = [i for i, page in enumerate(doc) if len(page.get_text().strip()) < 5]
        self.assertEqual(blank, [])

    def test_warm_paper_background_and_ink(self):
        doc = self._render([{"page": "1", "text": "ሰላም ዓለም።"}])
        pix = doc[1].get_pixmap()
        self.assertEqual(pix.pixel(4, 4), (251, 247, 239))  # #FBF7EF
        dark_pixels = [
            pix.pixel(x, y)
            for y in range(60, 200, 3)
            for x in range(60, 300, 3)
            if sum(pix.pixel(x, y)) < 200
        ]
        self.assertTrue(dark_pixels)

    def test_title_page_and_body_numbering(self):
        doc = self._render([{"page": "1", "text": "ምዕራፍ ፩\n\nዝናና መታቀቅ።"}])
        self.assertEqual(doc[0].get_label(), "i")
        self.assertEqual(doc[1].get_label(), "1")
        self.assertTrue(doc[1].get_text().startswith("1"))
        self.assertNotIn("1\n", doc[0].get_text())

    def test_chapter_headings_become_bookmarks(self):
        doc = self._render(
            [
                {"page": "1", "text": "ምዕራፍ ፩\n\nየመጀመሪያ ጽሑፍ።"},
                {"page": "2", "text": "ምዕራፍ ፪\n\nሁለተኛ ጽሑፍ።"},
            ]
        )
        toc = doc.get_toc()
        self.assertEqual(len(toc), 2)
        self.assertEqual(toc[0][1], "ምዕራፍ ፩")

    def test_slug_is_capped_by_utf8_bytes(self):
        slug = safe_slug("ሀ" * 200)
        self.assertLessEqual(len(slug.encode("utf-8")), 120)
        self.assertTrue(slug)


class EpubProductionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _render(self, pages, name="book.epub"):
        path = os.path.join(self.temp_dir.name, name)
        BookRenderer(BookTemplate(title="የፈተና መጽሐፍ", author="ፈታኝ")).render_epub(
            pages, path
        )
        return path

    def test_chapters_are_grouped_and_styled(self):
        pages = [
            {"page": str(i), "text": "ይህ ጽሑፍ በየገጹ ይደገማል። " * 30} for i in range(1, 41)
        ]
        path = self._render(pages)
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            chapters = [n for n in names if "/chapter_" in n]
            self.assertEqual(len(chapters), 3)  # 40 pages / 15 per chapter
            self.assertTrue(any("/fonts/" in n for n in names))
            css = archive.read("EPUB/style/default.css").decode()
            self.assertIn("line-height: 1.65", css)
            self.assertIn("#FBF7EF", css)
            self.assertNotIn("margin: 6%", css)
            body = archive.read(chapters[0]).decode()
            self.assertNotIn("page-header", body)
            self.assertIn("<p", body)

    def test_headings_drive_chapters(self):
        pages = [
            {"page": "1", "text": "መግቢያ ጽሑፍ።"},
            {"page": "2", "text": "ምዕራፍ ፩\n\nየመጀመሪያ ጽሑፍ።"},
            {"page": "3", "text": "ምዕራፍ ፪\n\nሁለተኛ ጽሑፍ።"},
        ]
        path = self._render(pages)
        book = epub.read_epub(path)
        self.assertEqual(len(book.toc), 3)


class ErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            render_book([], format="docx")

    def test_invalid_json_structure(self):
        path = os.path.join(self.temp_dir.name, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"foo": 1}, handle)
        with self.assertRaises(ValueError):
            render_book(path)

    def test_non_string_page_text(self):
        with self.assertRaises(ValueError):
            render_book([{"page": "1", "text": 5}])

    def test_render_book_does_not_mutate_the_template(self):
        template = BookTemplate(title="Untitled Book", author="")
        render_book(
            [{"page": "1", "text": "ሰላም።"}],
            output_dir=self.temp_dir.name,
            format="pdf",
            template=template,
        )
        self.assertEqual(template.title, "Untitled Book")
        self.assertEqual(template.author, "")


if __name__ == "__main__":
    unittest.main()
