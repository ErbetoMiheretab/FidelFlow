"""Fast checks that do not require Tesseract or a scanned image."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from amharic_ocr.config import OCRConfig
from amharic_ocr.engines.base import OCRResult
from amharic_ocr.evaluation import (
    character_error_rate,
    discover_evaluation_pairs,
    ethiopic_numeral_accuracy,
    word_error_rate,
)
from amharic_ocr.fusion.combiner import score_ocr_result
from amharic_ocr.postprocessing.corrections import (
    convert_digits_to_ethiopic,
    correct_visual_confusions,
    post_process_text_with_report,
)
from amharic_ocr.preprocessing.page_split import detect_page_gutter
from amharic_ocr.preprocessing.variants import build_preprocessing_variants


class MetricTests(unittest.TestCase):
    def test_cer_and_wer(self):
        self.assertEqual(character_error_rate("ሰላም", "ሰላም"), 0.0)
        self.assertAlmostEqual(character_error_rate("abc", "abd"), 1 / 3, places=6)
        self.assertAlmostEqual(word_error_rate("a b", "a c"), 0.5, places=6)

    def test_numeral_accuracy(self):
        result = ethiopic_numeral_accuracy("፩፪ ፫", "፩፪ ፫")
        self.assertIsNotNone(result)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertTrue(result["exact_run_match"])
        self.assertIsNone(ethiopic_numeral_accuracy("no numerals", "፩"))


class CorrectionTests(unittest.TestCase):
    def test_paragraph_separator_preserved_in_prose(self):
        self.assertEqual(correct_visual_confusions("ሰላም፨ዓለም"), "ሰላም፨ዓለም")

    def test_visual_confusion_only_in_numeral_context(self):
        self.assertEqual(correct_visual_confusions("፩፨፪"), "፩፰፪")

    def test_numeral_deduplication_is_flagged_not_default(self):
        report = post_process_text_with_report("፩፩፩", is_numeral_heavy=True)
        self.assertEqual(report["text"], "፩፩፩")
        self.assertTrue(any("Suspicious" in flag for flag in report["review_flags"]))

    def test_digit_conversion_only_standalone(self):
        self.assertEqual(
            convert_digits_to_ethiopic("abc123 456"),
            "abc123 ፬፭፮",
        )

    def test_noise_only_lines_removed(self):
        report = post_process_text_with_report("kept text\n|||`\nmore text")
        self.assertIn("kept text", report["text"])
        self.assertIn("more text", report["text"])
        self.assertNotIn("|||`", report["text"])

    def test_punctuation_only_line_preserved(self):
        report = post_process_text_with_report("first\n...\nsecond")
        self.assertIn("...", report["text"])


class SplitTests(unittest.TestCase):
    def test_off_center_gutter_is_used(self):
        image = np.full((400, 800), 255, np.uint8)
        for y in range(60, 340, 40):
            cv2.line(image, (40, y), (300, y), 0, 3)
            cv2.line(image, (540, y), (760, y), 0, 3)
        info = detect_page_gutter(image)
        self.assertTrue(info["possible_facing_pages"])
        self.assertIsNotNone(info["split_x"])
        self.assertLess(abs(info["split_x"] - 420), 45)

    def test_portrait_page_not_auto_split(self):
        image = np.full((1000, 700), 255, np.uint8)
        for y in range(80, 920, 45):
            cv2.line(image, (60, y), (640, y), 0, 3)
        info = detect_page_gutter(image)
        self.assertFalse(info["possible_facing_pages"])
        self.assertIsNone(info["split_x"])

    def test_photographed_spread_with_dark_binding_and_blank_margins(self):
        """A portrait scan can contain a landscape spread in its middle."""
        image = np.full((1200, 900), 255, np.uint8)
        for y in range(280, 820, 40):
            cv2.line(image, (60, y), (400, y), 0, 3)
            cv2.line(image, (500, y), (840, y), 0, 3)
        # Binding shadow instead of a clean white gutter.
        image[250:850, 430:470] = 40
        # Scanner watermark, well below the book content.
        cv2.rectangle(image, (700, 1080), (860, 1095), 0, -1)

        info = detect_page_gutter(image)
        self.assertTrue(info["possible_facing_pages"])
        self.assertEqual(info["method"], "binding_shadow")
        self.assertGreater(info["aspect_ratio"], 1.05)
        self.assertLess(abs(info["split_x"] - 450), 35)


class VariantTests(unittest.TestCase):
    def test_default_variants_include_raw_baseline(self):
        image = np.full((80, 120, 3), 255, np.uint8)
        variants = build_preprocessing_variants(image, OCRConfig())
        names = [variant.name for variant in variants]
        self.assertEqual(names, ["raw", "normalized", "local"])
        self.assertEqual(variants[0].image.shape, (80, 120))
        self.assertEqual(variants[0].transformations, [])


class LazyImportTests(unittest.TestCase):
    def test_easyocr_and_torch_are_not_imported_at_startup(self):
        """The optional EasyOCR engine must not pull in torch on every CLI run."""
        code = (
            "import sys; import amharic_ocr.engines; "
            "print(int('easyocr' in sys.modules or 'torch' in sys.modules))"
        )
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "0")


class ScoringTests(unittest.TestCase):
    def test_garbage_does_not_beat_clean_amharic(self):
        clean = OCRResult("የአማርኛ ቋንቋ ትምህርት", 0.8, "test")
        garbage = OCRResult("|||||~~~^^^____", 0.8, "test")
        self.assertGreater(score_ocr_result(clean), score_ocr_result(garbage))


class DiscoveryTests(unittest.TestCase):
    def test_directory_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "page_001.png")
            image = np.full((20, 20), 255, np.uint8)
            cv2.imwrite(image_path, image)
            with open(
                os.path.join(directory, "page_001.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("ሰላም")
            pairs, warnings = discover_evaluation_pairs(directory)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["reference"], "ሰላም")
            self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
