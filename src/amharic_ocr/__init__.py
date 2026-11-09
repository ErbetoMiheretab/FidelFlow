"""
Amharic OCR package with focused Ethiopic numeral recognition.
"""

from .config import OCRConfig
from .constants import (
    ALL_ETHIOPIC_NUMERALS,
    ALLOWED_EXTRA_CHARS,
    AMHARIC_FIDEL,
    ETHIOPIC_NUMERALS,
    ETHIOPIC_PUNCTUATION,
)
from .diagnostics import (
    check_engine_capabilities,
    check_system_setup,
    test_numeral_recognition,
)
from .engines.tesseract import TesseractEngine
from .evaluation import (
    character_error_rate,
    evaluate_dataset,
    evaluate_pair,
    run_synthetic_evaluation,
    word_error_rate,
)
from .pipeline import (
    process_image_file,
    process_page_image,
    process_single_page,
    run_pipeline,
)
from .postprocessing.corrections import (
    convert_digits_to_ethiopic,
    correct_common_errors,
    deduplicate_numeral_loops,
    extract_ethiopic_numerals,
    find_suspicious_numeral_runs,
    post_process_text,
    post_process_text_with_report,
)

__all__ = [
    "ALLOWED_EXTRA_CHARS",
    "ALL_ETHIOPIC_NUMERALS",
    "AMHARIC_FIDEL",
    "ETHIOPIC_NUMERALS",
    "ETHIOPIC_PUNCTUATION",
    "OCRConfig",
    "TesseractEngine",
    "character_error_rate",
    "check_engine_capabilities",
    "check_system_setup",
    "convert_digits_to_ethiopic",
    "correct_common_errors",
    "deduplicate_numeral_loops",
    "evaluate_dataset",
    "evaluate_pair",
    "extract_ethiopic_numerals",
    "find_suspicious_numeral_runs",
    "post_process_text",
    "post_process_text_with_report",
    "process_image_file",
    "process_page_image",
    "process_single_page",
    "run_pipeline",
    "run_synthetic_evaluation",
    "test_numeral_recognition",
    "word_error_rate",
]
