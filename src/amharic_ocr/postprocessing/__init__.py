"""
Post-processing and conservative corrections.
"""

from .corrections import (
    convert_digits_to_ethiopic,
    correct_common_errors,
    correct_latin_lookalikes,
    correct_visual_confusions,
    deduplicate_numeral_loops,
    extract_ethiopic_numerals,
    find_suspicious_numeral_runs,
    post_process_text,
    post_process_text_with_report,
)

__all__ = [
    "convert_digits_to_ethiopic",
    "correct_common_errors",
    "correct_latin_lookalikes",
    "correct_visual_confusions",
    "deduplicate_numeral_loops",
    "extract_ethiopic_numerals",
    "find_suspicious_numeral_runs",
    "post_process_text",
    "post_process_text_with_report",
]
