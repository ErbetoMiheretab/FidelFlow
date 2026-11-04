"""
Post-processing and conservative corrections for Amharic text and Ethiopic
numerals.

The original OCR text is always kept by the pipeline. Corrections in this
module are context-limited and, where a change is ambiguous, the preferred
behaviour is to leave the text unchanged and add a review flag.
"""

from __future__ import annotations

import re

from ..constants import (
    ALL_ETHIOPIC_NUMERALS,
    ETHIOPIC_NUMERALS,
    LATIN_TO_ETHIOPIC_LOOKALIKES,
    VISUAL_CONFUSIONS,
)

_NUMERAL_SET = set(ALL_ETHIOPIC_NUMERALS)


def _contextual_visual_confusions(text: str) -> tuple[str, int]:
    """Apply visual-confusion substitutions only next to a real numeral."""
    chars = list(text)
    changed = 0
    for index, char in enumerate(chars):
        replacement = VISUAL_CONFUSIONS.get(char)
        if not replacement:
            continue
        left = next((c for c in reversed(chars[max(0, index - 3) : index]) if not c.isspace()), "")
        right = next((c for c in chars[index + 1 : index + 4] if not c.isspace()), "")
        if left in _NUMERAL_SET or right in _NUMERAL_SET:
            chars[index] = replacement
            changed += 1
    return "".join(chars), changed


def correct_visual_confusions(text: str) -> str:
    """
    Fix Ethiopic punctuation/numerals only when adjacent to a real numeral.

    For example, ``፨`` is valid Ethiopic punctuation and is left untouched in
    normal prose. It is only treated as a misread ``፰`` when it sits next to
    other Ethiopic numerals. Ambiguous cases are preserved.
    """
    corrected, _ = _contextual_visual_confusions(text)
    return corrected


def correct_latin_lookalikes(text: str) -> str:
    """
    Repair Latin look-alikes inside numeral zones.

    A numeral zone must already contain at least one genuine Ethiopic numeral.
    Normal Amharic words and English abbreviations are therefore not rewritten.
    """
    latin_lookalikes = "".join(re.escape(c) for c in LATIN_TO_ETHIOPIC_LOOKALIKES)
    numeral_zone_pattern = re.compile(
        rf"[{re.escape(ALL_ETHIOPIC_NUMERALS)}{latin_lookalikes}]*"
        rf"[{re.escape(ALL_ETHIOPIC_NUMERALS)}]"
        rf"[{re.escape(ALL_ETHIOPIC_NUMERALS)}{latin_lookalikes}]*"
    )

    def fix_token(match: re.Match) -> str:
        token = match.group(0)
        for wrong, correct in LATIN_TO_ETHIOPIC_LOOKALIKES.items():
            token = token.replace(wrong, correct)
        return token

    return numeral_zone_pattern.sub(fix_token, text)


def correct_common_errors(text: str) -> str:
    """Backwards-compatible combination of the two contextual corrections."""
    return correct_latin_lookalikes(correct_visual_confusions(text))


def find_suspicious_numeral_runs(text: str, min_run: int = 3) -> list[str]:
    """Return repeated identical numeral runs that may be OCR looping artifacts.

    A run is reported, not automatically changed. Ethiopic numerals can
    legitimately repeat (for example ``፼፼፼``), so deleting characters would
    alter a real value.
    """
    pattern = re.compile(rf"([{re.escape(ALL_ETHIOPIC_NUMERALS)}])\1{{{min_run - 1},}}")
    return [m.group(0) for m in pattern.finditer(text)]


def deduplicate_numeral_loops(text: str) -> str:
    """
    Collapse ``3+`` repeated identical numeral characters to two.

    .. warning::
       This can change a genuine numeral. It is opt-in and is not part of the
       default correction pass; use :func:`find_suspicious_numeral_runs` to
       flag suspicious runs for human review instead.
    """
    dedup_pattern = re.compile(rf"([{re.escape(ALL_ETHIOPIC_NUMERALS)}])\1{{2,}}")
    return dedup_pattern.sub(r"\1\1", text)


# Alias for backwards compatibility
_deduplicate_numeral = deduplicate_numeral_loops


def convert_digits_to_ethiopic(text: str) -> str:
    """
    Convert standalone Latin digit runs to Ethiopic numerals.

    .. warning::
       This is opt-in. It can change dates, page numbers, measurements and
       mixed-script identifiers. The function only touches digit runs that are
       not embedded inside an alphanumeric token.
    """
    sorted_keys = sorted(ETHIOPIC_NUMERALS.keys(), key=len, reverse=True)

    def convert_run(match: re.Match) -> str:
        value = match.group(0)
        for key in sorted_keys:
            value = value.replace(key, ETHIOPIC_NUMERALS[key])
        return value

    return re.sub(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])", convert_run, text)


def extract_ethiopic_numerals(text: str) -> list[str]:
    """Extract all runs of Ethiopic numerals from text."""
    pattern = rf"[{re.escape(ALL_ETHIOPIC_NUMERALS)}]+"
    return re.findall(pattern, text)


# Alias
extract_numerals = extract_ethiopic_numerals


def clean_margin_artifacts(
    text: str,
    strip_edges: bool = False,
) -> tuple[str, int]:
    """
    Remove lines that consist only of scanner noise.

    By default this does **not** strip leading/trailing dashes or punctuation
    because those can be meaningful (lists, dialogue, ranges). Set
    ``strip_edges=True`` to opt into the older aggressive behaviour.

    Returns ``(cleaned_text, removed_line_count)``.
    """
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    removed = 0
    for line in lines:
        stripped = line.strip()
        # Remove only lines that contain actual scanner-noise characters and
        # no letters/digits. A line of ordinary punctuation such as "..." is
        # preserved because it may be meaningful text.
        noise_only = re.fullmatch(r"[|~^\\_`<>{}•\s\-—.:]+", stripped)
        has_scanner_noise = re.search(r"[|~^\\_`<>{}•]", stripped)
        if stripped and noise_only and has_scanner_noise:
            removed += 1
            continue
        if strip_edges:
            stripped = re.sub(r"^[|~^\\_`<>{}—\-•]+\s*", "", stripped)
            stripped = re.sub(r"\s*[|~^\\_`<>{}—\-•]+$", "", stripped)
        cleaned_lines.append(stripped)

    result = "\n".join(cleaned_lines)
    return re.sub(r"\n{3,}", "\n\n", result).strip(), removed


def post_process_text_with_report(
    text: str,
    is_numeral_heavy: bool = False,
    *,
    apply_numeral_corrections: bool = True,
    convert_latin_digits: bool = False,
    deduplicate_numeral_runs: bool = False,
    clean_margin_artifacts_enabled: bool = True,
    strip_margin_noise_edges: bool = False,
) -> dict:
    """
    Apply conservative cleanup and return both cleaned text and an audit trail.

    The returned dictionary contains ``raw_text`` (the original input),
    ``text`` (cleaned), ``changes``, and ``review_flags``.
    """
    original = text
    changes: list[dict] = []
    review_flags: list[str] = []

    corrected, visual_count = _contextual_visual_confusions(text)
    if visual_count:
        changes.append(
            {
                "type": "visual_confusion",
                "count": visual_count,
                "detail": "Substitution only next to an Ethiopic numeral.",
            }
        )
    text = corrected

    if is_numeral_heavy and apply_numeral_corrections:
        before = text
        text = correct_latin_lookalikes(text)
        if text != before:
            changes.append(
                {
                    "type": "latin_lookalike",
                    "count": sum(1 for a, b in zip(before, text, strict=False) if a != b),
                    "detail": "Repair limited to tokens containing an Ethiopic numeral.",
                }
            )

    if is_numeral_heavy and convert_latin_digits:
        before = text
        text = convert_digits_to_ethiopic(text)
        if text != before:
            changes.append(
                {
                    "type": "latin_digits_to_ethiopic",
                    "count": sum(1 for a, b in zip(before, text, strict=False) if a != b),
                    "detail": "Opt-in conversion of standalone digit runs.",
                }
            )

    suspicious_runs = find_suspicious_numeral_runs(text)
    if suspicious_runs:
        review_flags.append(
            "Suspicious repeated Ethiopic numeral run(s): "
            + ", ".join(repr(run) for run in suspicious_runs[:5])
        )
        if deduplicate_numeral_runs:
            before = text
            text = deduplicate_numeral_loops(text)
            changes.append(
                {
                    "type": "numeral_deduplication",
                    "count": len(suspicious_runs),
                    "detail": "Opt-in; repeated numeral runs were shortened.",
                }
            )
            if text == before:
                review_flags.append("Numeral de-duplication was requested but made no change.")

    if clean_margin_artifacts_enabled:
        text, removed_lines = clean_margin_artifacts(
            text, strip_edges=strip_margin_noise_edges
        )
        if removed_lines:
            changes.append(
                {
                    "type": "noise_only_lines_removed",
                    "count": removed_lines,
                    "detail": "Lines containing only scanner-noise characters were removed.",
                }
            )

    return {
        "raw_text": original,
        "text": text,
        "changes": changes,
        "review_flags": review_flags,
    }


def post_process_text(
    text: str,
    is_numeral_heavy: bool = False,
    *,
    apply_numeral_corrections: bool = True,
    convert_latin_digits: bool = False,
    deduplicate_numeral_runs: bool = False,
    clean_margin_artifacts_enabled: bool = True,
    strip_margin_noise_edges: bool = False,
) -> str:
    """Backwards-compatible wrapper returning only the cleaned text."""
    return post_process_text_with_report(
        text,
        is_numeral_heavy=is_numeral_heavy,
        apply_numeral_corrections=apply_numeral_corrections,
        convert_latin_digits=convert_latin_digits,
        deduplicate_numeral_runs=deduplicate_numeral_runs,
        clean_margin_artifacts_enabled=clean_margin_artifacts_enabled,
        strip_margin_noise_edges=strip_margin_noise_edges,
    )["text"]
