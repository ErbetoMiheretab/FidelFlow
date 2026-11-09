"""
Lightweight evaluation for Amharic OCR.

Two modes are supported:

* A small user-supplied set of image/transcription pairs. These are real
  examples (or whatever the user supplies) and are labelled as such.
* A synthetic sample rendered with an installed Ethiopic font. Synthetic
  results only check character coverage and catch regressions; they are not a
  prediction of accuracy on scanned books.

No external dataset is downloaded, bundled or redistributed.
"""

from __future__ import annotations

import csv
import json
import os
import unicodedata
from dataclasses import replace
from typing import Any

from .config import OCRConfig
from .constants import AMHARIC_FIDEL
from .pipeline import process_image_file
from .postprocessing.corrections import extract_ethiopic_numerals

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
REFERENCE_SUFFIXES = (".txt", ".gt.txt", ".reference.txt")


def normalize_text(text: str, collapse_whitespace: bool = True) -> str:
    """Normalise Unicode and whitespace for metric comparison."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if collapse_whitespace:
        text = " ".join(text.split())
    return text.strip()


def levenshtein_distance(reference: str | list, hypothesis: str | list) -> int:
    """Standard edit distance with O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate after whitespace/Unicode normalisation."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_distance(ref, hyp) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate using whitespace-delimited tokens."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_distance(ref, hyp) / len(ref)


def ethiopic_numeral_accuracy(reference: str, hypothesis: str) -> dict | None:
    """
    Compare Ethiopic numeral runs.

    Returns ``None`` when the reference contains no Ethiopic numerals. The
    main accuracy is a character-level edit accuracy over numeral runs; an
    exact-run-match indicator is also reported.
    """
    ref_runs = extract_ethiopic_numerals(normalize_text(reference))
    hyp_runs = extract_ethiopic_numerals(normalize_text(hypothesis))
    if not ref_runs:
        return None

    ref_value = " ".join(ref_runs)
    hyp_value = " ".join(hyp_runs)
    distance = levenshtein_distance(ref_value, hyp_value)
    accuracy = max(0.0, 1.0 - distance / max(len(ref_value), 1))
    return {
        "accuracy": round(float(accuracy), 6),
        "exact_run_match": ref_runs == hyp_runs,
        "reference_runs": ref_runs,
        "hypothesis_runs": hyp_runs,
        "reference_numeral_chars": len(ref_value),
    }


def evaluate_pair(
    reference: str,
    hypothesis: str,
    sample_id: str = "",
    source: str = "user_supplied",
) -> dict:
    """Calculate metrics for one reference/hypothesis pair."""
    cer = character_error_rate(reference, hypothesis)
    wer = word_error_rate(reference, hypothesis)
    numeral = ethiopic_numeral_accuracy(reference, hypothesis)
    return {
        "id": sample_id,
        "source": source,
        "cer": round(float(cer), 6),
        "wer": round(float(wer), 6),
        "numeral_accuracy": numeral["accuracy"] if numeral else None,
        "numeral_exact_match": numeral["exact_run_match"] if numeral else None,
        "numeral_details": numeral,
        "reference_chars": len(normalize_text(reference)),
        "hypothesis_chars": len(normalize_text(hypothesis)),
    }


def _reference_for_image(image_path: str) -> str | None:
    stem, _ = os.path.splitext(image_path)
    for suffix in REFERENCE_SUFFIXES:
        candidate = stem + suffix
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8-sig") as f:
                return f.read()
    return None


def discover_evaluation_pairs(path: str) -> tuple[list[dict], list[str]]:
    """
    Discover image/transcription pairs.

    Supported layouts:
    - A directory containing ``page.png`` and ``page.txt`` (or ``page.gt.txt``).
    - A JSON manifest: a list of objects with ``image`` and either
      ``reference`` (path) or ``text`` (inline reference).
    - A CSV manifest with an ``image`` column and ``reference`` or ``text``.

    Returns ``(pairs, warnings)``.
    """
    pairs: list[dict] = []
    warnings: list[str] = []

    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            image_path = os.path.join(path, name)
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            reference = _reference_for_image(image_path)
            if reference is None:
                warnings.append(f"No transcription found for {name}; skipped.")
                continue
            pairs.append({"id": os.path.splitext(name)[0], "image": image_path, "reference": reference})
        return pairs, warnings

    extension = os.path.splitext(path)[1].lower()
    base_dir = os.path.dirname(os.path.abspath(path))

    def resolve(value: str) -> str:
        return value if os.path.isabs(value) else os.path.join(base_dir, value)

    if extension == ".json":
        with open(path, encoding="utf-8-sig") as f:
            manifest = json.load(f)
        items = manifest.get("samples", []) if isinstance(manifest, dict) else manifest
        for index, item in enumerate(items):
            image = resolve(str(item["image"]))
            if "text" in item:
                reference = str(item["text"])
            elif "reference" in item:
                with open(resolve(str(item["reference"])), encoding="utf-8-sig") as f:
                    reference = f.read()
            else:
                warnings.append(f"Manifest item {index} has no text/reference; skipped.")
                continue
            pairs.append(
                {
                    "id": str(item.get("id", os.path.splitext(os.path.basename(image))[0])),
                    "image": image,
                    "reference": reference,
                }
            )
        return pairs, warnings

    if extension == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            for index, row in enumerate(csv.DictReader(f)):
                if not row.get("image"):
                    warnings.append(f"CSV row {index + 2} has no image; skipped.")
                    continue
                image = resolve(row["image"])
                if row.get("text"):
                    reference = row["text"]
                elif row.get("reference"):
                    with open(resolve(row["reference"]), encoding="utf-8-sig") as ref_file:
                        reference = ref_file.read()
                else:
                    warnings.append(f"CSV row {index + 2} has no text/reference; skipped.")
                    continue
                pairs.append(
                    {
                        "id": str(row.get("id", os.path.splitext(os.path.basename(image))[0])),
                        "image": image,
                        "reference": reference,
                    }
                )
        return pairs, warnings

    if extension in IMAGE_EXTENSIONS:
        reference = _reference_for_image(path)
        if reference is None:
            return [], [f"No transcription found for {path}."]
        return [
            {
                "id": os.path.splitext(os.path.basename(path))[0],
                "image": path,
                "reference": reference,
            }
        ], warnings

    return [], [f"Unsupported evaluation path: {path}"]


def evaluate_dataset(
    dataset_path: str,
    config: OCRConfig | None = None,
    max_samples: int | None = None,
) -> dict:
    """Run OCR on a small user-supplied evaluation set and report metrics."""
    if config is None:
        config = OCRConfig()
    pairs, warnings = discover_evaluation_pairs(dataset_path)
    if max_samples is not None:
        pairs = pairs[: max(0, int(max_samples))]

    samples = []
    for pair in pairs:
        try:
            results = process_image_file(pair["image"], config, page_id=pair["id"])
            hypothesis = "\n".join(r.get("text", "") for r in results)
            raw_hypothesis = "\n".join(r.get("raw_text", "") for r in results)
            metrics = evaluate_pair(
                pair["reference"],
                hypothesis,
                sample_id=pair["id"],
                source="user_supplied_scan",
            )
            metrics["raw_cer"] = round(character_error_rate(pair["reference"], raw_hypothesis), 6)
            metrics["raw_wer"] = round(word_error_rate(pair["reference"], raw_hypothesis), 6)
            metrics["hypothesis"] = hypothesis
            metrics["raw_hypothesis"] = raw_hypothesis
            metrics["images"] = [pair["image"]]
            metrics["pages"] = [r.get("page") for r in results]
            metrics["flagged_pages"] = [
                r.get("page") for r in results if r.get("flagged")
            ]
            samples.append(metrics)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Failed to evaluate {pair['id']}: {e}")

    valid = [s for s in samples if s.get("cer") is not None]
    numeral_samples = [s for s in valid if s.get("numeral_accuracy") is not None]

    aggregate = {
        "samples": len(valid),
        "mean_cer": round(sum(s["cer"] for s in valid) / len(valid), 6) if valid else None,
        "mean_wer": round(sum(s["wer"] for s in valid) / len(valid), 6) if valid else None,
        "mean_raw_cer": round(
            sum(s["raw_cer"] for s in valid) / len(valid), 6
        )
        if valid
        else None,
        "mean_raw_wer": round(
            sum(s["raw_wer"] for s in valid) / len(valid), 6
        )
        if valid
        else None,
        "numeral_samples": len(numeral_samples),
        "mean_numeral_accuracy": round(
            sum(s["numeral_accuracy"] for s in numeral_samples) / len(numeral_samples), 6
        )
        if numeral_samples
        else None,
    }

    return {
        "result_kind": "user_supplied_evaluation_set",
        "synthetic": False,
        "caveat": (
            "Metrics describe only the supplied examples. A small set, or a set "
            "from one book/font/layout, does not represent all Ethiopic books."
        ),
        "dataset_path": dataset_path,
        "aggregate": aggregate,
        "samples": samples,
        "warnings": warnings,
    }


def _font_has_ethiopic_glyphs(font_path: str) -> bool:
    """Weak local check: missing-glyph boxes are identical for different letters."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(font_path, 32)

        def render(character: str) -> bytes:
            image = Image.new("L", (64, 64), color=255)
            ImageDraw.Draw(image).text((4, 4), character, fill=0, font=font)
            return image.tobytes()

        first = render("ሀ")
        second = render("ለ")
        return bool(first) and bool(second) and first != second
    except Exception:  # noqa: BLE001
        return False


def find_ethiopic_font() -> str | None:
    """Find a locally installed font that can render Ethiopic text."""
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerifEthiopic-Regular.ttf",
        "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf",
        "/usr/share/fonts/truetype/fonts-senamirmir-washra/washrab.ttf",
        "/usr/share/fonts/truetype/fonts-senamirmir-washra/jiret.ttf",
        "/usr/share/fonts/truetype/fonts-senamirmir-washra/goffer.ttf",
        "/usr/share/fonts/truetype/fonts-senamirmir-washra/yigezubisratgothic.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ]

    # Ask fontconfig when available; this finds fonts outside the common paths.
    import shutil
    import subprocess

    if shutil.which("fc-list"):
        try:
            result = subprocess.run(
                ["fc-list", ":lang=am", "file"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in result.stdout.splitlines():
                path = line.split(":", 1)[0].strip()
                if path:
                    candidates.append(path)
        except Exception:  # noqa: BLE001, S110
            pass

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate) and _font_has_ethiopic_glyphs(candidate):
            return candidate
    return None


def _render_synthetic_text(text: str, font_path: str, font_size: int = 48) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, font_size)
    lines = text.split("\n")
    dummy = Image.new("L", (10, 10), color=255)
    draw = ImageDraw.Draw(dummy)
    widths = []
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    padding = font_size
    width = max(widths, default=1) + padding * 2
    line_height = max(max(heights, default=font_size), font_size) + int(font_size * 0.45)
    height = line_height * max(len(lines), 1) + padding * 2

    image = Image.new("L", (max(width, 64), max(height, 64)), color=255)
    draw = ImageDraw.Draw(image)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=0, font=font)
        y += line_height
    return image


def run_synthetic_evaluation(
    config: OCRConfig | None = None,
    text: str | None = None,
) -> dict:
    """
    Render known text with a local Ethiopic font and run the pipeline on it.

    This is explicitly synthetic. It checks character coverage and catches
    regressions; it must not be presented as real-scan accuracy.
    """
    import numpy as np

    if config is None:
        config = OCRConfig()

    sample_text = text or (
        "ሀለሐመሠረሰሸቀበቨተቸኀነኘአከኸወዐዘዠየደጀገጠጨጰጸፀፈፐ\n"
        "፩ ፪ ፫ ፬ ፭ ፮ ፯ ፰ ፱ ፲ ፳ ፻ ፼\n"
        "የአማርኛ ቋንቋ ትምህርት። ሰላም ለሁሉም።"
    )
    font_path = find_ethiopic_font()
    report: dict = {
        "result_kind": "synthetic",
        "synthetic": True,
        "caveat": (
            "Synthetic text rendered with an installed font. This checks "
            "character coverage and regressions only and does NOT predict "
            "accuracy on real book scans."
        ),
        "font": font_path,
        "reference_text": sample_text,
        "aggregate": {},
        "samples": [],
        "warnings": [],
    }
    if not font_path:
        report["warnings"].append(
            "No Ethiopic font was found. Install fonts-noto-ethiopic or "
            "Abyssinica SIL to run synthetic checks."
        )
        return report

    try:
        pil_image = _render_synthetic_text(sample_text, font_path)
        image = np.array(pil_image)
        # Synthetic images are clean and wide; avoid accidental page splitting.
        eval_config = replace(
            config,
            auto_split=False,
            split_pages=False,
            split_at=None,
            save_page_images=False,
            save_candidates=False,
        )
        results = process_image_file_from_array(image, eval_config, page_id="synthetic")
        hypothesis = "\n".join(r.get("text", "") for r in results)
        raw_hypothesis = "\n".join(r.get("raw_text", "") for r in results)
        metrics = evaluate_pair(
            sample_text, hypothesis, sample_id="synthetic", source="synthetic"
        )
        metrics["raw_cer"] = round(
            character_error_rate(sample_text, raw_hypothesis), 6
        )
        metrics["raw_wer"] = round(
            word_error_rate(sample_text, raw_hypothesis), 6
        )
        ref_chars = {c for c in normalize_text(sample_text) if c in AMHARIC_FIDEL}
        hyp_chars = {c for c in normalize_text(hypothesis) if c in AMHARIC_FIDEL}
        coverage = len(ref_chars & hyp_chars) / len(ref_chars) if ref_chars else None
        metrics["character_coverage"] = round(float(coverage), 6) if coverage is not None else None
        metrics["missing_fidel_characters"] = "".join(sorted(ref_chars - hyp_chars))
        metrics["hypothesis"] = hypothesis
        metrics["images"] = ["<in-memory synthetic render>"]
        metrics["pages"] = [r.get("page") for r in results]
        report["samples"].append(metrics)
        report["aggregate"] = {
            "samples": 1,
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "numeral_accuracy": metrics["numeral_accuracy"],
            "character_coverage": metrics["character_coverage"],
        }
    except Exception as e:  # noqa: BLE001
        report["warnings"].append(f"Synthetic evaluation failed: {e}")
    return report


def process_image_file_from_array(image, config: OCRConfig, page_id: str = "1") -> list[dict]:
    """Internal helper used by the synthetic evaluation."""
    from .pipeline import process_page_image

    return process_page_image(image, page_id, config)


def format_evaluation_report(report: dict) -> str:
    """Format an evaluation report for terminal output."""
    lines = []
    kind = report.get("result_kind", "evaluation")
    lines.append(f"Evaluation type: {kind}")
    if report.get("synthetic"):
        lines.append("LABEL: SYNTHETIC - not a prediction of real-scan accuracy.")
    lines.append(f"Caveat: {report.get('caveat', '')}")
    lines.append("")
    aggregate = report.get("aggregate", {}) or {}
    if aggregate:
        for key, value in aggregate.items():
            lines.append(f"  {key}: {value}")
    for sample in report.get("samples", []):
        lines.append("")
        lines.append(f"  Sample: {sample.get('id')}")
        lines.append(
            f"    CER={sample.get('cer')} WER={sample.get('wer')} "
            f"numeral_accuracy={sample.get('numeral_accuracy')} "
            f"raw_CER={sample.get('raw_cer')} raw_WER={sample.get('raw_wer')}"
        )
        if sample.get("character_coverage") is not None:
            lines.append(
                f"    character_coverage={sample.get('character_coverage')} "
                f"missing={sample.get('missing_fidel_characters')!r}"
            )
    for warning in report.get("warnings", []) or []:
        lines.append(f"  warning: {warning}")
    if not report.get("samples"):
        lines.append("  No evaluated samples.")
    return "\n".join(lines)
