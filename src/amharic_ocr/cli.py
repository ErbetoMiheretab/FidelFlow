"""
Command-line interface for Amharic OCR.
"""

from __future__ import annotations

import argparse
import os
import sys

from amharic_ocr.config import OCRConfig
from amharic_ocr.pipeline import run_pipeline


def _parse_preprocessing(value: str) -> list[str]:
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    allowed = {"raw", "normalized", "local", "otsu"}
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown preprocessing variant(s): {', '.join(unknown)}. "
            f"Choose from {', '.join(sorted(allowed))}."
        )
    return names


def _build_config(args) -> OCRConfig:
    variants = list(args.preprocess)
    if args.sauvola and "local" not in variants:
        variants.append("local")
    return OCRConfig(
        dpi=args.dpi,
        split_pages=args.split,
        auto_split=not args.no_auto_split,
        split_at=args.split_at,
        use_tesseract=not args.no_tesseract,
        use_easyocr=args.use_easyocr,
        easyocr_gpu=args.gpu,
        tesseract_psm=args.psm,
        layout_mode=args.layout,
        min_char_threshold=args.min_char,
        output_dir=args.output_dir,
        use_sauvola=args.sauvola,
        preprocessing_variants=variants,
        crop_to_content=args.crop,
        remove_binding_shadow=args.remove_shadow,
        remove_scan_artifacts=args.trim_edges,
        save_page_images=args.save_page_images,
        apply_numeral_corrections=not args.no_numeral_corrections,
        convert_latin_digits=args.convert_digits,
        deduplicate_numeral_runs=args.deduplicate_numerals,
        clean_margin_artifacts=not args.no_clean_margin,
        strip_margin_noise_edges=args.strip_margin_edges,
    )


def _run_evaluation(args) -> None:
    from amharic_ocr.evaluation import evaluate_dataset, format_evaluation_report
    from amharic_ocr.io.output_writer import save_evaluation_output

    config = _build_config(args)
    report = evaluate_dataset(
        args.evaluate,
        config,
        max_samples=args.max_samples,
    )
    print(format_evaluation_report(report))
    output_path = args.eval_output or os.path.join(
        args.output_dir, "evaluation_report.json"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_evaluation_output(report, output_path)
    print(f"\nSaved evaluation JSON: {output_path}")


def _run_synthetic_evaluation(args) -> None:
    from amharic_ocr.evaluation import (
        format_evaluation_report,
        run_synthetic_evaluation,
    )
    from amharic_ocr.io.output_writer import save_evaluation_output

    config = _build_config(args)
    report = run_synthetic_evaluation(config)
    print(format_evaluation_report(report))
    output_path = args.eval_output or os.path.join(
        args.output_dir, "synthetic_evaluation_report.json"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_evaluation_output(report, output_path)
    print(f"\nSaved synthetic evaluation JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Amharic OCR pipeline for unknown scanned books. The default "
            "workflow keeps raw OCR, compares a few preprocessing variants, "
            "and does not require a font/book-family label."
        )
    )

    parser.add_argument(
        "input_pdf",
        nargs="?",
        default=None,
        help="Path to input PDF file (can also be passed with -i/--input)",
    )
    parser.add_argument("-i", "--input", dest="input_opt", help="Path to input PDF file")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output_texts",
        help="Directory to save output files (default: output_texts)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="DPI for converting PDF pages to images (default: 400)",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Force splitting every page into left/right (default: auto-detect)",
    )
    parser.add_argument(
        "--no-auto-split",
        action="store_true",
        help="Disable automatic facing-page detection",
    )
    parser.add_argument(
        "--split-at",
        type=float,
        metavar="FRACTION",
        default=None,
        help="Manual split point as a fraction of width, e.g. 0.48 (overrides detection)",
    )
    parser.add_argument(
        "--no-tesseract",
        action="store_true",
        help="Disable the Tesseract OCR engine",
    )
    parser.add_argument(
        "--use-easyocr",
        action="store_true",
        help=(
            "Enable EasyOCR as a diagnostic engine. EasyOCR has no Amharic "
            "model, so it is not used to outvote Amharic OCR."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU for EasyOCR if installed (default: False)",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=3,
        help="Tesseract page segmentation mode (default: 3, automatic page layout)",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "page", "block", "line"],
        default="auto",
        help="Layout mode override: block=PSM 6, line=PSM 7, auto/page use --psm",
    )
    parser.add_argument(
        "--preprocess",
        type=_parse_preprocessing,
        default=["raw", "normalized", "local"],
        metavar="LIST",
        help=(
            "Comma-separated preprocessing variants to compare: "
            "raw,normalized,local,otsu (default: raw,normalized,local)"
        ),
    )
    parser.add_argument(
        "--sauvola",
        action="store_true",
        help="Ensure Sauvola local thresholding is among the compared variants",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Opt in to content cropping (can remove marginal text)",
    )
    parser.add_argument(
        "--remove-shadow",
        action="store_true",
        help="Opt in to cropping dark binding shadows (can remove edge text)",
    )
    parser.add_argument(
        "--trim-edges",
        action="store_true",
        help="Opt in to trimming the bottom scanner/watermark strip (can remove footnotes)",
    )
    parser.add_argument(
        "--save-page-images",
        action="store_true",
        help="Save untouched rendered page images for inspection",
    )
    parser.add_argument(
        "--no-numeral-corrections",
        action="store_true",
        help="Disable contextual Latin-lookalike repair inside numeral zones",
    )
    parser.add_argument(
        "--convert-digits",
        action="store_true",
        help="Opt in to converting standalone Arabic digits to Ethiopic numerals",
    )
    parser.add_argument(
        "--deduplicate-numerals",
        action="store_true",
        help="Opt in to collapsing 3+ repeated identical numerals (can change valid values)",
    )
    parser.add_argument(
        "--no-clean-margin",
        action="store_true",
        help="Keep even lines that contain only scanner-noise characters",
    )
    parser.add_argument(
        "--strip-margin-edges",
        action="store_true",
        help="Opt in to stripping leading/trailing dash/punctuation noise from lines",
    )
    parser.add_argument(
        "--min-char",
        type=int,
        default=50,
        help="Minimum character threshold for review flagging (default: 50)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run environment setup diagnostics and Amharic model checks",
    )
    parser.add_argument(
        "--test-numerals",
        action="store_true",
        help="Run a synthetic Ethiopic numeral recognition test",
    )
    parser.add_argument(
        "--evaluate",
        metavar="PATH",
        default=None,
        help=(
            "Evaluate on a small image/transcription set (directory, image, "
            "JSON manifest or CSV manifest) and report CER/WER/numeral accuracy"
        ),
    )
    parser.add_argument(
        "--synthetic-eval",
        action="store_true",
        help="Run a clearly labelled synthetic character-coverage check",
    )
    parser.add_argument(
        "--eval-output",
        default=None,
        help="Path for evaluation JSON output",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of evaluation samples",
    )

    args = parser.parse_args()

    if args.diagnose:
        from amharic_ocr.diagnostics import check_system_setup

        check_system_setup()
        sys.exit(0)

    if args.test_numerals:
        from amharic_ocr.diagnostics import test_numeral_recognition

        test_numeral_recognition()
        sys.exit(0)

    if args.synthetic_eval:
        _run_synthetic_evaluation(args)
        sys.exit(0)

    if args.evaluate:
        _run_evaluation(args)
        sys.exit(0)

    pdf_file = args.input_opt or args.input_pdf
    if not pdf_file:
        parser.print_help()
        print("\nError: Please specify a PDF file to process, or use --evaluate.")
        sys.exit(1)

    if not os.path.exists(pdf_file):
        print(f"Error: File not found: {pdf_file}")
        sys.exit(1)

    config = _build_config(args)

    if not config.use_tesseract and not config.use_easyocr:
        print("Error: You cannot disable all OCR engines.")
        sys.exit(1)

    try:
        run_pipeline(pdf_file, config)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error running pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
