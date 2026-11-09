"""
Tesseract OCR engine with conservative multi-variant recognition.

The engine intentionally does **not** use a character whitelist for its default
page recognition. Whitelists can silently delete valid characters (extended
Ethiopic, punctuation, Latin abbreviations) and make output look cleaner while
losing text. Numeral-only whitelisting remains available as a separate,
diagnostic candidate.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytesseract
from PIL import Image

from ..constants import ALL_ETHIOPIC_NUMERALS, ALLOWED_EXTRA_CHARS, AMHARIC_FIDEL
from ..preprocessing.image_prep import preprocess_for_numerals, preprocess_for_text
from ..preprocessing.variants import (
    PreprocessedVariant,
    build_preprocessing_variants,
)
from .base import OCREngine, OCRResult


class TesseractEngine(OCREngine):
    def __init__(self):
        try:
            self.available_langs = pytesseract.get_languages()
        except Exception:  # noqa: BLE001
            self.available_langs = []

    @property
    def amharic_available(self) -> bool:
        return "amh" in self.available_langs

    def model_status(self) -> dict:
        return {
            "tesseract_amh_available": self.amharic_available,
            "available_languages": list(self.available_langs),
            "amharic_model": "amh" if self.amharic_available else None,
        }

    def _get_confidence(self, pil_img: Image.Image, lang: str, config_str: str) -> float:
        """Calculate mean confidence for recognized words (legacy helper)."""
        try:
            data = pytesseract.image_to_data(
                pil_img,
                lang=lang,
                config=config_str,
                output_type=pytesseract.Output.DICT,
            )
            confidences = [
                int(c)
                for c, text in zip(data["conf"], data["text"], strict=False)
                if c not in (-1, "-1") and str(text).strip()
            ]
            if confidences:
                return float(np.mean(confidences)) / 100.0
            return 0.0
        except Exception:  # noqa: BLE001
            return -1.0

    def _build_config_file(
        self, psm: int, oem: int, whitelist: str | None
    ) -> tuple[str, str | None]:
        """
        Write Tesseract options to a temp config file to avoid quoting issues
        with special characters in a whitelist.
        """
        base_config = f"--psm {psm} --oem {oem}"
        if not whitelist:
            return base_config, None

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cfg", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(f"tessedit_char_whitelist {whitelist}\n")
            path = tmp.name
        return f"{base_config} {path}", path

    @staticmethod
    def _reconstruct_text_and_regions(data: dict) -> tuple[str, list[dict]]:
        """Rebuild reading-order text and line regions from Tesseract TSV data."""
        line_words: dict[tuple[int, int, int], list[dict]] = {}

        for i, raw_text in enumerate(data.get("text", [])):
            text = str(raw_text).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            key = (
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            line_words.setdefault(key, []).append(
                {
                    "text": text,
                    "conf": conf,
                    "left": int(data["left"][i]),
                    "top": int(data["top"][i]),
                    "width": int(data["width"][i]),
                    "height": int(data["height"][i]),
                    "word_num": int(data["word_num"][i]),
                }
            )

        lines: list[str] = []
        regions: list[dict] = []
        for order, key in enumerate(sorted(line_words)):
            words = sorted(line_words[key], key=lambda w: w["word_num"])
            line_text = " ".join(w["text"] for w in words)
            lines.append(line_text)
            left = min(w["left"] for w in words)
            top = min(w["top"] for w in words)
            right = max(w["left"] + w["width"] for w in words)
            bottom = max(w["top"] + w["height"] for w in words)
            confs = [w["conf"] for w in words if w["conf"] >= 0]
            regions.append(
                {
                    "order": order,
                    "text": line_text,
                    "bbox": [left, top, right - left, bottom - top],
                    "block": key[0],
                    "paragraph": key[1],
                    "line": key[2],
                    "confidence": round(float(np.mean(confs)) / 100.0, 4)
                    if confs
                    else None,
                }
            )

        return "\n".join(lines).strip(), regions

    def _run_tesseract(
        self,
        image: np.ndarray,
        whitelist: str | None,
        lang: str,
        psm: int,
        oem: int,
    ) -> OCRResult:
        pil_img = Image.fromarray(image)
        config_str, tmp_path = self._build_config_file(psm, oem, whitelist)
        try:
            data = pytesseract.image_to_data(
                pil_img,
                lang=lang,
                config=config_str,
                output_type=pytesseract.Output.DICT,
            )
            text, regions = self._reconstruct_text_and_regions(data)
            confs = [
                float(c)
                for c, t in zip(data["conf"], data["text"], strict=False)
                if float(c) >= 0 and str(t).strip()
            ]
            confidence = float(np.mean(confs)) / 100.0 if confs else 0.0
            return OCRResult(
                text=text,
                confidence=confidence,
                engine_name=f"tesseract_{lang}_psm{psm}",
                metadata={
                    "psm": psm,
                    "oem": oem,
                    "lang": lang,
                    "has_whitelist": bool(whitelist),
                    "regions": regions,
                },
            )
        except Exception as e:  # noqa: BLE001
            return OCRResult(
                text=f"[Tesseract Error: {e}]",
                confidence=0.0,
                engine_name="tesseract_error",
                metadata={"error": str(e)},
            )
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _resolve_language(self, config) -> tuple[str, list[str]]:
        notes: list[str] = []
        requested = list(getattr(config, "languages", ["amh", "eng"]))
        usable = [lang for lang in requested if lang in self.available_langs]
        if "amh" in requested and "amh" not in self.available_langs:
            notes.append(
                "Amharic Tesseract model ('amh') is not installed; falling back "
                "to English. Amharic recognition will be poor."
            )
        if not usable:
            usable = ["eng"] if "eng" in self.available_langs else []
            if not usable and self.available_langs:
                usable = [self.available_langs[0]]
        return "+".join(usable) if usable else "eng", notes

    def _resolve_psm(self, config) -> int:
        mode = getattr(config, "layout_mode", "auto")
        if mode == "block":
            return 6
        if mode == "line":
            return 7
        return int(getattr(config, "tesseract_psm", 3))

    def ocr_with_whitelist(
        self,
        image: np.ndarray,
        whitelist: str | None = None,
        lang: str = "amh",
        psm: int = 6,
        oem: int = 1,
    ) -> OCRResult:
        """Run OCR with an optional character whitelist and PSM."""
        return self._run_tesseract(image, whitelist, lang, psm, oem)

    def ocr_numerals_only(self, image: np.ndarray, use_sauvola: bool = False) -> OCRResult:
        """
        OCR focused only on Ethiopic numerals.

        This is a separate diagnostic candidate. Its output must not be used
        as the whole page text unless the caller has independent evidence that
        the page is numeral-only.
        """
        whitelist = ALL_ETHIOPIC_NUMERALS
        processed = preprocess_for_numerals(image, use_sauvola=use_sauvola)
        lang = "amh" if self.amharic_available else "eng"

        results: list[OCRResult] = []
        for psm in (10, 7, 8):
            result = self.ocr_with_whitelist(
                processed, whitelist=whitelist, lang=lang, psm=psm
            )
            result.metadata.update(
                {
                    "variant": "numeral_whitelist",
                    "scope": "numerals",
                    "preprocessing": "numeral_whitelist",
                    "description": "Ethiopic numeral-only whitelist candidate.",
                }
            )
            results.append(result)

        valid_results = [
            r for r in results if r.text and not r.text.startswith("[Tesseract Error")
        ]
        if valid_results:
            from ..fusion.combiner import score_ocr_result

            return max(valid_results, key=score_ocr_result)
        return (
            results[0]
            if results
            else OCRResult(text="", confidence=0.0, engine_name="tesseract_numerals")
        )

    def ocr_mixed_content(
        self, image: np.ndarray, lang: str = "amh+eng", psm: int = 6, use_sauvola: bool = False
    ) -> OCRResult:
        """
        Legacy mixed-content pass with a Fidel whitelist.

        Kept for API compatibility. The default page workflow does not use it
        because whitelisting can remove valid characters.
        """
        whitelist = AMHARIC_FIDEL + ALL_ETHIOPIC_NUMERALS + ALLOWED_EXTRA_CHARS
        processed = preprocess_for_text(image, use_sauvola=use_sauvola)
        if "amh" not in self.available_langs:
            lang = "eng"
        return self.ocr_with_whitelist(
            processed, whitelist=whitelist, lang=lang, psm=psm
        )

    def recognize_variants(
        self,
        variants: list[PreprocessedVariant],
        config,
    ) -> list[OCRResult]:
        """Run one unwhitelisted Tesseract pass per preprocessing variant."""
        lang, language_notes = self._resolve_language(config)
        psm = self._resolve_psm(config)
        oem = int(getattr(config, "tesseract_oem", 1))
        results: list[OCRResult] = []

        for variant in variants:
            result = self._run_tesseract(
                variant.image,
                whitelist=None,
                lang=lang,
                psm=psm,
                oem=oem,
            )
            result.metadata.update(
                {
                    "variant": variant.name,
                    "scope": "page",
                    "preprocessing": variant.name,
                    "description": variant.description,
                    "transformations": variant.transformations,
                    "language_notes": language_notes,
                    "amharic_model_available": self.amharic_available,
                }
            )
            results.append(result)
        return results

    def recognize_candidates(
        self,
        image: np.ndarray,
        config,
        include_numeral_pass: bool = False,
    ) -> list[OCRResult]:
        """Return all page candidates (and optionally a numeral-only candidate)."""
        variants = build_preprocessing_variants(image, config)
        results = self.recognize_variants(variants, config)
        if include_numeral_pass:
            numeral_result = self.ocr_numerals_only(
                image, use_sauvola=bool(getattr(config, "use_sauvola", False))
            )
            results.append(numeral_result)
        return results

    def ocr_with_fallback(self, image: np.ndarray, config=None) -> OCRResult:
        """Backwards-compatible wrapper that returns the best page candidate."""
        if config is None:
            from ..config import OCRConfig

            config = OCRConfig()
        from ..fusion.combiner import score_ocr_result

        candidates = self.recognize_candidates(image, config)
        valid = [r for r in candidates if r.text.strip()]
        if valid:
            return max(valid, key=score_ocr_result)
        return OCRResult(text="", confidence=0.0, engine_name="tesseract_fallback")

    def recognize(self, image: np.ndarray, config=None) -> OCRResult:
        """Primary engine interface. Returns the best page-text candidate."""
        return self.ocr_with_fallback(image, config=config)
