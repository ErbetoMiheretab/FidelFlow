"""
EasyOCR Engine implementation (optional fallback engine).

Important: EasyOCR 1.7.x has no Amharic recognition model. The reader is
therefore restricted to a Latin model and must not be presented as an Amharic
accuracy improvement or used to outvote Tesseract on Amharic text.
"""

import numpy as np

from ..preprocessing.image_prep import to_grayscale
from .base import OCREngine, OCRResult

# Global reader instance for worker processes (prevents reloading per page)
_reader_instance = None
# Cached Amharic-support answer. Importing EasyOCR pulls in torch, which is
# slow, so this must only happen when EasyOCR is explicitly used.
_amharic_supported: bool | None = None


def easyocr_supports_amharic() -> bool:
    """Return whether the installed EasyOCR build lists Amharic support."""
    global _amharic_supported
    if _amharic_supported is None:
        try:
            from easyocr.config import all_lang_list

            _amharic_supported = any(lang in all_lang_list for lang in ("am", "amh"))
        except Exception:  # noqa: BLE001
            _amharic_supported = False
    return _amharic_supported


def get_easyocr_reader(languages, gpu=False):
    global _reader_instance
    if _reader_instance is None:
        import easyocr

        # EasyOCR uses 2-letter ISO codes (e.g. 'en' for English).
        lang_map = {"eng": "en"}
        ocr_langs = [
            lang_map.get(lang, lang)
            for lang in languages
            if lang not in ("amh", "am")
            and lang in ("eng", "en")
        ]
        if not ocr_langs:
            ocr_langs = ["en"]

        try:
            _reader_instance = easyocr.Reader(ocr_langs, gpu=gpu, verbose=False)
        except Exception:  # noqa: BLE001
            _reader_instance = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    return _reader_instance


class EasyOCREngine(OCREngine):
    @property
    def amharic_supported(self) -> bool:
        """Lazily check EasyOCR's language list without importing it eagerly."""
        return easyocr_supports_amharic()

    def model_status(self) -> dict:
        return {
            "easyocr_amharic_available": bool(self.amharic_supported),
            "note": (
                "EasyOCR Amharic model available."
                if self.amharic_supported
                else "EasyOCR has no Amharic model; Latin/digit text only."
            ),
        }

    def recognize(self, image: np.ndarray, config) -> OCRResult:
        try:
            gray = to_grayscale(image)
            reader = get_easyocr_reader(config.languages, config.easyocr_gpu)

            results = reader.readtext(gray, detail=1, paragraph=True)

            if not results:
                return OCRResult(
                    text="",
                    confidence=1.0,
                    engine_name="easyocr",
                    metadata={
                        "amharic_model_available": bool(self.amharic_supported),
                        "scope": "page",
                    },
                )

            text_blocks = []
            confidences = []

            for _bbox, text, confidence in results:
                text_blocks.append(text)
                confidences.append(confidence)

            full_text = "\n".join(text_blocks)
            mean_confidence = float(np.mean(confidences)) if confidences else 1.0

            return OCRResult(
                text=full_text,
                confidence=mean_confidence,
                engine_name="easyocr",
                metadata={
                    "num_blocks": len(results),
                    "amharic_model_available": bool(self.amharic_supported),
                    "scope": "page",
                    "warning": (
                        None
                        if self.amharic_supported
                        else "EasyOCR was run with a Latin model and cannot be "
                        "treated as an Amharic recognition improvement."
                    ),
                },
            )
        except Exception as e:  # noqa: BLE001
            return OCRResult(
                text=f"[EasyOCR Error: {e}]",
                confidence=0.0,
                engine_name="easyocr",
                metadata={"amharic_model_available": bool(self.amharic_supported)},
            )
