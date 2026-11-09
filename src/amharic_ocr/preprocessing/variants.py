"""
Builds a small set of comparable preprocessing variants.

The default workflow compares an untouched grayscale page with background
normalisation and local thresholding. This avoids committing to a single
destructive cleaning recipe for an unknown book. Geometry/cleanup operations
that can delete content are applied only to the cleaned variants and only when
the user opts in; the untouched page remains available as the ``raw`` variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .image_prep import (
    crop_to_text_content,
    normalize_background,
    preprocess_for_text,
    remove_binding_shadow,
    remove_scan_artifacts,
    to_grayscale,
)


@dataclass
class PreprocessedVariant:
    name: str
    image: np.ndarray
    description: str
    transformations: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "shape": list(self.image.shape),
            "transformations": self.transformations,
        }


def _apply_optional_cleanup(image: np.ndarray, config) -> tuple[np.ndarray, list[dict]]:
    """Apply user-requested destructive cleanup and record exactly what happened."""
    result = image
    changes: list[dict] = []

    if getattr(config, "remove_binding_shadow", False):
        before = result.shape
        result = remove_binding_shadow(result)
        changes.append(
            {
                "name": "remove_binding_shadow",
                "requested": True,
                "applied": result.shape != before,
                "before_shape": list(before),
                "after_shape": list(result.shape),
                "warning": "Cropping a dark edge can remove text near the page edge.",
            }
        )

    if getattr(config, "remove_scan_artifacts", False):
        before = result.shape
        result = remove_scan_artifacts(result)
        changes.append(
            {
                "name": "trim_bottom_edge",
                "requested": True,
                "applied": result.shape != before,
                "before_shape": list(before),
                "after_shape": list(result.shape),
                "warning": "Trimming the bottom edge can remove footnotes or page numbers.",
            }
        )

    if getattr(config, "crop_to_content", False):
        before = result.shape
        result = crop_to_text_content(result)
        changes.append(
            {
                "name": "crop_to_content",
                "requested": True,
                "applied": result.shape != before,
                "before_shape": list(before),
                "after_shape": list(result.shape),
                "warning": "Content cropping can remove marginal text and page numbers.",
            }
        )

    return result, changes


def build_preprocessing_variants(
    original_image: np.ndarray,
    config,
    working_image: np.ndarray | None = None,
    geometry_transformations: list[dict] | None = None,
) -> list[PreprocessedVariant]:
    """
    Build the requested preprocessing variants.

    ``original_image`` is used for the ``raw`` variant so there is always an
    untouched-grayscale safety candidate. ``working_image`` may be a deskewed
    or otherwise geometry-corrected copy used for the cleaned variants.
    """
    requested = list(getattr(config, "preprocessing_variants", None) or [])
    if not requested:
        requested = ["raw", "normalized", "local"]
    if getattr(config, "use_sauvola", False) and "local" not in requested:
        requested.append("local")

    base = working_image if working_image is not None else original_image
    cleaned_base, cleanup_changes = _apply_optional_cleanup(base, config)
    cleaned_transformations = list(geometry_transformations or []) + cleanup_changes

    variants: list[PreprocessedVariant] = []
    seen: set[str] = set()

    for name in requested:
        name = str(name).strip().lower()
        if name in seen:
            continue
        seen.add(name)

        if name == "raw":
            variants.append(
                PreprocessedVariant(
                    name="raw",
                    image=to_grayscale(original_image),
                    description="Untouched grayscale page (safety baseline).",
                    transformations=[],
                )
            )
        elif name == "normalized":
            variants.append(
                PreprocessedVariant(
                    name="normalized",
                    image=normalize_background(to_grayscale(cleaned_base)),
                    description="Background/illumination normalisation, no binarisation.",
                    transformations=cleaned_transformations,
                )
            )
        elif name == "local":
            variants.append(
                PreprocessedVariant(
                    name="local",
                    image=preprocess_for_text(cleaned_base, use_sauvola=True),
                    description="Background normalisation plus Sauvola local thresholding.",
                    transformations=cleaned_transformations,
                )
            )
        elif name == "otsu":
            variants.append(
                PreprocessedVariant(
                    name="otsu",
                    image=preprocess_for_text(cleaned_base, use_sauvola=False),
                    description="Background normalisation plus global Otsu thresholding.",
                    transformations=cleaned_transformations,
                )
            )

    if not variants:
        # Never leave a page without a candidate.
        variants.append(
            PreprocessedVariant(
                name="raw",
                image=to_grayscale(original_image),
                description="Untouched grayscale page (fallback).",
                transformations=[],
            )
        )
    return variants
