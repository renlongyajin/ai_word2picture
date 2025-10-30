"""Utility helpers for image preprocessing and postprocessing."""

from __future__ import annotations

from typing import Any, Tuple


def prepare_image(image: Any, target_size: Tuple[int, int]) -> Any:
    """Resize or pad the image to the target size."""
    _ = target_size
    raise NotImplementedError("Image preparation will be implemented alongside pipelines.")


def generate_thumbnail(image: Any, max_size: Tuple[int, int] = (256, 256)) -> Any:
    """Create a thumbnail suitable for history previews."""
    _ = max_size
    raise NotImplementedError("Thumbnail generation will be implemented alongside pipelines.")
