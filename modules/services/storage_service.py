"""File storage helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


class StorageService:
    """Handle saving generated assets."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def save_image(self, image, metadata: Dict[str, str]) -> Path:
        """Persist an image and return the file path."""
        _ = metadata
        raise NotImplementedError("Image persistence will be implemented in phase 4.")

    def cleanup(self, max_items: int = 100) -> None:
        """Limit the number of stored artifacts."""
        _ = max_items
        raise NotImplementedError("Cleanup logic will be implemented in phase 4.")
