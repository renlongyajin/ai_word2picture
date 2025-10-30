"""Shared model loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class ModelLoader:
    """Cache and reuse diffusion pipelines."""

    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root
        self._cache: Dict[str, Any] = {}

    def get_pipeline(self, name: str) -> Any:
        """Return a cached pipeline instance."""
        if name in self._cache:
            return self._cache[name]
        raise NotImplementedError("Pipeline loading will be implemented in phase 1.")

    def clear(self) -> None:
        """Empty the pipeline cache."""
        self._cache.clear()
