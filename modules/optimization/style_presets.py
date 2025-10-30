"""Style preset management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(slots=True)
class StylePreset:
    """Style attributes injected into the diffusion prompt."""

    name: str
    positive: str
    negative: str = ""
    guidance_scale: Optional[float] = None
    sampler: Optional[str] = None


class StylePresetRegistry:
    """In-memory registry of style presets."""

    def __init__(self) -> None:
        self._presets: Dict[str, StylePreset] = {}

    def load_from_file(self, path: Path) -> None:
        """Load presets from a JSON file."""
        _ = path
        # Actual file loading will be implemented when assets are finalized.

    def add(self, preset: StylePreset) -> None:
        """Register a new style preset."""
        self._presets[preset.name] = preset

    def list_presets(self) -> List[StylePreset]:
        """Return all registered presets."""
        return list(self._presets.values())

    def get(self, name: str) -> StylePreset:
        """Retrieve a preset by name."""
        try:
            return self._presets[name]
        except KeyError as exc:
            raise KeyError(f"Style preset '{name}' not found") from exc
