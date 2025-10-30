"""Style preset management."""

from __future__ import annotations

import json
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
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        for entry in data:
            preset = StylePreset(
                name=entry["name"],
                positive=entry.get("positive", ""),
                negative=entry.get("negative", ""),
                guidance_scale=entry.get("guidance_scale"),
                sampler=entry.get("sampler"),
            )
            self.add(preset)

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
