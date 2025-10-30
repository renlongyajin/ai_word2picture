"""Generation history tracking."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass(slots=True)
class GenerationRecord:
    """Metadata describing a generation event."""

    prompt: str
    image_path: Path
    mode: str  # text2img or img2img
    created_at: float


class GenerationHistoryService:
    """Simple JSON-backed history store."""

    def __init__(self, history_path: Path) -> None:
        self.history_path = history_path

    def record(self, record: GenerationRecord) -> None:
        """Append a history record to disk."""
        raise NotImplementedError("History persistence will be implemented in phase 4.")

    def list(self, limit: int = 10) -> List[GenerationRecord]:
        """Return the most recent records."""
        raise NotImplementedError("History retrieval will be implemented in phase 4.")
