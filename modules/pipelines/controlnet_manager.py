"""ControlNet model management utilities."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ControlType(str, Enum):
    """Supported ControlNet adapters."""

    CANNY = "canny"
    DEPTH = "depth"


class ControlNetManager:
    """Load and cache ControlNet models."""

    def __init__(self) -> None:
        self._models: Dict[ControlType, Any] = {}

    def load(self, control_type: ControlType) -> Any:
        """Return a ControlNet model for the given type."""
        if control_type in self._models:
            return self._models[control_type]
        raise NotImplementedError("ControlNet loading will be implemented in phase 2.")

    def available_models(self) -> list[str]:
        """List available ControlNet adapters."""
        return [control_type.value for control_type in ControlType]
