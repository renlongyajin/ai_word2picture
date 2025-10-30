"""ControlNet model management utilities."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from diffusers import ControlNetModel


class ControlType(str, Enum):
    """Supported ControlNet adapters."""

    CANNY = "canny"
    DEPTH = "depth"


class ControlNetManager:
    """Load and cache ControlNet models."""

    def __init__(
        self,
        model_ids: Optional[Dict[ControlType, str]] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._models: Dict[ControlType, Any] = {}
        self._model_ids = model_ids or {}
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    def load(self, control_type: ControlType) -> Any:
        """Return a ControlNet model for the given type."""
        if control_type in self._models:
            return self._models[control_type]

        model_id = self._model_ids.get(control_type)
        if model_id is None:
            raise KeyError(f"未配置 {control_type.value} 对应的 ControlNet 模型。")

        kwargs: Dict[str, Any] = {}
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(self._cache_dir)

        controlnet = ControlNetModel.from_pretrained(model_id, **kwargs)
        self._models[control_type] = controlnet
        return controlnet

    def available_models(self) -> list[str]:
        """List available ControlNet adapters."""
        return [control_type.value for control_type in self._model_ids.keys()]
