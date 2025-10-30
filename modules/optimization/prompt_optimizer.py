"""Prompt optimization via third-party LLM APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.settings import AppConfig
from modules.optimization.style_presets import StylePreset


@dataclass(slots=True)
class PromptBundle:
    """Container for optimized prompt data."""

    original: str
    optimized: str
    negative_prompt: Optional[str] = None


class PromptOptimizer:
    """Interface to Claude/GPT prompt enhancement."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def optimize(self, prompt: str, preset: StylePreset, model: str = "claude") -> PromptBundle:
        """Return the optimized prompt bundle."""
        raise NotImplementedError("Prompt optimization will be implemented in phase 3.")
