"""Callback implementations for the Gradio interface."""

from __future__ import annotations

from typing import Any, Optional

from config.settings import AppConfig
from modules.optimization.prompt_optimizer import PromptOptimizer
from modules.optimization.style_presets import StylePreset
from modules.pipelines.text2img import PromptRequest, Text2ImageService


def build_callbacks(
    config: AppConfig,
    optimizer: Optional[PromptOptimizer] = None,
    text2img: Optional[Text2ImageService] = None,
) -> dict[str, Any]:
    """Return a dictionary of Gradio callback functions."""

    def on_generate_text(prompt: str, style_name: str) -> str:
        _ = (style_name,)
        if text2img is None:
            raise RuntimeError("Text2ImageService is not configured.")
        request = PromptRequest(prompt=prompt)
        text2img.generate(request)
        return "生成任务已提交（占位实现）。"

    return {"on_generate_text": on_generate_text}
