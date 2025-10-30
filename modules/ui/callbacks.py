"""Callback implementations for the Gradio interface."""

from __future__ import annotations

from typing import Any, Optional

from config.settings import AppConfig
from modules.optimization.prompt_optimizer import PromptOptimizer
from modules.pipelines.text2img import ImageResult, PromptRequest, Text2ImageService


def build_callbacks(
    config: AppConfig,
    optimizer: Optional[PromptOptimizer] = None,
    text2img: Optional[Text2ImageService] = None,
) -> dict[str, Any]:
    """Return a dictionary of Gradio callback functions."""

    def _ensure_service() -> Text2ImageService:
        if text2img is None:
            raise RuntimeError("Text2ImageService 未配置。")
        return text2img

    def _normalize_seed(seed: Any) -> Optional[int]:
        if seed in ("", None):
            return None
        try:
            return int(seed)
        except (TypeError, ValueError):
            return None

    def _normalize_dim(value: Any, default: int = 512) -> int:
        try:
            numeric = int(value)
            return max(64, min(numeric, 2048))
        except (TypeError, ValueError):
            return default

    def on_generate_text(
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        steps: int,
        seed: Optional[int],
        height: int,
        width: int,
    ) -> tuple[Optional[Any], str]:
        service = _ensure_service()
        request = PromptRequest(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            guidance_scale=float(guidance_scale),
            steps=int(steps),
            seed=_normalize_seed(seed),
            height=_normalize_dim(height),
            width=_normalize_dim(width),
        )
        try:
            result: ImageResult = service.generate(request)
        except Exception as exc:  # 捕捉底层推理异常，返回友好提示
            return None, f"生成失败：{exc}"

        info = "生成成功"
        if result.seed is not None:
            info += f"（seed={result.seed}）"
        return result.image, info

    return {"on_generate_text": on_generate_text}
