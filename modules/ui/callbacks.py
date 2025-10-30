"""Callback implementations for the Gradio interface."""

from __future__ import annotations

from typing import Any, Optional

from config.settings import AppConfig
from modules.optimization.prompt_optimizer import PromptOptimizer
from modules.optimization.style_presets import StylePreset, StylePresetRegistry
from modules.pipelines.controlnet_manager import ControlType
from modules.pipelines.img2img import Image2ImageService, ImageToImageRequest, ImageToImageResult
from modules.pipelines.text2img import ImageResult, PromptRequest, Text2ImageService


def build_callbacks(
    config: AppConfig,
    optimizer: Optional[PromptOptimizer] = None,
    text2img: Optional[Text2ImageService] = None,
    image2img: Optional[Image2ImageService] = None,
    style_registry: Optional[StylePresetRegistry] = None,
) -> dict[str, Any]:
    """Return a dictionary of Gradio callback functions."""

    registry = style_registry or StylePresetRegistry()
    if not registry.list_presets():
        registry.add(StylePreset(name="default", positive="", negative=""))

    def _ensure_text_service() -> Text2ImageService:
        if text2img is None:
            raise RuntimeError("Text2ImageService 未配置。")
        return text2img

    def _ensure_image_service() -> Image2ImageService:
        if image2img is None:
            raise RuntimeError("Image2ImageService 未配置。")
        return image2img

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

    def _resolve_style(name: str) -> StylePreset:
        try:
            return registry.get(name)
        except KeyError:
            return registry.get("default")

    def _compose_prompt(base: str, preset: StylePreset) -> str:
        parts = [base.strip()]
        if preset.positive:
            parts.append(f"风格提示：{preset.positive.strip()}")
        return "\n".join(filter(None, parts))

    def _combine_negative(
        preset: StylePreset, user_negative: str, bundle_negative: Optional[str]
    ) -> Optional[str]:
        segments = [
            (bundle_negative or "").strip(),
            (preset.negative or "").strip(),
            (user_negative or "").strip(),
        ]
        merged = "\n".join(segment for segment in segments if segment)
        return merged or None

    def on_generate_text(
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        steps: int,
        seed: Optional[int],
        height: int,
        width: int,
        style_name: str,
        use_optimizer: bool,
        model_name: str,
    ) -> tuple[Optional[Any], str]:
        service = _ensure_text_service()
        style = _resolve_style(style_name or config.default_prompt_style)

        bundle_negative: Optional[str] = None
        if use_optimizer and optimizer is not None:
            bundle = optimizer.optimize(prompt, style, model=model_name or "claude")
            prompt_for_model = bundle.optimized
            bundle_negative = bundle.negative_prompt
        else:
            prompt_for_model = _compose_prompt(prompt, style)

        request = PromptRequest(
            prompt=prompt_for_model,
            negative_prompt=_combine_negative(style, negative_prompt, bundle_negative),
            guidance_scale=float(guidance_scale),
            steps=int(steps),
            seed=_normalize_seed(seed),
            height=_normalize_dim(height),
            width=_normalize_dim(width),
        )
        try:
            result: ImageResult = service.generate(request)
        except Exception as exc:
            return None, f"生成失败：{exc}"

        info = "生成成功"
        if result.seed is not None:
            info += f"（seed={result.seed}）"
        return result.image, info

    def _parse_control_type(name: str) -> Optional[ControlType]:
        if not name:
            return None
        try:
            return ControlType(name)
        except ValueError:
            return None

    def on_generate_image(
        init_image: Any,
        prompt: str,
        negative_prompt: str,
        strength: float,
        guidance_scale: float,
        steps: int,
        seed: Optional[int],
        style_name: str,
        use_optimizer: bool,
        model_name: str,
        control_type_name: str,
        control_image: Any,
        conditioning_scale: float,
        guess_mode: bool,
    ) -> tuple[Optional[Any], str]:
        if init_image is None:
            return None, "生成失败：请先上传初始图像。"

        service = _ensure_image_service()
        style = _resolve_style(style_name or config.default_prompt_style)

        bundle_negative: Optional[str] = None
        if use_optimizer and optimizer is not None:
            bundle = optimizer.optimize(prompt, style, model=model_name or "claude")
            prompt_for_model = bundle.optimized
            bundle_negative = bundle.negative_prompt
        else:
            prompt_for_model = _compose_prompt(prompt, style)

        control_type = _parse_control_type(control_type_name)
        request = ImageToImageRequest(
            prompt=prompt_for_model,
            init_image=init_image,
            negative_prompt=_combine_negative(style, negative_prompt, bundle_negative),
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            steps=int(steps),
            seed=_normalize_seed(seed),
            control_type=control_type,
            control_image=control_image,
            controlnet_conditioning_scale=float(conditioning_scale),
            guess_mode=bool(guess_mode),
        )
        try:
            result: ImageToImageResult = service.generate(request)
        except Exception as exc:
            return None, f"生成失败：{exc}"

        info = "生成成功"
        if result.seed is not None:
            info += f"（seed={result.seed}）"
        return result.image, info

    return {
        "on_generate_text": on_generate_text,
        "on_generate_image": on_generate_image,
    }
