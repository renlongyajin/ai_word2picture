"""Callback implementations for the Gradio interface."""

from __future__ import annotations

from typing import Any, Dict, Optional

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

    model_entries = config.metadata.get("available_models", [])
    model_labels: Dict[str, str] = {}
    if isinstance(model_entries, list):
        for item in model_entries:
            if not isinstance(item, dict):
                continue
            value = str(item.get("value") or "")
            label = str(item.get("label") or value)
            if value:
                model_labels[label] = value

    def _ensure_text_service() -> Text2ImageService:
        if text2img is None:
            raise RuntimeError("文生图服务未配置")
        return text2img

    def _ensure_image_service() -> Image2ImageService:
        if image2img is None:
            raise RuntimeError("图生图服务未配置")
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
        preset: StylePreset, user_negative: str, optimized_negative: Optional[str]
    ) -> Optional[str]:
        segments: list[str] = []
        for candidate in (
            (optimized_negative or "").strip(),
            (preset.negative or "").strip(),
            (user_negative or "").strip(),
        ):
            if candidate and candidate not in segments:
                segments.append(candidate)
        merged = "\n".join(segments)
        return merged or None

    def _resolve_model_selection(selection: str, fallback: str) -> str:
        if not selection:
            return fallback
        return model_labels.get(selection, selection)

    def on_optimize_prompt(
        prompt: str,
        style_name: str,
        model_name: str,
    ) -> tuple[str, str, str]:
        style = _resolve_style(style_name or config.default_prompt_style)

        if optimizer is None:
            return prompt, "", "未配置提示词优化服务，已返回原始提示词。"

        try:
            bundle = optimizer.optimize(prompt, style, model=model_name or "claude")
        except Exception as exc:  # noqa: BLE001
            return prompt, "", f"提示词优化失败：{exc}"

        optimized_prompt = bundle.optimized or prompt
        optimized_negative_parts: list[str] = []
        if bundle.negative_prompt:
            optimized_negative_parts.append(bundle.negative_prompt.strip())
        if style.negative:
            base_negative = style.negative.strip()
            if base_negative and base_negative not in optimized_negative_parts:
                optimized_negative_parts.append(base_negative)
        optimized_negative = "\n".join(filter(None, optimized_negative_parts))
        return (
            optimized_prompt,
            optimized_negative,
            "提示词优化成功，可在生成前继续编辑。",
        )

    def on_change_text_model(selection: str) -> str:
        service = _ensure_text_service()
        target = _resolve_model_selection(selection, config.metadata.get("text2img_model_id", ""))
        try:
            resolved = service.set_model(target)
        except Exception as exc:  # noqa: BLE001
            return f"切换文生图模型失败：{exc}"
        return f"已切换文生图模型：{resolved}"

    def on_change_img_model(selection: str) -> str:
        service = _ensure_image_service()
        target = _resolve_model_selection(selection, config.metadata.get("img2img_model_id", ""))
        try:
            resolved = service.set_model(target)
        except Exception as exc:  # noqa: BLE001
            return f"切换图生图模型失败：{exc}"
        return f"已切换图生图模型：{resolved}"

    def on_generate_text(
        prompt: str,
        optimized_prompt: str,
        negative_prompt: str,
        optimized_negative: str,
        guidance_scale: float,
        steps: int,
        seed: Optional[int],
        height: int,
        width: int,
        style_name: str,
        model_name: str,
    ) -> tuple[Optional[Any], str]:
        service = _ensure_text_service()
        if image2img is not None and hasattr(image2img, "offload"):
            try:
                image2img.offload()
            except Exception:
                pass
        style = _resolve_style(style_name or config.default_prompt_style)

        prompt_for_model = optimized_prompt.strip() or _compose_prompt(prompt, style)
        request = PromptRequest(
            prompt=prompt_for_model,
            negative_prompt=_combine_negative(style, negative_prompt, optimized_negative or None),
            guidance_scale=float(guidance_scale),
            steps=int(steps),
            seed=_normalize_seed(seed),
            height=_normalize_dim(height),
            width=_normalize_dim(width),
        )
        try:
            result: ImageResult = service.generate(request)
        except Exception as exc:  # noqa: BLE001
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
        optimized_prompt: str,
        negative_prompt: str,
        optimized_negative: str,
        strength: float,
        guidance_scale: float,
        steps: int,
        seed: Optional[int],
        style_name: str,
        model_name: str,
        control_type_name: str,
        control_image: Any,
        conditioning_scale: float,
        guess_mode: bool,
    ) -> tuple[Optional[Any], str]:
        if init_image is None:
            return None, "生成失败：请先上传初始图像。"

        service = _ensure_image_service()
        if text2img is not None and hasattr(text2img, "offload"):
            try:
                text2img.offload()
            except Exception:
                pass
        style = _resolve_style(style_name or config.default_prompt_style)

        prompt_for_model = optimized_prompt.strip() or _compose_prompt(prompt, style)
        control_type = _parse_control_type(control_type_name)

        request = ImageToImageRequest(
            prompt=prompt_for_model,
            init_image=init_image,
            negative_prompt=_combine_negative(style, negative_prompt, optimized_negative or None),
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
        except Exception as exc:  # noqa: BLE001
            return None, f"生成失败：{exc}"

        info = "生成成功"
        if result.seed is not None:
            info += f"（seed={result.seed}）"
        return result.image, info

    return {
        "on_optimize_prompt": on_optimize_prompt,
        "on_generate_text": on_generate_text,
        "on_generate_image": on_generate_image,
        "on_change_text_model": on_change_text_model,
        "on_change_img_model": on_change_img_model,
    }
