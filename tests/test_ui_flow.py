"""Gradio UI callback tests."""

from __future__ import annotations

from typing import Optional

import pytest

from config.settings import AppConfig
from modules.optimization.prompt_optimizer import PromptBundle
from modules.optimization.style_presets import StylePreset, StylePresetRegistry
from modules.pipelines import text2img
from modules.pipelines.controlnet_manager import ControlType
from modules.pipelines.img2img import ImageToImageRequest, ImageToImageResult
from modules.ui import callbacks


class DummyText2ImageService:
    """Stub text-to-image service for capturing inputs."""

    def __init__(self) -> None:
        self.last_request: Optional[text2img.PromptRequest] = None
        self.last_model: Optional[str] = None
        self.should_fail = False
        self.offloaded = False

    def generate(self, request: text2img.PromptRequest) -> text2img.ImageResult:
        if self.should_fail:
            raise RuntimeError("生成失败")
        self.last_request = request
        return text2img.ImageResult(
            image="dummy-image",
            images=["dummy-image"],
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
        )

    def set_model(self, model_id: str) -> str:
        self.last_model = model_id
        return model_id

    def offload(self) -> None:
        self.offloaded = True


class DummyImage2ImageService:
    """Stub image-to-image service for capturing inputs."""

    def __init__(self) -> None:
        self.last_request: Optional[ImageToImageRequest] = None
        self.last_model: Optional[str] = None
        self.offloaded = False

    def generate(self, request: ImageToImageRequest) -> ImageToImageResult:
        self.last_request = request
        return ImageToImageResult(
            image="dummy-image",
            images=["dummy-image"],
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            control_type=request.control_type,
        )

    def set_model(self, model_id: str) -> str:
        self.last_model = model_id
        return model_id

    def offload(self) -> None:
        self.offloaded = True


class DummyOptimizer:
    """Minimal optimizer stub returning fixed prompts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, StylePreset, str]] = []

    def optimize(self, prompt: str, preset: StylePreset, model: str = "claude") -> PromptBundle:
        self.calls.append((prompt, preset, model))
        return PromptBundle(
            original=prompt,
            optimized=f"{prompt} - OPT",
            negative_prompt="optimizer negative",
        )


def build_callbacks(
    *,
    optimizer: DummyOptimizer | None = None,
    text_service: DummyText2ImageService | None = None,
    image_service: DummyImage2ImageService | None = None,
    registry: StylePresetRegistry | None = None,
    config: AppConfig | None = None,
):
    config = config or AppConfig()
    return callbacks.build_callbacks(
        config,
        optimizer=optimizer,
        text2img=text_service,
        image2img=image_service,
        style_registry=registry,
    )


def test_on_optimize_prompt_success():
    optimizer = DummyOptimizer()
    registry = StylePresetRegistry()
    registry.add(StylePreset(name="dreamy", positive="dreamy light"))
    cb = build_callbacks(optimizer=optimizer, registry=registry)["on_optimize_prompt"]

    optimized, optimized_negative, message = cb("prompt", "dreamy", "claude")

    assert optimized.endswith("OPT")
    assert optimized_negative == "optimizer negative"
    assert "成功" in message
    assert optimizer.calls


def test_on_optimize_prompt_without_backend():
    cb = build_callbacks()["on_optimize_prompt"]
    optimized, optimized_negative, message = cb("prompt", "default", "claude")

    assert optimized == "prompt"
    assert optimized_negative == ""
    assert "未配置" in message


def test_on_generate_text_uses_optimized_prompt():
    text_service = DummyText2ImageService()
    image_service = DummyImage2ImageService()
    cb = build_callbacks(text_service=text_service, image_service=image_service)["on_generate_text"]

    image, message = cb(
        "base prompt",
        "optimized prompt",
        "user negative",
        "opt negative",
        6.5,
        20,
        None,
        512,
        512,
        "default",
        "claude",
    )

    assert image == "dummy-image"
    assert "生成成功" in message
    request = text_service.last_request
    assert request is not None
    assert request.prompt == "optimized prompt"
    assert "opt negative" in (request.negative_prompt or "")
    assert "user negative" in (request.negative_prompt or "")
    assert image_service.offloaded is True


def test_on_generate_text_handles_exception():
    text_service = DummyText2ImageService()
    text_service.should_fail = True
    cb = build_callbacks(text_service=text_service)["on_generate_text"]

    image, message = cb(
        "prompt",
        "",
        "",
        "",
        7.5,
        30,
        None,
        512,
        512,
        "default",
        "claude",
    )

    assert image is None
    assert "生成失败" in message


def test_on_generate_image_with_control_and_optimized_prompt():
    text_service = DummyText2ImageService()
    image_service = DummyImage2ImageService()
    cb = build_callbacks(text_service=text_service, image_service=image_service)["on_generate_image"]

    image, message = cb(
        "init-image",
        "base prompt",
        "optimized prompt",
        "user negative",
        "opt negative",
        0.7,
        7.5,
        30,
        None,
        "default",
        "claude",
        ControlType.CANNY.value,
        "edge-map",
        0.8,
        True,
    )

    assert image == "dummy-image"
    assert "生成成功" in message
    request = image_service.last_request
    assert request is not None
    assert request.prompt == "optimized prompt"
    assert request.control_type == ControlType.CANNY
    assert request.control_image == "edge-map"
    negative = request.negative_prompt or ""
    assert "user negative" in negative
    assert "opt negative" in negative
    assert text_service.offloaded is True


def test_on_generate_image_requires_init_image():
    cb = build_callbacks(image_service=DummyImage2ImageService())["on_generate_image"]
    image, message = cb(
        None,
        "prompt",
        "",
        "",
        "",
        0.5,
        7.0,
        20,
        None,
        "default",
        "claude",
        "",
        None,
        1.0,
        False,
    )

    assert image is None
    assert "请先上传初始图像" in message


def test_on_change_text_model_switches_service():
    text_service = DummyText2ImageService()
    config = AppConfig()
    config.metadata["available_models"] = [
        {"label": "模型A", "value": "path/to/model_a"},
        {"label": "模型B", "value": "path/to/model_b"},
    ]
    cb_map = build_callbacks(
        optimizer=DummyOptimizer(),
        text_service=text_service,
        image_service=DummyImage2ImageService(),
        registry=StylePresetRegistry(),
        config=config,
    )

    message = cb_map["on_change_text_model"]("模型B")

    assert "切换文生图模型" in message
    assert text_service.last_model == "path/to/model_b"


def test_on_change_img_model_switches_service():
    image_service = DummyImage2ImageService()
    config = AppConfig()
    config.metadata["available_models"] = [
        {"label": "模型A", "value": "path/to/model_a"},
        {"label": "模型B", "value": "path/to/model_b"},
    ]
    cb_map = build_callbacks(
        optimizer=DummyOptimizer(),
        text_service=DummyText2ImageService(),
        image_service=image_service,
        registry=StylePresetRegistry(),
        config=config,
    )

    message = cb_map["on_change_img_model"]("模型A")

    assert "切换图生图模型" in message
    assert image_service.last_model == "path/to/model_a"
