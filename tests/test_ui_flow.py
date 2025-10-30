"""Gradio UI callback tests."""

from __future__ import annotations

from typing import Optional

import pytest

from config.settings import AppConfig
from modules.optimization.prompt_optimizer import PromptBundle
from modules.optimization.style_presets import StylePreset, StylePresetRegistry
from modules.pipelines.controlnet_manager import ControlType
from modules.pipelines.img2img import ImageToImageRequest, ImageToImageResult
from modules.pipelines import text2img
from modules.ui import callbacks


class DummyText2ImageService:
    """Stub text-to-image service for capturing inputs."""

    def __init__(self) -> None:
        self.last_request: Optional[text2img.PromptRequest] = None
        self.should_fail = False

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


class DummyImage2ImageService:
    """Stub image-to-image service for capturing inputs."""

    def __init__(self) -> None:
        self.last_request: Optional[ImageToImageRequest] = None

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


class DummyOptimizer:
    def __init__(self) -> None:
        self.calls = []

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
):
    return callbacks.build_callbacks(
        AppConfig(),
        optimizer=optimizer,
        text2img=text_service,
        image2img=image_service,
        style_registry=registry,
    )


def test_on_generate_text_returns_image():
    text_service = DummyText2ImageService()
    cb = build_callbacks(text_service=text_service)["on_generate_text"]

    image, message = cb(
        "a cat",
        "",
        6.5,
        20,
        None,
        512,
        512,
        "default",
        False,
        "claude",
    )

    assert image == "dummy-image"
    assert "生成成功" in message
    assert text_service.last_request is not None
    assert text_service.last_request.prompt == "a cat"
    assert text_service.last_request.steps == 20


def test_on_generate_text_handles_exception():
    text_service = DummyText2ImageService()
    text_service.should_fail = True
    cb = build_callbacks(text_service=text_service)["on_generate_text"]

    image, message = cb("prompt", "", 7.5, 30, None, 512, 512, "default", False, "claude")

    assert image is None
    assert "生成失败" in message


def test_on_generate_text_with_optimizer():
    text_service = DummyText2ImageService()
    optimizer = DummyOptimizer()
    registry = StylePresetRegistry()
    registry.add(StylePreset(name="dreamy", positive="dreamy light", negative="style negative"))

    cb = build_callbacks(
        optimizer=optimizer,
        text_service=text_service,
        registry=registry,
    )["on_generate_text"]

    image, message = cb(
        "prompt base",
        "user negative",
        7.0,
        25,
        None,
        512,
        512,
        "dreamy",
        True,
        "claude",
    )

    assert image == "dummy-image"
    assert "生成成功" in message
    assert optimizer.calls, "优化器应被调用"
    request = text_service.last_request
    assert request is not None
    assert request.prompt.endswith("OPT")
    assert "style negative" in (request.negative_prompt or "")
    assert "optimizer negative" in (request.negative_prompt or "")
    assert "user negative" in (request.negative_prompt or "")


def test_on_generate_image_without_control():
    text_service = DummyText2ImageService()
    image_service = DummyImage2ImageService()
    cb = build_callbacks(text_service=text_service, image_service=image_service)["on_generate_image"]

    image, message = cb(
        "init-image",
        "base prompt",
        "",
        0.6,
        8.0,
        18,
        99,
        "default",
        False,
        "claude",
        "",
        None,
        1.0,
        False,
    )

    assert image == "dummy-image"
    assert "生成成功" in message
    request = image_service.last_request
    assert request is not None
    assert request.prompt == "base prompt"
    assert request.control_type is None
    assert request.strength == pytest.approx(0.6)
    assert request.seed == 99


def test_on_generate_image_with_control_and_optimizer():
    text_service = DummyText2ImageService()
    image_service = DummyImage2ImageService()
    optimizer = DummyOptimizer()
    registry = StylePresetRegistry()
    registry.add(StylePreset(name="dreamy", positive="dreamy", negative="style neg"))

    cb = build_callbacks(
        optimizer=optimizer,
        text_service=text_service,
        image_service=image_service,
        registry=registry,
    )["on_generate_image"]

    image, message = cb(
        "init-image",
        "base prompt",
        "user negative",
        0.7,
        7.5,
        30,
        None,
        "dreamy",
        True,
        "claude",
        ControlType.CANNY.value,
        "edge-map",
        0.8,
        True,
    )

    assert image == "dummy-image"
    assert "生成成功" in message
    assert optimizer.calls
    request = image_service.last_request
    assert request is not None
    assert request.prompt.endswith("OPT")
    assert request.control_type == ControlType.CANNY
    assert request.control_image == "edge-map"
    assert request.controlnet_conditioning_scale == pytest.approx(0.8)
    assert request.guess_mode is True
    negative = request.negative_prompt or ""
    assert "style neg" in negative
    assert "optimizer negative" in negative
    assert "user negative" in negative
