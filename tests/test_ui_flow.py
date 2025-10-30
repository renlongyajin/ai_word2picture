"""Gradio UI 回调流程测试。"""

from __future__ import annotations

from typing import Optional

import pytest

from config.settings import AppConfig
from modules.pipelines import text2img
from modules.optimization.prompt_optimizer import PromptBundle
from modules.optimization.style_presets import StylePreset, StylePresetRegistry
from modules.ui import callbacks


class DummyText2ImageService:
    """伪造的文生图服务，用于捕获调用。"""

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


def test_on_generate_text_returns_image(monkeypatch):
    service = DummyText2ImageService()
    cbs = callbacks.build_callbacks(AppConfig(), text2img=service)
    on_generate = cbs["on_generate_text"]

    image, message = on_generate(
        "一只猫",  # prompt
        "",  # negative
        6.5,  # guidance
        20,  # steps
        None,  # seed
        512,
        512,
        "default",
        False,
        "claude",
    )

    assert image == "dummy-image"
    assert "生成成功" in message
    assert service.last_request is not None
    assert service.last_request.prompt == "一只猫"
    assert service.last_request.steps == 20
    assert service.last_request.guidance_scale == 6.5


def test_on_generate_text_handles_exception():
    service = DummyText2ImageService()
    service.should_fail = True
    cbs = callbacks.build_callbacks(AppConfig(), text2img=service)
    on_generate = cbs["on_generate_text"]

    image, message = on_generate(
        "prompt", "", 7.5, 30, None, 512, 512, "default", False, "claude"
    )

    assert image is None
    assert "生成失败" in message


class DummyOptimizer:
    def __init__(self) -> None:
        self.calls = []

    def optimize(self, prompt: str, preset: StylePreset, model: str = "claude") -> PromptBundle:
        self.calls.append((prompt, preset, model))
        return PromptBundle(
            original=prompt,
            optimized=f"{prompt} - 优化后",
            negative_prompt="optimizer negative",
        )


def test_on_generate_text_with_optimizer():
    service = DummyText2ImageService()
    optimizer = DummyOptimizer()
    registry = StylePresetRegistry()
    registry.add(StylePreset(name="dreamy", positive="dreamy light", negative="style negative"))

    cbs = callbacks.build_callbacks(
        AppConfig(),
        optimizer=optimizer,
        text2img=service,
        style_registry=registry,
    )
    on_generate = cbs["on_generate_text"]

    image, message = on_generate(
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
    request = service.last_request
    assert request is not None
    assert request.prompt.endswith("优化后")
    assert "style negative" in request.negative_prompt
    assert "optimizer negative" in request.negative_prompt
    assert "user negative" in request.negative_prompt
