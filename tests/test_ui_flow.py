"""Gradio UI 回调流程测试。"""

from __future__ import annotations

from typing import Optional

import pytest

from config.settings import AppConfig
from modules.pipelines import text2img
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

    image, message = on_generate("prompt", "", 7.5, 30, None, 512, 512)

    assert image is None
    assert "生成失败" in message
