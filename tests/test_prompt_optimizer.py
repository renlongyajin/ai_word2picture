"""PromptOptimizer 单元测试。"""

from __future__ import annotations

import types

import pytest

from config.settings import AppConfig
from modules.optimization.prompt_optimizer import PromptOptimizer
from modules.optimization.style_presets import StylePreset


def build_optimizer() -> PromptOptimizer:
    config = AppConfig()
    optimizer = PromptOptimizer(config)
    optimizer.clear_backends()
    return optimizer


def test_optimize_with_registered_backend():
    optimizer = build_optimizer()
    preset = StylePreset(name="dreamy", positive="dreamy lighting", negative="low quality")

    def fake_backend(request):
        assert "dreamy lighting" in request.prompt_text
        return "LLM 优化后的提示"

    optimizer.register_backend("claude", fake_backend)
    bundle = optimizer.optimize("夕阳下的城市", preset, model="claude")

    assert bundle.optimized == "LLM 优化后的提示"
    assert bundle.negative_prompt == "low quality"
    assert bundle.original == "夕阳下的城市"


def test_optimize_fallback_without_backend():
    optimizer = build_optimizer()
    preset = StylePreset(name="watercolor", positive="watercolor style", negative="harsh shadows")

    bundle = optimizer.optimize("一只猫", preset, model="claude")

    assert "一只猫" in bundle.optimized
    assert "watercolor style" in bundle.optimized
    assert bundle.negative_prompt == "harsh shadows"


def test_openai_backend_registered(monkeypatch):
    """Ensure GPT backend registers and transforms prompt when OpenAI SDK 可用。"""

    # 构造假的 openai 模块
    class DummyCompletion:
        def __init__(self, text: str) -> None:
            self.output_text = text

    class DummyResponses:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return DummyCompletion("优化后的提示：猫咪在暮色街头跳跃，光影柔和。")

    class DummyOpenAI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.responses = DummyResponses()

    import modules.optimization.prompt_optimizer as prompt_optimizer_module

    original_import = prompt_optimizer_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "openai":
            module = types.SimpleNamespace(OpenAI=DummyOpenAI)
            return module
        return original_import(name)

    monkeypatch.setattr(prompt_optimizer_module.importlib, "import_module", fake_import_module)

    config = AppConfig()
    config.openai_key = "test-key"
    config.metadata = {}

    optimizer = PromptOptimizer(config)
    bundle = optimizer.optimize(
        prompt="一只可爱的白猫在屋顶晒太阳",
        preset=StylePreset(name="default", positive="soft lighting"),
        model="gpt",
    )

    assert "优化后的提示" in bundle.optimized
    assert optimizer.warnings == []
