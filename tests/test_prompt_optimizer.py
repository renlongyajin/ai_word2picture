"""PromptOptimizer 单元测试。"""

from __future__ import annotations

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
