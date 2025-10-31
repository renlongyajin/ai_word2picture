"""PromptOptimizer unit tests."""

from __future__ import annotations

import os
import types

import pytest

from config.settings import AppConfig, load_config
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
        return "LLM optimized prompt"

    optimizer.register_backend("claude", fake_backend)
    bundle = optimizer.optimize("A sunset city skyline", preset, model="claude")

    assert bundle.optimized == "LLM optimized prompt"
    assert bundle.negative_prompt == "low quality"
    assert bundle.original == "A sunset city skyline"


def test_optimize_fallback_without_backend():
    optimizer = build_optimizer()
    preset = StylePreset(name="watercolor", positive="watercolor style", negative="harsh shadows")

    bundle = optimizer.optimize("A playful cat", preset, model="claude")

    assert "A playful cat" in bundle.optimized
    assert "watercolor style" in bundle.optimized
    assert bundle.negative_prompt == "harsh shadows"


def test_openai_backend_registered(monkeypatch):
    """Ensure GPT backend registers and transforms prompt when OpenAI SDK is available."""

    class DummyCompletion:
        def __init__(self, text: str) -> None:
            self.output_text = text

    class DummyResponses:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return DummyCompletion("Optimized prompt: cat in neon-lit alley")

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
        prompt="A white cat stretches on a warm rooftop",
        preset=StylePreset(name="default", positive="soft lighting"),
        model="gpt",
    )

    assert "Optimized prompt" in bundle.optimized
    assert optimizer.warnings == []


@pytest.mark.integration
def test_openai_backend_real_call():
    """Invoke the real OpenAI-compatible backend to ensure prompts are enhanced."""

    config_env = load_config()
    api_key = (
        config_env.openai_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("SILICONFLOW_API_KEY")
    )
    if not api_key:
        pytest.skip("未检测到 OPENAI_API_KEY/SILICONFLOW_API_KEY，跳过真实调用测试。")

    config = AppConfig()
    config.openai_key = api_key
    config.metadata = dict(config_env.metadata)

    optimizer = PromptOptimizer(config)
    assert optimizer.has_backend("gpt"), f"OpenAI 后端未注册：{optimizer.warnings}"

    preset = StylePreset(name="studio", positive="cinematic lighting, depth of field")

    original_prompt = (
        "一只可爱风格的白色小猫，参考《罗小黑战记》中的罗小黑的猫咪形态。"
    )
    try:
        bundle = optimizer.optimize(original_prompt, preset, model="gpt")
    except Exception as exc:  # pragma: no cover - integration handling
        from openai import AuthenticationError, BadRequestError

        message = str(exc)
        if isinstance(exc, AuthenticationError) or "Api key is invalid" in message:
            pytest.skip(f"硅基流动 / OpenAI API 认证失败：{exc}")
        if isinstance(exc, BadRequestError) and (
            "Model does not exist" in message or "code': 20012" in message
        ):
            pytest.skip(f"硅基流动模型不可用：{exc}")
        raise

    optimized = bundle.optimized.strip()
    print("优化后的提示词：", optimized)
    assert optimized, "优化结果不应为空"
    assert optimized != original_prompt, "优化后的提示词应该与原始提示词有所不同"
