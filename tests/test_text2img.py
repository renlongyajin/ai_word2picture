"""Text2ImageService 单元测试。"""

from types import SimpleNamespace

import pytest
import torch

from config.settings import AppConfig
from modules.pipelines import text2img


class DummyPipeline:
    """模拟 Diffusers 管线，捕获调用参数。"""

    latest: "DummyPipeline | None" = None

    def __init__(self) -> None:
        self.model_id = ""
        self.kwargs = {}
        self.device = None
        self.xformers_enabled = False
        self.vae_tiling_enabled = False
        self.called_with = None

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        instance = cls()
        instance.model_id = model_id
        instance.kwargs = kwargs
        cls.latest = instance
        return instance

    def to(self, device: str):
        self.device = device
        return self

    def enable_xformers_memory_efficient_attention(self):
        self.xformers_enabled = True

    def enable_vae_tiling(self):
        self.vae_tiling_enabled = True

    def __call__(self, **kwargs):
        self.called_with = kwargs
        return SimpleNamespace(images=["fake-image"])


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """强制使用 CPU，避免与真实 CUDA 环境耦合。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield


def test_load_pipeline_uses_config(monkeypatch, tmp_path):
    monkeypatch.setattr(text2img, "StableDiffusionXLPipeline", DummyPipeline)

    config = AppConfig(model_dir=tmp_path, enable_xformers=True, enable_vae_tiling=True)
    service = text2img.Text2ImageService(config)
    service.load_pipeline()

    pipeline = DummyPipeline.latest
    assert pipeline is not None
    assert pipeline.model_id == config.text2img_model_id
    assert pipeline.kwargs["torch_dtype"] == torch.float32  # CPU 环境应使用 float32
    assert pipeline.kwargs["cache_dir"] == str(tmp_path)
    assert service._pipeline is pipeline


def test_generate_returns_image_result(monkeypatch, tmp_path):
    monkeypatch.setattr(text2img, "StableDiffusionXLPipeline", DummyPipeline)

    config = AppConfig(model_dir=tmp_path, use_fp16=False)
    service = text2img.Text2ImageService(config)

    request = text2img.PromptRequest(prompt="test prompt", seed=123, steps=5)
    result = service.generate(request)

    pipeline = DummyPipeline.latest
    assert pipeline is not None
    assert result.image == "fake-image"
    assert result.prompt == "test prompt"
    assert pipeline.called_with["num_inference_steps"] == 5
    assert pipeline.called_with["generator"] is not None
