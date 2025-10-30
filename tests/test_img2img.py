"""Image2ImageService 单元测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from config.settings import AppConfig
from modules.pipelines import img2img
from modules.pipelines.controlnet_manager import ControlType


class DummyImg2ImgPipeline:
    """模拟 StableDiffusionXLImg2ImgPipeline。"""

    latest: "DummyImg2ImgPipeline | None" = None

    def __init__(self) -> None:
        self.model_id = ""
        self.kwargs = {}
        self.device = None
        self.called_with = None
        self.xformers_enabled = False
        self.vae_tiling_enabled = False

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
        return SimpleNamespace(images=["generated-image"])


class DummyControlPipeline(DummyImg2ImgPipeline):
    """模拟 StableDiffusionXLControlNetPipeline。"""

    def __init__(self) -> None:
        super().__init__()
        self.controlnet = None

    @classmethod
    def from_pretrained(cls, model_id: str, controlnet=None, **kwargs):
        instance = super().from_pretrained(model_id, **kwargs)
        instance.controlnet = controlnet
        return instance


class DummyControlManager:
    def __init__(self) -> None:
        self.loaded: list[ControlType] = []
        self.mapping: dict[ControlType, str] = {
            ControlType.CANNY: "controlnet-canny",
            ControlType.DEPTH: "controlnet-depth",
        }

    def load(self, control_type: ControlType) -> str:
        self.loaded.append(control_type)
        return self.mapping[control_type]


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield


def test_load_pipeline_uses_config(monkeypatch, tmp_path):
    monkeypatch.setattr(img2img, "StableDiffusionXLImg2ImgPipeline", DummyImg2ImgPipeline)
    monkeypatch.setattr(img2img, "StableDiffusionXLControlNetPipeline", DummyControlPipeline)
    config = AppConfig(model_dir=tmp_path)
    config.metadata["img2img_model_id"] = "repo/custom-img2img"
    service = img2img.Image2ImageService(config, control_manager=DummyControlManager())

    pipeline = service._load_img2img_pipeline()

    assert pipeline is not None
    assert pipeline.model_id == "repo/custom-img2img"
    assert pipeline.kwargs["torch_dtype"] == torch.float32
    assert pipeline.kwargs["cache_dir"] == str(tmp_path)
    assert pipeline.device == "cpu"


def test_generate_without_control(monkeypatch, tmp_path):
    monkeypatch.setattr(img2img, "StableDiffusionXLImg2ImgPipeline", DummyImg2ImgPipeline)
    monkeypatch.setattr(img2img, "StableDiffusionXLControlNetPipeline", DummyControlPipeline)
    config = AppConfig(model_dir=tmp_path, use_fp16=False)
    service = img2img.Image2ImageService(config, control_manager=DummyControlManager())

    request = img2img.ImageToImageRequest(
        prompt="a cat sketch",
        init_image="init-image",
        strength=0.55,
        guidance_scale=8.0,
        steps=15,
        seed=123,
    )

    result = service.generate(request)

    pipeline = DummyImg2ImgPipeline.latest
    assert pipeline is not None
    assert result.image == "generated-image"
    assert pipeline.called_with["image"] == "init-image"
    assert pipeline.called_with["strength"] == pytest.approx(0.55)
    assert pipeline.called_with["num_inference_steps"] == 15
    assert pipeline.called_with["generator"] is not None


def test_generate_with_control(monkeypatch, tmp_path):
    monkeypatch.setattr(img2img, "StableDiffusionXLImg2ImgPipeline", DummyImg2ImgPipeline)
    monkeypatch.setattr(img2img, "StableDiffusionXLControlNetPipeline", DummyControlPipeline)
    manager = DummyControlManager()
    config = AppConfig(model_dir=tmp_path)
    service = img2img.Image2ImageService(config, control_manager=manager)

    request = img2img.ImageToImageRequest(
        prompt="street at night",
        init_image="init-image",
        control_type=ControlType.CANNY,
        control_image="edge-map",
        controlnet_conditioning_scale=0.8,
        guess_mode=True,
    )

    service.generate(request)

    pipeline = DummyControlPipeline.latest
    assert pipeline is not None
    assert manager.loaded == [ControlType.CANNY]
    assert pipeline.controlnet == "controlnet-canny"
    assert pipeline.called_with["control_image"] == "edge-map"
    assert pipeline.called_with["controlnet_conditioning_scale"] == pytest.approx(0.8)
    assert pipeline.called_with["guess_mode"] is True
