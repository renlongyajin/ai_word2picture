"""Text-to-image pipeline service implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import torch
from diffusers import StableDiffusionXLPipeline

from config.settings import AppConfig


@dataclass(slots=True)
class PromptRequest:
    """Request data for text-to-image generation."""

    prompt: str
    negative_prompt: Optional[str] = None
    guidance_scale: float = 7.5
    steps: int = 30
    seed: Optional[int] = None
    height: int = 512
    width: int = 512


@dataclass(slots=True)
class ImageResult:
    """Result payload produced by the text-to-image pipeline."""

    image: Any
    images: List[Any]
    prompt: str
    negative_prompt: Optional[str]
    seed: Optional[int]


class Text2ImageService:
    """Facade around a Stable Diffusion pipeline."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._pipeline: Optional[StableDiffusionXLPipeline] = None
        self._device: Optional[str] = None

    def _preferred_device(self) -> str:
        """Return the preferred torch device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _preferred_dtype(self, device: str) -> torch.dtype:
        """Select dtype based on configuration和设备能力."""
        if self.config.use_fp16 and device == "cuda":
            return torch.float16
        return torch.float32

    def _resolve_model_id(self) -> str:
        """Determine which model权重用于推理."""
        metadata_model_id = self.config.metadata.get("text2img_model_id")
        return metadata_model_id or self.config.text2img_model_id

    def load_pipeline(self) -> None:
        """Lazy-load the text-to-image pipeline."""
        if self._pipeline is not None:
            return

        device = self._preferred_device()
        dtype = self._preferred_dtype(device)
        model_id = self._resolve_model_id()
        cache_dir = Path(self.config.model_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=str(cache_dir),
            use_safetensors=True,
        )
        pipeline.to(device)

        if self.config.enable_xformers:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                # 当 xformers 不可用时忽略，保持功能可用
                pass

        if self.config.enable_vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()

        self._pipeline = pipeline
        self._device = device

    def generate(self, request: PromptRequest) -> ImageResult:
        """Generate an image from text prompt."""
        self.load_pipeline()
        assert self._pipeline is not None  # For type checkers

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self._device or self._preferred_device())
            generator.manual_seed(request.seed)

        result = self._pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.steps,
            generator=generator,
            height=request.height,
            width=request.width,
        )

        images = list(getattr(result, "images", []))
        if not images:
            raise RuntimeError("生成失败：未收到任何图像输出。")

        return ImageResult(
            image=images[0],
            images=images,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
        )
