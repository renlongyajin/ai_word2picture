"""Image-to-image pipeline service implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

from config.settings import AppConfig
from modules.pipelines.controlnet_manager import ControlNetManager, ControlType


@dataclass(slots=True)
class ImageToImageRequest:
    """Request data for image-to-image generation."""

    prompt: str
    init_image: Any
    negative_prompt: Optional[str] = None
    strength: float = 0.8
    guidance_scale: float = 7.5
    steps: int = 30
    seed: Optional[int] = None
    control_type: Optional[ControlType] = None
    control_image: Optional[Any] = None
    controlnet_conditioning_scale: float = 1.0
    guess_mode: bool = False


@dataclass(slots=True)
class ImageToImageResult:
    """Result payload for image-to-image generation."""

    image: Any
    images: List[Any]
    prompt: str
    negative_prompt: Optional[str]
    seed: Optional[int]
    control_type: Optional[ControlType]


class Image2ImageService:
    """Facade around a ControlNet-enabled Stable Diffusion pipeline."""

    def __init__(self, config: AppConfig, control_manager: Optional[ControlNetManager] = None) -> None:
        self.config = config
        self._img2img_pipeline: Optional[StableDiffusionXLImg2ImgPipeline] = None
        self._control_pipelines: Dict[ControlType, StableDiffusionXLControlNetPipeline] = {}
        self._device: Optional[str] = None
        self._control_manager = control_manager or self._create_control_manager()
        self._model_override: Optional[str] = None

    def _create_control_manager(self) -> ControlNetManager:
        mapping: dict[ControlType, str] = {}
        raw_map = self.config.metadata.get("controlnet_models", {})
        if isinstance(raw_map, dict):
            for key, model_id in raw_map.items():
                try:
                    control_type = ControlType(key)
                except ValueError:
                    continue
                if isinstance(model_id, str):
                    mapping[control_type] = model_id
        cache_dir = Path(self.config.model_dir)
        return ControlNetManager(model_ids=mapping, cache_dir=cache_dir)

    def _preferred_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _preferred_dtype(self, device: str) -> torch.dtype:
        if self.config.use_fp16 and device == "cuda":
            return torch.float16
        return torch.float32

    def _resolve_model_id(self) -> str:
        if self._model_override:
            return self._model_override
        metadata_model_id = self.config.metadata.get("img2img_model_id")
        return metadata_model_id or self.config.img2img_model_id

    def set_model(self, model_id: str) -> str:
        """Switch the active img2img model and drop cached pipelines."""
        if not model_id:
            raise ValueError("模型标识不能为空")

        current = self._resolve_model_id()
        if model_id == current:
            return current

        self._model_override = model_id
        self.config.metadata["img2img_model_id"] = model_id
        self.config.img2img_model_id = model_id

        if self._img2img_pipeline is not None:
            try:
                self._img2img_pipeline.to("cpu", dtype=torch.float32)
            except Exception:
                pass
        for pipeline in self._control_pipelines.values():
            try:
                pipeline.to("cpu", dtype=torch.float32)
            except Exception:
                pass
        self._img2img_pipeline = None
        self._control_pipelines.clear()
        self._device = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return model_id

    def offload(self) -> None:
        """Release cached ControlNet and img2img pipelines from GPU."""
        if self._img2img_pipeline is not None:
            try:
                self._img2img_pipeline.to("cpu", dtype=torch.float32)
            except Exception:
                pass
            self._img2img_pipeline = None

        for control_type, pipeline in list(self._control_pipelines.items()):
            try:
                pipeline.to("cpu", dtype=torch.float32)
            except Exception:
                pass
            self._control_pipelines.pop(control_type, None)

        self._device = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _pipeline_kwargs(self, dtype: torch.dtype) -> Dict[str, Any]:
        cache_dir = Path(self.config.model_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "cache_dir": str(cache_dir),
            "use_safetensors": True,
        }
        return kwargs

    def _configure_pipeline(self, pipeline: Any, device: str) -> None:
        pipeline.to(device)

        if self.config.enable_xformers:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        if self.config.enable_vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()

    def _load_img2img_pipeline(self) -> StableDiffusionXLImg2ImgPipeline:
        if self._img2img_pipeline is not None:
            return self._img2img_pipeline

        device = self._preferred_device()
        dtype = self._preferred_dtype(device)
        device = self._preferred_device()
        dtype = self._preferred_dtype(device)
        model_id = self._resolve_model_id()
        kwargs = self._pipeline_kwargs(dtype)

        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            **kwargs,
        )
        self._configure_pipeline(pipeline, device)

        self._img2img_pipeline = pipeline
        self._device = device
        return pipeline

    def _load_control_pipeline(self, control_type: ControlType) -> StableDiffusionXLControlNetPipeline:
        if control_type in self._control_pipelines:
            return self._control_pipelines[control_type]

        controlnet = self._control_manager.load(control_type)
        device = self._preferred_device()
        dtype = self._preferred_dtype(device)
        model_id = self._resolve_model_id()
        kwargs = self._pipeline_kwargs(dtype)

        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            **kwargs,
        )
        self._configure_pipeline(pipeline, device)
        self._control_pipelines[control_type] = pipeline
        self._device = device
        return pipeline

    def _prepare_generator(self) -> Optional[torch.Generator]:
        if self._device is None:
            self._device = self._preferred_device()
        return torch.Generator(device=self._device)

    def generate(self, request: ImageToImageRequest) -> ImageToImageResult:
        """Generate an image conditioned on an input image."""
        generator = None
        if request.seed is not None:
            generator = self._prepare_generator()
            generator.manual_seed(request.seed)

        if request.control_type is None:
            pipeline = self._load_img2img_pipeline()
        else:
            pipeline = self._load_control_pipeline(request.control_type)

        kwargs = {
            "prompt": request.prompt,
            "image": request.init_image,
            "negative_prompt": request.negative_prompt,
            "guidance_scale": request.guidance_scale,
            "strength": request.strength,
            "num_inference_steps": request.steps,
        }
        if generator is not None:
            kwargs["generator"] = generator

        if isinstance(pipeline, StableDiffusionXLControlNetPipeline) and request.control_type is not None:
            control_image = request.control_image or request.init_image
            kwargs["control_image"] = control_image
            kwargs["controlnet_conditioning_scale"] = request.controlnet_conditioning_scale
            kwargs["guess_mode"] = request.guess_mode

        result = pipeline(**kwargs)
        images = list(getattr(result, "images", []))
        if not images:
            raise RuntimeError("生成失败：未获得任何输出图像。")

        return ImageToImageResult(
            image=images[0],
            images=images,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            control_type=request.control_type,
        )
