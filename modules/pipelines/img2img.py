"""Image-to-image pipeline service skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from config.settings import AppConfig


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


class Image2ImageService:
    """Facade around a ControlNet-enabled Stable Diffusion pipeline."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._pipeline: Optional[Any] = None

    def load_pipeline(self) -> None:
        """Lazy-load the image-to-image pipeline."""
        if self._pipeline is not None:
            return
        raise NotImplementedError("Pipeline loading will be implemented in phase 2.")

    def generate(self, request: ImageToImageRequest) -> Any:
        """Generate an image conditioned on an input image."""
        self.load_pipeline()
        raise NotImplementedError("Image-to-image generation will be implemented in phase 2.")
