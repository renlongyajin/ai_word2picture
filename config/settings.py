"""Configuration helpers for the AI Word2Picture project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class AppConfig:
    """Centralized application configuration."""

    model_dir: Path = Path("models")
    assets_dir: Path = Path("assets")
    text2img_model_id: str = "models/sdxl-turbo"
    img2img_model_id: str = "models/sdxl-turbo"
    use_fp16: bool = True
    enable_xformers: bool = True
    enable_vae_tiling: bool = True
    anthropic_key: Optional[str] = None
    openai_key: Optional[str] = None
    log_dir: Path = Path("logs")
    history_path: Path = Path("logs/history.json")
    default_prompt_style: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_env_file(path: Path) -> None:
    """Populate environment variables from a simple KEY=VALUE .env file."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Return an AppConfig instance with environment-aware settings."""
    env_path = Path(config_path) if config_path else Path(".env")
    _load_env_file(env_path)

    metadata: dict[str, Any] = {
        "text2img_model_id": "models/sdxl-turbo",
        "img2img_model_id": "models/sdxl-turbo",
        "controlnet_models": {
            "canny": "models/controlnet-canny",
            "depth": "models/controlnet-depth",
        },
    }

    return AppConfig(
        text2img_model_id=metadata["text2img_model_id"],
        img2img_model_id=metadata["img2img_model_id"],
        anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_key=os.getenv("OPENAI_API_KEY"),
        metadata=metadata,
    )
