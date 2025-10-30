"""Configuration helpers for the AI Word2Picture project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class AppConfig:
    """Centralized application configuration."""

    model_dir: Path = Path("models")
    assets_dir: Path = Path("assets")
    text2img_model_id: str = "stabilityai/sdxl-turbo"
    use_fp16: bool = True
    enable_xformers: bool = True
    enable_vae_tiling: bool = True
    anthropic_key: Optional[str] = None
    openai_key: Optional[str] = None
    log_dir: Path = Path("logs")
    history_path: Path = Path("logs/history.json")
    default_prompt_style: str = "default"
    metadata: dict[str, str] = field(default_factory=dict)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Return an AppConfig instance, placeholder for future file loading."""
    _ = config_path
    return AppConfig()
