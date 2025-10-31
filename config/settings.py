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
        os.environ[key.strip()] = value.strip()


def _discover_local_models(model_dir: Path) -> list[dict[str, str]]:
    """Enumerate locally available diffusion models for selection."""
    choices: list[dict[str, str]] = []
    if not model_dir.exists():
        return choices

    for child in sorted(model_dir.iterdir()):
        if not child.is_dir():
            continue
        has_index = (child / "model_index.json").exists()
        has_weights = any(child.glob("*.safetensors")) or any(child.glob("*.bin"))
        if has_index or has_weights:
            choices.append({"label": child.name, "value": str(child)})
    return choices


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Return an AppConfig instance with environment-aware settings."""
    env_path = Path(config_path) if config_path else Path(".env")
    _load_env_file(env_path)

    model_dir_env = os.getenv("MODEL_DIR", "models")
    model_dir = Path(model_dir_env).expanduser().resolve()
    for env_name in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "DIFFUSERS_CACHE"):
        os.environ.setdefault(env_name, str(model_dir))

    text2img_override = os.getenv("TEXT2IMG_MODEL_ID")
    img2img_override = os.getenv("IMG2IMG_MODEL_ID")
    default_model_path = str(model_dir / "sdxl-turbo")
    text2img_model_id = text2img_override or default_model_path
    img2img_model_id = img2img_override or default_model_path

    discovered_models = _discover_local_models(model_dir)
    available_models: list[dict[str, str]] = []
    seen_values: set[str] = set()

    def _add_option(label: str, value: str) -> None:
        if not value or value in seen_values:
            return
        available_models.append({"label": label, "value": value})
        seen_values.add(value)

    for item in discovered_models:
        _add_option(item["label"], item["value"])

    default_label_text = Path(text2img_model_id).name if "/" not in text2img_model_id else text2img_model_id
    default_label_img = Path(img2img_model_id).name if "/" not in img2img_model_id else img2img_model_id
    _add_option(default_label_text, text2img_model_id)
    _add_option(default_label_img, img2img_model_id)

    metadata: dict[str, Any] = {
        "text2img_model_id": text2img_model_id,
        "img2img_model_id": img2img_model_id,
        "controlnet_models": {
            "canny": str(model_dir / "controlnet-canny"),
            "depth": str(model_dir / "controlnet-depth"),
        },
        "available_models": available_models,
    }

    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("SILICONFLOW_BASE_URL")
    openai_model = os.getenv("OPENAI_MODEL") or os.getenv("SILICONFLOW_MODEL")

    if openai_base_url:
        metadata["openai_base_url"] = openai_base_url
    if openai_model:
        metadata["openai_model"] = openai_model
    return AppConfig(
        model_dir=model_dir,
        text2img_model_id=text2img_model_id,
        img2img_model_id=img2img_model_id,
        anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_key=openai_key,
        metadata=metadata,
    )
