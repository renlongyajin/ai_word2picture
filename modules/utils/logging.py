"""Logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import AppConfig


def setup_logging(config: AppConfig) -> logging.Logger:
    """Configure and return a logger."""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "application.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("ai_word2picture")
