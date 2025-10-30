"""Application entry point for the AI Word2Picture project."""

from __future__ import annotations

from typing import Optional

from config.settings import load_config
from modules.ui.layout import build_app


def main(config_path: Optional[str] = None) -> None:
    """Load configuration and launch the Gradio interface."""
    config = load_config(config_path)
    app = build_app(config)
    app.queue()
    app.launch(share=False, inbrowser=False)


if __name__ == "__main__":
    main()
