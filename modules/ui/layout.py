"""Gradio layout composition."""

from __future__ import annotations

from typing import Any

try:
    import gradio as gr
except ImportError:  # pragma: no cover - optional dependency not installed yet
    gr = None  # type: ignore

from config.settings import AppConfig


def build_app(config: AppConfig) -> Any:
    """Compose and return the Gradio application."""
    if gr is None:
        raise RuntimeError("Gradio is not available. Install requirements first.")

    with gr.Blocks(title="AI Creative Image Assistant") as demo:
        gr.Markdown("# AI Creative Image Assistant\n项目初始化中，功能即将上线。")

    return demo
