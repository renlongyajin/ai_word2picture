"""Gradio layout composition."""

from __future__ import annotations

from typing import Any

try:
    import gradio as gr
except ImportError:  # pragma: no cover - optional dependency not installed yet
    gr = None  # type: ignore

from config.settings import AppConfig
from modules.pipelines.text2img import Text2ImageService
from modules.ui.callbacks import build_callbacks


def build_app(config: AppConfig) -> Any:
    """Compose and return the Gradio application."""
    if gr is None:
        raise RuntimeError("Gradio is not available. Install requirements first.")

    text_service = Text2ImageService(config=config)
    cbs = build_callbacks(config, text2img=text_service)

    with gr.Blocks(title="AI Creative Image Assistant") as demo:
        gr.Markdown("# AI 创意图像助手\n输入提示词生成图像。")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="提示词", lines=4, placeholder="描述你想要生成的图像")
                negative = gr.Textbox(label="反向提示词", lines=2, placeholder="不希望出现的元素")
                guidance = gr.Slider(label="引导系数", minimum=1.0, maximum=15.0, step=0.5, value=7.5)
                steps = gr.Slider(label="采样步数", minimum=10, maximum=50, step=1, value=30)
                seed = gr.Number(label="随机种子（可选）", precision=0)
                height = gr.Slider(label="高度", minimum=256, maximum=1024, step=64, value=512)
                width = gr.Slider(label="宽度", minimum=256, maximum=1024, step=64, value=512)
                generate_btn = gr.Button("生成图像", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="生成结果", type="pil")
                status = gr.Markdown("准备就绪。")

        generate_btn.click(
            fn=cbs["on_generate_text"],
            inputs=[prompt, negative, guidance, steps, seed, height, width],
            outputs=[output_image, status],
        )

    return demo
