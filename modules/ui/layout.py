"""Gradio layout composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None  # type: ignore

from config.settings import AppConfig
from modules.optimization.prompt_optimizer import PromptOptimizer
from modules.optimization.style_presets import StylePreset, StylePresetRegistry
from modules.pipelines.text2img import Text2ImageService
from modules.ui.callbacks import build_callbacks


def _load_style_registry(config: AppConfig) -> StylePresetRegistry:
    registry = StylePresetRegistry()
    styles_path = Path(config.assets_dir) / "styles.json"
    registry.load_from_file(styles_path)
    if not registry.list_presets():
        registry.add(StylePreset(name="default", positive="", negative=""))
    return registry


def _style_choices(registry: StylePresetRegistry) -> Sequence[str]:
    names = [preset.name for preset in registry.list_presets()]
    if "default" not in names:
        names.insert(0, "default")
    return names


def build_app(config: AppConfig) -> Any:
    """Compose and return the Gradio application."""
    if gr is None:
        raise RuntimeError("Gradio is not available. Install requirements first.")

    text_service = Text2ImageService(config=config)
    optimizer = PromptOptimizer(config)
    style_registry = _load_style_registry(config)

    callbacks_map = build_callbacks(
        config,
        optimizer=optimizer,
        text2img=text_service,
        style_registry=style_registry,
    )

    style_choices = _style_choices(style_registry)
    backend_choices = ["claude", "gpt"]

    with gr.Blocks(title="AI Creative Image Assistant") as demo:
        gr.Markdown("## AI 创意图像助手")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="提示词",
                    lines=4,
                    placeholder="描述你想要生成的图像…",
                )
                negative = gr.Textbox(
                    label="反向提示词",
                    lines=2,
                    placeholder="不希望出现的元素…",
                )
                style_select = gr.Dropdown(
                    label="风格模板",
                    choices=style_choices,
                    value=style_choices[0] if style_choices else "default",
                )
                use_optimizer = gr.Checkbox(label="启用 Prompt 优化", value=False)
                backend_select = gr.Dropdown(
                    label="优化模型",
                    choices=backend_choices,
                    value=backend_choices[0],
                )
                guidance = gr.Slider(
                    label="引导系数",
                    minimum=1.0,
                    maximum=15.0,
                    step=0.5,
                    value=7.5,
                )
                steps = gr.Slider(
                    label="采样步数",
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=30,
                )
                seed = gr.Number(label="随机种子（可选）", precision=0)
                height = gr.Slider(
                    label="图像高度",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=512,
                )
                width = gr.Slider(
                    label="图像宽度",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=512,
                )
                generate_btn = gr.Button("生成图像", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="生成结果", type="pil")
                status = gr.Markdown("准备就绪。")

        generate_btn.click(
            fn=callbacks_map["on_generate_text"],
            inputs=[
                prompt,
                negative,
                guidance,
                steps,
                seed,
                height,
                width,
                style_select,
                use_optimizer,
                backend_select,
            ],
            outputs=[output_image, status],
        )

    return demo
