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
from modules.pipelines.controlnet_manager import ControlType
from modules.pipelines.img2img import Image2ImageService
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


def _control_choices() -> Sequence[str]:
    return [""] + [control_type.value for control_type in ControlType]


def build_app(config: AppConfig) -> Any:
    """Compose and return the Gradio application."""
    if gr is None:
        raise RuntimeError("Gradio 未安装，请先执行依赖安装。")

    text_service = Text2ImageService(config=config)
    image_service = Image2ImageService(config=config)
    optimizer = PromptOptimizer(config)
    style_registry = _load_style_registry(config)

    callbacks_map = build_callbacks(
        config,
        optimizer=optimizer,
        text2img=text_service,
        image2img=image_service,
        style_registry=style_registry,
    )

    style_choices = _style_choices(style_registry)
    backend_choices = ["claude", "gpt"]
    control_choices = _control_choices()

    with gr.Blocks(title="AI Creative Image Assistant") as demo:
        gr.Markdown("## AI 创意图像助手")

        with gr.Tab("文生图"):
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

        with gr.Tab("图生图"):
            with gr.Row():
                with gr.Column():
                    init_image = gr.Image(label="初始图像", type="pil")
                    control_image = gr.Image(label="ControlNet 参考图像（可选）", type="pil")
                    prompt_img = gr.Textbox(label="提示词", lines=4, placeholder="描述目标风格…")
                    negative_img = gr.Textbox(label="反向提示词", lines=2)
                    style_select_img = gr.Dropdown(
                        label="风格模板",
                        choices=style_choices,
                        value=style_choices[0] if style_choices else "default",
                    )
                    use_optimizer_img = gr.Checkbox(label="启用 Prompt 优化", value=False)
                    backend_select_img = gr.Dropdown(
                        label="优化模型",
                        choices=backend_choices,
                        value=backend_choices[0],
                    )
                    strength = gr.Slider(
                        label="重绘强度",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.6,
                    )
                    guidance_img = gr.Slider(
                        label="引导系数",
                        minimum=1.0,
                        maximum=15.0,
                        step=0.5,
                        value=7.5,
                    )
                    steps_img = gr.Slider(
                        label="采样步数",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=30,
                    )
                    seed_img = gr.Number(label="随机种子（可选）", precision=0)
                    control_select = gr.Dropdown(
                        label="ControlNet 类型",
                        choices=control_choices,
                        value="",
                    )
                    conditioning_scale = gr.Slider(
                        label="ControlNet 影响权重",
                        minimum=0.1,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                    )
                    guess_mode = gr.Checkbox(label="Guess Mode", value=False)
                    generate_img_btn = gr.Button("生成图生图结果", variant="primary")

                with gr.Column():
                    output_image_img = gr.Image(label="生成结果", type="pil")
                    status_img = gr.Markdown("准备就绪。")

            generate_img_btn.click(
                fn=callbacks_map["on_generate_image"],
                inputs=[
                    init_image,
                    prompt_img,
                    negative_img,
                    strength,
                    guidance_img,
                    steps_img,
                    seed_img,
                    style_select_img,
                    use_optimizer_img,
                    backend_select_img,
                    control_select,
                    control_image,
                    conditioning_scale,
                    guess_mode,
                ],
                outputs=[output_image_img, status_img],
            )

    return demo
