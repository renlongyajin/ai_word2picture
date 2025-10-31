"""One-off script for debugging text-to-image generation."""

from pathlib import Path

from config.settings import load_config
from modules.optimization.prompt_optimizer import PromptOptimizer
from modules.optimization.style_presets import StylePreset, StylePresetRegistry
from modules.pipelines.text2img import Text2ImageService
from modules.ui.callbacks import build_callbacks


def main() -> None:
    # 1. 准备真实配置与服务对象
    config = load_config()

    style_registry = StylePresetRegistry()
    style_registry.add(
        StylePreset(
            name="debug-style",
            positive="cinematic lighting, high detail",
            negative="low quality, watermark",
        )
    )

    optimizer = PromptOptimizer(config)
    text_service = Text2ImageService(config)

    callbacks = build_callbacks(
        config,
        optimizer=optimizer,
        text2img=text_service,
        image2img=None,  # 调试文生图即可，不必注入图生图服务
        style_registry=style_registry,
    )

    # 2. 准备提示词；若需观察优化效果，可先调用 on_optimize_prompt
    prompt = "夕阳下的未来城市街景，穿红色和服的少女，霓虹灯闪烁"
    optimized_prompt = prompt  # 如果先调用 on_optimize_prompt，则替换为返回值
    negative_prompt = "low quality, watermark"
    optimized_negative = "jpeg artifacts, blurry"

    # 3. 调用文生图回调，执行真实推理
    cb = callbacks["on_generate_text"]
    image, status = cb(
        prompt=prompt,
        optimized_prompt=optimized_prompt,
        negative_prompt=negative_prompt,
        optimized_negative=optimized_negative,
        guidance_scale=7.5,
        steps=30,
        seed=42,
        height=768,
        width=512,
        style_name="debug-style",
        model_name="gpt",  # 如果先跑过优化，可忽略
    )

    print("状态:", status)
    if image:
        out_path = Path("debug_text2img_output.png")
        image.save(out_path)
        print("图像已保存:", out_path.resolve())
    else:
        print("未返回图像，请检查状态信息。")


if __name__ == "__main__":
    main()

