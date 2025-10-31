"""One-off script for debugging image-to-image generation."""

from pathlib import Path
from PIL import Image

from config.settings import load_config
from modules.optimization.prompt_optimizer import PromptOptimizer
from modules.optimization.style_presets import StylePreset, StylePresetRegistry
from modules.pipelines.img2img import Image2ImageService
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
    img_service = Image2ImageService(config)

    callbacks = build_callbacks(
        config,
        optimizer=optimizer,
        text2img=text_service,
        image2img=img_service,
        style_registry=style_registry,
    )

    # 2. 准备测试输入（请按需替换）
    init_image_path = Path("tests/assets/debug_input.png")
    if not init_image_path.exists():
        raise FileNotFoundError(f"缺少初始图像: {init_image_path}")

    init_image = Image.open(init_image_path)

    prompt = "一位穿黑色机甲的少女站在霓虹灯闪烁的未来城市街头"
    optimized_prompt = prompt  # 若先调用 on_optimize_prompt，可替换为结果
    negative_prompt = "low quality, watermark"
    optimized_negative = "jpeg artifacts, blurry"

    # ControlNet 参数示例（此处不使用，可按需填写）
    control_type = ""
    control_image = None

    # 3. 调用图生图回调，执行真实推理
    cb = callbacks["on_generate_image"]
    image, status = cb(
        init_image=init_image,
        prompt=prompt,
        optimized_prompt=optimized_prompt,
        negative_prompt=negative_prompt,
        optimized_negative=optimized_negative,
        strength=0.6,
        guidance_scale=7.5,
        steps=30,
        seed=42,
        style_name="debug-style",
        model_name="gpt",  # 若已先做优化，可忽略
        control_type_name=control_type,
        control_image=control_image,
        conditioning_scale=1.0,
        guess_mode=False,
    )

    print("状态:", status)
    if image:
        out_path = Path("debug_img2img_output.png")
        image.save(out_path)
        print("图像已保存:", out_path.resolve())
    else:
        print("没有生成图像，请检查日志。")


if __name__ == "__main__":
    main()

