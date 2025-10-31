# 提示词到图像的完整数据流解析（附源码片段）

本文按照执行顺序，梳理提示词从 Gradio 前端输入，经提示词优化、再进入 diffusers 管线并生成图像的全过程。每个阶段都列出关键源码片段，方便对照阅读与深度理解。

---

## 1. 配置加载与模型扫描（`config/settings.py`）

启动时调用 `load_config()`，负责读取 `.env`、设定模型缓存目录，并扫描 `models/` 下的本地权重，生成可供前端选择的模型列表。

```python
# config/settings.py:43-101（节选）
def _discover_local_models(model_dir: Path) -> list[dict[str, str]]:
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
    env_path = Path(config_path) if config_path else Path(".env")
    _load_env_file(env_path)

    model_dir = Path(os.getenv("MODEL_DIR", "models")).expanduser().resolve()
    for env_name in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "DIFFUSERS_CACHE"):
        os.environ.setdefault(env_name, str(model_dir))

    text2img_model_id = os.getenv("TEXT2IMG_MODEL_ID") or str(model_dir / "sdxl-turbo")
    img2img_model_id = os.getenv("IMG2IMG_MODEL_ID") or str(model_dir / "sdxl-turbo")
    discovered_models = _discover_local_models(model_dir)

    metadata: dict[str, Any] = {
        "text2img_model_id": text2img_model_id,
        "img2img_model_id": img2img_model_id,
        "controlnet_models": {
            "canny": str(model_dir / "controlnet-canny"),
            "depth": str(model_dir / "controlnet-depth"),
        },
        "available_models": discovered_models,
    }
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("SILICONFLOW_BASE_URL")
    openai_model = os.getenv("OPENAI_MODEL") or os.getenv("SILICONFLOW_MODEL")
    if openai_base_url:
        metadata["openai_base_url"] = openai_base_url
    if openai_model:
        metadata["openai_model"] = openai_model
    return AppConfig(model_dir=model_dir,
                     text2img_model_id=text2img_model_id,
                     img2img_model_id=img2img_model_id,
                     openai_key=openai_key,
                     metadata=metadata)
```

> **提示**：`metadata["available_models"]` 随后会被前端的模型下拉框使用；`text2img_model_id` / `img2img_model_id` 则用于初始管线选择。

---

## 2. 前端事件绑定（`modules/ui/layout.py`）

Gradio 界面将各类输入控件绑定到回调函数，事件触发后由回调完成业务逻辑。

```python
# modules/ui/layout.py:124-196（节选）
with gr.Tab("文生图"):
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="提示词", lines=4)
            optimized_prompt = gr.Textbox(label="优化后提示词（可编辑）", lines=4)
            optimized_negative = gr.Textbox(label="优化后反向提示词（可编辑）", lines=3)
            with gr.Row():
                style_select = gr.Dropdown(label="风格模板", choices=style_choices,
                                           value=style_choices[0] if style_choices else "default")
                backend_select = gr.Dropdown(label="优化模型", choices=backend_choices,
                                             value=default_backend)
                optimize_btn = gr.Button("提示词优化")
            model_select = gr.Dropdown(label="推理模型",
                                       choices=text_model_labels,
                                       value=default_text_model,
                                       interactive=True)
            generate_btn = gr.Button("生成图像", variant="primary")

    optimize_btn.click(
        fn=callbacks_map["on_optimize_prompt"],
        inputs=[prompt, style_select, backend_select],
        outputs=[optimized_prompt, optimized_negative, status],
    )
    model_select.change(
        fn=callbacks_map["on_change_text_model"],
        inputs=model_select,
        outputs=status,
    )
    generate_btn.click(
        fn=callbacks_map["on_generate_text"],
        inputs=[prompt, optimized_prompt, negative, optimized_negative,
                guidance, steps, seed, height, width, style_select, backend_select],
        outputs=[output_image, status],
    )
```

> **提示**：界面只负责收集输入，真正的控制逻辑（包括模型切换、提示词优化、推理等）全部在 `callbacks_map` 指向的函数里完成。

---

## 3. 提示词优化流程

### 3.1 回调层封装（`modules/ui/callbacks.py`）

```python
# modules/ui/callbacks.py:95-135（节选）
def on_optimize_prompt(prompt: str, style_name: str, model_name: str) -> tuple[str, str, str]:
    style = _resolve_style(style_name或config.default_prompt_style)
    if optimizer is None:
        return prompt, "", "未配置提示词优化服务，已返回原始提示词。"
    try:
        bundle = optimizer.optimize(prompt, style, model=model_name或"claude")
    except Exception as exc:
        return prompt, "", f"提示词优化失败：{exc}"

    optimized_prompt = bundle.optimized或prompt
    negatives: list[str] = []
    if bundle.negative_prompt:
        negatives.append(bundle.negative_prompt.strip())
    if style.negative:
        base_negative = style.negative.strip()
        if base_negative and base_negative not in negatives:
            negatives.append(base_negative)
    optimized_negative = "\n".join(filter(None, negatives))
    return optimized_prompt, optimized_negative, "提示词优化成功，可在生成前继续编辑。"
```

### 3.2 优化器核心（`modules/optimization/prompt_optimizer.py`）

```python
# modules/optimization/prompt_optimizer.py:81-188（节选）
def optimize(self, prompt: str, preset: StylePreset, model: str = "claude") -> PromptBundle:
    positive = preset.positive或""
    negative = preset.negative或""
    combined_prompt = self._compose_prompt(prompt, positive)

    backend = self._backends.get(model.lower())
    backend_result = BackendResult()
    if backend is not None:
        request = BackendRequest(
            prompt_text=combined_prompt,
            original_prompt=prompt,
            positive_style=positive,
            negative_style=negative,
            metadata=self.config.metadata,
        )
        payload = backend(request)
        backend_result = self._normalize_backend_response(payload)
        optimized = backend_result.positive_prompt或combined_prompt
    else:
        optimized = combined_prompt

    backend_negative = backend_result.negative_prompt
    negative_result = backend_negative.strip() if backend_negative else ""
    if not negative_result and negative:
        negative_result = negative.strip()

    return PromptBundle(
        original=prompt,
        optimized=optimized,
        negative_prompt=negative_result或None,
    )
```

```python
# modules/optimization/prompt_optimizer.py:312-371（节选）
def _gpt_backend(request: BackendRequest) -> BackendResult:
    base_prompt = (
        "你是一名视觉提示词优化师，请始终返回 JSON，包含 positive_prompt 和 negative_prompt。"
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": base_prompt},
            {
                "role": "user",
                "content": (
                    f"基础提示：{request.original_prompt}\n"
                    f"风格提示：{request.positive_style}\n"
                    "请输出 JSON 字符串，字段为 positive_prompt 和 negative_prompt。"
                ),
            },
        ],
        max_tokens=512,
        temperature=0.85,
    )
    raw = (completion.choices[0].message.content或"").strip()
    result = self._parse_backend_text(raw)
    if result is None或not (result.positive_prompt或"").strip():
        return BackendResult(positive_prompt="备用的风格化描述 …")
    return result
```

> **提示**：优化器强制 LLM 输出结构化 JSON，并对 ```json 包裹的文本做兼容解析，确保正向与反向提示都能提取。

---

## 4. 文生图推理流程

### 4.1 回调构造请求（`modules/ui/callbacks.py`）

```python
# modules/ui/callbacks.py:143-183（节选）
def on_generate_text(...):
    service = _ensure_text_service()
    if image2img is not None and hasattr(image2img, "offload"):
        image2img.offload()  # 释放图生图管线
    style = _resolve_style(style_name或config.default_prompt_style)
    prompt_for_model = optimized_prompt.strip()或_compose_prompt(prompt, style)
    request = PromptRequest(
        prompt=prompt_for_model,
        negative_prompt=_combine_negative(style, negative_prompt, optimized_negative或None),
        guidance_scale=float(guidance_scale),
        steps=int(steps),
        seed=_normalize_seed(seed),
        height=_normalize_dim(height),
        width=_normalize_dim(width),
    )
    result = service.generate(request)
    info = "生成成功"
    if result.seed is not None:
        info += f"（seed={result.seed}）"
    return result.image, info
```

### 4.2 管线加载与推理（`modules/pipelines/text2img.py`）

```python
# modules/pipelines/text2img.py:65-168（节选）
def set_model(self, model_id: str) -> str:
    if self._pipeline is not None:
        self._pipeline.to("cpu", dtype=torch.float32)
    self._pipeline = None
    self._device = None
    torch.cuda.empty_cache()
    return model_id

def offload(self) -> None:
    if self._pipeline is None:
        return
    self._pipeline.to("cpu", dtype=torch.float32)
    self._pipeline = None
    self._device = None
    torch.cuda.empty_cache()

def load_pipeline(self) -> None:
    if self._pipeline is not None:
        return
    device = self._preferred_device()
    dtype = self._preferred_dtype(device)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        self._resolve_model_id(),
        torch_dtype=dtype,
        cache_dir=str(Path(self.config.model_dir)),
        use_safetensors=True,
    )
    pipeline.to(device)
    if self.config.enable_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    if self.config.enable_vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    self._pipeline = pipeline
    self._device = device

def generate(self, request: PromptRequest) -> ImageResult:
