"""Prompt optimization via third-party LLM APIs."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from config.settings import AppConfig
from modules.optimization.style_presets import StylePreset


@dataclass(slots=True)
class PromptBundle:
    """Container for optimized prompt data."""

    original: str
    optimized: str
    negative_prompt: Optional[str] = None


@dataclass(slots=True)
class BackendRequest:
    """Information passed to backend optimizers."""

    prompt_text: str
    original_prompt: str
    positive_style: str
    negative_style: str
    metadata: Dict[str, Any]


BackendCallable = Callable[[BackendRequest], str]


class PromptOptimizer:
    """Interface to Claude/GPT prompt enhancement."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._backends: Dict[str, BackendCallable] = {}
        self.warnings: list[str] = []
        self._auto_register_backends()

    def register_backend(self, name: str, backend: BackendCallable) -> None:
        """Register a prompt optimization backend."""
        self._backends[name.lower()] = backend

    def clear_backends(self) -> None:
        """Remove all backends (mainly for tests)."""
        self._backends.clear()

    def has_backend(self, name: str) -> bool:
        """Return True when backend exists."""
        return name.lower() in self._backends

    def available_backends(self) -> list[str]:
        """Return the list of registered backends ordered by preference."""
        priority = {"gpt": 0, "claude": 1}
        return sorted(
            (backend for backend in self._backends.keys()),
            key=lambda item: (priority.get(item, 99), item),
        )

    def default_backend(self) -> str:
        """Return the preferred backend name, falling back to Claude."""
        choices = self.available_backends()
        if choices:
            return choices[0]
        return "claude"

    def optimize(self, prompt: str, preset: StylePreset, model: str = "claude") -> PromptBundle:
        """Return the optimized prompt bundle."""
        positive = preset.positive or ""
        negative = preset.negative or ""
        combined_prompt = self._compose_prompt(prompt, positive)

        backend = self._backends.get(model.lower())
        if backend is not None:
            request = BackendRequest(
                prompt_text=combined_prompt,
                original_prompt=prompt,
                positive_style=positive,
                negative_style=negative,
                metadata=self.config.metadata,
            )
            optimized = backend(request)
        else:
            if model.lower() == "gpt":
                if self.config.openai_key:
                    message = (
                        "OpenAI SDK 未安装或初始化失败："
                        + "; ".join(self.warnings)
                        if self.warnings
                        else "OpenAI 后端不可用，请检查依赖。"
                    )
                    raise RuntimeError(message)
            if model.lower() == "claude":
                if self.config.anthropic_key:
                    message = (
                        "Anthropic SDK 未安装或初始化失败："
                        + "; ".join(self.warnings)
                        if self.warnings
                        else "Claude 后端不可用，请检查依赖。"
                    )
                    raise RuntimeError(message)
            optimized = combined_prompt

        return PromptBundle(
            original=prompt,
            optimized=optimized,
            negative_prompt=negative or None,
        )

    # Internal helpers ---------------------------------------------------------
    def _compose_prompt(self, original: str, positive: str) -> str:
        parts = [original.strip()]
        if positive:
            parts.append(f"风格提示：{positive.strip()}")
        return "\n".join(part for part in parts if part)

    def _auto_register_backends(self) -> None:
        """Register backends automatically when dependencies are available."""
        self._register_claude_backend()
        self._register_openai_backend()

    def _register_claude_backend(self) -> None:
        if not self.config.anthropic_key:
            return
        try:
            anthropic_module = importlib.import_module("anthropic")
        except ImportError as exc:  # pragma: no cover - optional dependency
            self.warnings.append(f"无法导入 anthropic：{exc}")
            return

        client = anthropic_module.Anthropic(api_key=self.config.anthropic_key)

        def _claude_backend(request: BackendRequest) -> str:
            message = client.messages.create(
                model=self.config.metadata.get("claude_model", "claude-3-haiku-20240307"),
                max_tokens=256,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一名提示词打磨专家，请在保持语义一致的情况下，使提示词更具画面感与艺术性。"
                            " 使用中文输出，并强化氛围与风格描述。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"基础提示：{request.original_prompt}\n"
                            f"风格提示：{request.positive_style}"
                        ),
                    },
                ],
            )
            return message.content[0].text if message.content else request.prompt_text

        self.register_backend("claude", _claude_backend)

    def _extract_openai_text(self, completion: Any, fallback: str) -> str:
        if hasattr(completion, "output_text") and completion.output_text:
            return str(completion.output_text)

        output = getattr(completion, "output", None)
        if output:
            parts = []
            for item in output:
                if getattr(item, "type", "") == "message":
                    for content in getattr(item, "content", []):
                        if getattr(content, "type", "") == "text":
                            parts.append(getattr(content, "text", ""))
            joined = "\n".join(parts).strip()
            if joined:
                return joined

        choices = getattr(completion, "choices", None)
        if choices:
            text = choices[0].message.get("content")  # type: ignore[index]
            if isinstance(text, str) and text.strip():
                return text

        return fallback

    def _register_openai_backend(self) -> None:
        if not self.config.openai_key:
            return
        try:
            openai_module = importlib.import_module("openai")
        except ImportError as exc:
            self.warnings.append(f"无法导入 openai：{exc}")
            return

        base_url = self.config.metadata.get("openai_base_url")
        client_kwargs = {"api_key": self.config.openai_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = openai_module.OpenAI(**client_kwargs)

        def _gpt_backend(request: BackendRequest) -> str:
            base_prompt = (
                "你是一名资深视觉提示词优化师，专门负责文生图相关业务，你需要扩展用户的提示词，将其更改为具体、细节、整洁的提示词，便于文生图的进行"
                " 禁止简单重复原句。"
            )

            model_name = self.config.metadata.get("openai_model", "gpt-4o-mini")

            if base_url:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": base_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"基础提示：{request.original_prompt}\n"
                                f"风格提示：{request.positive_style}\n"
                                "请输出一个多句的优化提示词，可分多段，但不要枚举列表。"
                            ),
                        },
                    ],
                    max_tokens=512,
                    temperature=0.85,
                )
                if completion.choices:
                    optimized = (completion.choices[0].message.content or "").strip()
                else:
                    optimized = ""
            else:
                completion = client.responses.create(
                    model=model_name,
                    input=[
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": base_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"基础提示：{request.original_prompt}\n"
                                        f"风格提示：{request.positive_style}\n"
                                        "请输出一个多句的优化提示词，可分多段，但不要枚举列表。"
                                    ),
                                }
                            ],
                        },
                    ],
                    max_output_tokens=512,
                    temperature=0.85,
                )
                optimized = self._extract_openai_text(completion, request.prompt_text).strip()

            if optimized.lower() in {
                request.original_prompt.strip().lower(),
                request.prompt_text.strip().lower(),
            }:
                enriched = (
                    f"{request.original_prompt.strip()}，背景延伸到暮色城市屋顶的广阔视角，"
                    "远处霓虹与柔雾交织，烘托温暖而灵动的氛围。"
                    "镜头从低角度捕捉白猫轻盈的起跳动作，细腻的毛发被金橙色光晕勾勒，"
                    "毛尖反射出偏粉与紫罗兰的高光。空气里漂浮细小光尘，增强梦幻质感。"
                    f"\n风格提示：{request.positive_style or 'cinematic lighting'}"
                )
                return enriched

            return optimized

        self.register_backend("gpt", _gpt_backend)
