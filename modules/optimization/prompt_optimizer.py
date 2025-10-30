"""Prompt optimization via third-party LLM APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

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
    """Information passed to backend优化器."""

    prompt_text: str
    original_prompt: str
    positive_style: str
    negative_style: str
    metadata: Dict[str, str]


BackendCallable = Callable[[BackendRequest], str]


class PromptOptimizer:
    """Interface to Claude/GPT prompt enhancement."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._backends: Dict[str, BackendCallable] = {}
        self._auto_register_backends()

    def register_backend(self, name: str, backend: BackendCallable) -> None:
        """Register a prompt优化后端。"""
        self._backends[name.lower()] = backend

    def clear_backends(self) -> None:
        """移除所有后端（主要用于测试重置状态）。"""
        self._backends.clear()

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
            optimized = combined_prompt

        return PromptBundle(
            original=prompt,
            optimized=optimized,
            negative_prompt=negative or None,
        )

    # 内部工具方法 ---------------------------------------------------------
    def _compose_prompt(self, original: str, positive: str) -> str:
        parts = [original.strip()]
        if positive:
            parts.append(f"风格提示：{positive.strip()}")
        return "\n".join(part for part in parts if part)

    def _auto_register_backends(self) -> None:
        """根据现有依赖自动注册后端（如果可用）。"""
        # Anthropic / Claude
        if self.config.anthropic_key:
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=self.config.anthropic_key)

                def _claude_backend(request: BackendRequest) -> str:
                    message = client.messages.create(
                        model=self.config.metadata.get(
                            "claude_model", "claude-3-haiku-20240307"
                        ),
                        max_tokens=256,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "请根据提供的提示词生成更具表现力的描述，用中文输出。"
                                    " 保留关键信息并加强风格描述。"
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
            except Exception:
                # 如果依赖不可用或初始化失败，则跳过自动注册
                pass

        # OpenAI / GPT
        if self.config.openai_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.config.openai_key)

                def _gpt_backend(request: BackendRequest) -> str:
                    completion = client.responses.create(
                        model=self.config.metadata.get("openai_model", "gpt-4o-mini"),
                        input=(
                            "请改写以下提示词，使其更加细腻且具备艺术表达力，"
                            "并保持中文输出。\n"
                            f"基础提示：{request.original_prompt}\n"
                            f"风格提示：{request.positive_style}"
                        ),
                        max_output_tokens=256,
                    )
                    return (
                        completion.output_text if hasattr(completion, "output_text") else request.prompt_text
                    )

                self.register_backend("gpt", _gpt_backend)
            except Exception:
                pass
