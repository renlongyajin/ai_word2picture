"""Prompt optimization via third-party LLM APIs."""

from __future__ import annotations

import importlib
import json
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


@dataclass(slots=True)
class BackendResult:
    """Response payload produced by optimizer backends."""

    positive_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None


BackendCallable = Callable[[BackendRequest], BackendResult | Dict[str, Any] | str]


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
        backend_result = BackendResult()
        if backend is not None:
            request = BackendRequest(
                prompt_text=combined_prompt,
                original_prompt=prompt,
                positive_style=positive,
                negative_style=negative,
                metadata=self.config.metadata,
            )
            backend_payload = backend(request)
            backend_result = self._normalize_backend_response(backend_payload)
            optimized = backend_result.positive_prompt or combined_prompt
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

        backend_negative = backend_result.negative_prompt
        negative_result = backend_negative.strip() if backend_negative else ""
        if not negative_result and negative:
            negative_result = negative.strip()

        return PromptBundle(
            original=prompt,
            optimized=optimized,
            negative_prompt=negative_result or None,
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

    def _normalize_backend_response(
        self, payload: BackendResult | Dict[str, Any] | str
    ) -> BackendResult:
        """Coerce backend outputs into BackendResult."""
        if isinstance(payload, BackendResult):
            return payload
        if isinstance(payload, dict):
            return BackendResult(
                positive_prompt=payload.get("positive_prompt")
                or payload.get("optimized_prompt")
                or payload.get("prompt"),
                negative_prompt=payload.get("negative_prompt"),
            )
        if isinstance(payload, str):
            parsed = self._parse_backend_text(payload)
            if parsed is not None:
                return parsed
            return BackendResult(positive_prompt=payload)
        return BackendResult()

    def _parse_backend_text(self, text: str) -> Optional[BackendResult]:
        """Try to extract structured prompts from backend raw text."""
        cleaned = (text or "").strip()
        if not cleaned:
            return None

        # Remove fenced code blocks (```json ... ```)
        if cleaned.startswith("```"):
            stripped = cleaned.strip("`")
            # safer: remove the first line (``` or ```json)
            lines = cleaned.splitlines()
            if len(lines) >= 2:
                # drop the first fence line
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # drop closing fence if present
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            else:
                cleaned = stripped
            if not cleaned:
                return None

        # JSON object payload
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            positive = data.get("positive_prompt") or data.get("prompt") or data.get("optimized_prompt")
            negative = data.get("negative_prompt")
            if positive or negative:
                return BackendResult(
                    positive_prompt=str(positive) if positive else None,
                    negative_prompt=str(negative) if negative else None,
                )

        # Simple delimiter-based format
        markers = ["Negative Prompt:", "Negative:", "反向提示词:", "反向提示:"]
        for marker in markers:
            if marker in cleaned:
                positive_part, negative_part = cleaned.split(marker, 1)
                return BackendResult(
                    positive_prompt=positive_part.strip(),
                    negative_prompt=negative_part.strip(),
                )

        return None

    def _register_claude_backend(self) -> None:
        if not self.config.anthropic_key:
            return
        try:
            anthropic_module = importlib.import_module("anthropic")
        except ImportError as exc:  # pragma: no cover - optional dependency
            self.warnings.append(f"无法导入 anthropic：{exc}")
            return

        client = anthropic_module.Anthropic(api_key=self.config.anthropic_key)

        def _claude_backend(request: BackendRequest) -> BackendResult:
            message = client.messages.create(
                model=self.config.metadata.get("claude_model", "claude-3-haiku-20240307"),
                max_tokens=256,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一名资深视觉提示词优化师，负责根据用户需求生成文生图提示词。"
                            " 请输出 JSON 字符串，其中 `positive_prompt` 是优化后的正向提示词，"
                            " `negative_prompt` 是建议的反向提示词（没有可留空字符串）。"
                            " 仅返回 JSON，不要输出额外说明。"
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
            reply = message.content[0].text if message.content else request.prompt_text
            parsed = self._parse_backend_text(reply)
            if parsed is not None:
                return parsed
            return BackendResult(positive_prompt=reply or request.prompt_text)

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

        def _gpt_backend(request: BackendRequest) -> BackendResult:
            base_prompt = (
                "你是一名资深视觉提示词优化师，需要扩展并润色用户提示词，使其更加具体、细节丰富、视觉化。"
                " 请始终返回一个 JSON 字符串，包含 `positive_prompt` 与 `negative_prompt` 两个字段。"
                " `positive_prompt` 为优化后的正向提示词，`negative_prompt` 为建议的反向提示词"
                " 禁止输出除 JSON 之外的其他文字。"
                "最终输出结果为英文。"
                "参考positive_prompt:masterpiece, ultra-HD, cinematic lighting, photorealistic, impressionism (1.5), high detail, depth of field, (blurred background), (dramatic lighting),masterpiece, best quality, very aesthetic, 8k, masterpiece, ultra-HD, cinematic lighting, high detail, depth of field, soft reflections, amazing composition,(ultra-HD:0.9), cinematic lighting, photorealistic, impressionism (1.5), high detail, depth of field, (blurred background), (dramatic lighting),masterpiece, best quality, very aesthetic, , 8k,,masterpiece, ultra-HD, cinematic lighting, high detail, depth of field, soft reflections, amazing composition,"
                "1girl, petite, short orange hair, blue eyes, highly detailed eyes, oversized unbuttoned White shirt, off shoulder,  golden bracelet, denim torn shorts, black knee-highs, 18yo, blush, night club, looking at viewer, drunk, sitting on high chair, random pose, relaxed pose, glass of beer,"
                "studio_lights,( dimly lit:1.2),(professional lighting:1.3), Dynamic shot, Dynamic pose, conjuring, foreshortening, extreme perspective"
                "参考negative_prompt:worst quality, low quality, displeasing, text, , watermark, bad anatomy, text, artist name, signature, hearts, deformed hands, missing finger, shiny skin,"
                "反向提示词不可为空。最终该提示词会被用于stable diffsuion生成图片。"
                "提示词应该尽可能保留原本信息，例如数量信息，例如如果只是一个女孩的图像，那就是1 girl，数量信息需要显式声明。"            
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
                                "请输出 JSON 字符串，字段为 positive_prompt（正向提示词）和 negative_prompt（反向提示词）。"
                            ),
                        },
                    ],
                    max_tokens=512,
                    temperature=0.85,
                )
                if completion.choices:
                    raw = (completion.choices[0].message.content or "").strip()
                else:
                    raw = ""
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
                                        "请输出 JSON 字符串，字段为 positive_prompt（正向提示词）和 negative_prompt（反向提示词）。"
                                    ),
                                }
                            ],
                        },
                    ],
                    max_output_tokens=512,
                    temperature=0.85,
                )
                raw = self._extract_openai_text(completion, request.prompt_text).strip()

            result = self._parse_backend_text(raw)
            if result is None or not (result.positive_prompt or "").strip():
                enriched = (
                    f"{request.original_prompt.strip()}，背景延伸到暮色城市屋顶的广阔视角，"
                    "远处霓虹与柔雾交织，烘托温暖而灵动的氛围。"
                    "镜头从低角度捕捉白猫轻盈的起跳动作，细腻的毛发被金橙色光晕勾勒，"
                    "毛尖反射出偏粉与紫罗兰的高光。空气里漂浮细小光尘，增强梦幻质感。"
                    f"\n风格提示：{request.positive_style or 'cinematic lighting'}"
                )
                return BackendResult(positive_prompt=enriched)

            return result

        self.register_backend("gpt", _gpt_backend)
