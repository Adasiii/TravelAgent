from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .config import settings


@dataclass
class ChatMessage:
    role: str
    content: str


class DashScopeLLM:
    def __init__(self, model: str | None = None):
        self.model = model or settings.dashscope_model
        self.enabled = bool(settings.dashscope_api_key)
        self._client = None

        if self.enabled:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=settings.dashscope_api_key,
                    base_url=settings.dashscope_base_url,
                )
            except Exception:
                self.enabled = False

    def chat(self, messages: list[ChatMessage], temperature: float = 0.3) -> str:
        if not self.enabled or self._client is None:
            return self._mock_response(messages)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            return self._mock_response(messages, error=str(exc))

    def stream_chat(self, messages: list[ChatMessage], temperature: float = 0.3) -> Iterator[str]:
        if not self.enabled or self._client is None:
            yield self._mock_response(messages)
            return

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
        except Exception as exc:
            yield self._mock_response(messages, error=str(exc))

    def _mock_response(self, messages: list[ChatMessage], error: str = "") -> str:
        last = messages[-1].content if messages else ""
        error_text = f"\n降级原因：{error[:300]}\n" if error else ""
        return (
            "当前未检测到可用 DASHSCOPE_API_KEY，或模型请求失败，已进入 mock 模式。\n"
            f"{error_text}"
            "我会基于输入生成一个可替换的示例行程。\n\n"
            f"用户需求摘要：{last[:500]}"
        )
