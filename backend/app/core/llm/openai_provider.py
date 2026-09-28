"""OpenAI 兼容通道（云端 API，或本地 vLLM / LM Studio / one-api 网关）。

只实现 /chat/completions 这个事实标准，因此 DeepSeek、通义、Moonshot、
OpenAI、本地 vLLM 都能直接接进来。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from app.core.llm.base import (
    ChatMessage,
    ChatResult,
    LLMError,
    LLMProvider,
    StreamChunk,
)


class OpenAICompatProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.deepseek.com/v1",
        api_key: str = "",
        timeout: float = 300.0,
        **options: Any,
    ) -> None:
        super().__init__(model, timeout=timeout, **options)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.timeout, connect=10.0)

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stream:
            # 让服务端在最后一片里带上 usage；不支持的服务端会忽略该字段
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _require_key(self) -> None:
        if not self.api_key:
            raise LLMError(
                "云端通道未配置 API Key。请在「设置」页填入，或改用本地 Ollama 通道。",
                provider=self.name,
            )

    def _explain_http_error(self, status: int, body: str) -> str:
        body_short = body.strip()[:300]
        if status == 401:
            return f"云端通道鉴权失败（401），API Key 可能无效或已过期。原始信息：{body_short}"
        if status == 404:
            return f"云端通道找不到模型 `{self.model}`（404）。原始信息：{body_short}"
        if status == 429:
            return f"云端通道触发限流（429），请稍后重试或降低并发。原始信息：{body_short}"
        return f"云端通道返回 HTTP {status}：{body_short}"

    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self._require_key()
        payload = self._payload(messages, False, temperature, max_tokens)
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
        except httpx.HTTPError as exc:
            raise self._friendly_connect_error(exc, "云端 LLM", self.base_url) from exc

        if resp.status_code >= 400:
            raise LLMError(
                self._explain_http_error(resp.status_code, resp.text),
                provider=self.name,
                status=resp.status_code,
            )

        data = resp.json()
        choices = data.get("choices") or []
        content = (choices[0].get("message") or {}).get("content", "") if choices else ""
        usage = data.get("usage") or {}
        return ChatResult(
            content=content or "",
            provider=self.name,
            model=data.get("model", self.model),
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    # ------------------------------------------------------------------
    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self._require_key()
        payload = self._payload(messages, True, temperature, max_tokens)
        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                ) as resp:
                    if resp.status_code >= 400:
                        raw = (await resp.aread()).decode("utf-8", "replace")
                        raise LLMError(
                            self._explain_http_error(resp.status_code, raw),
                            provider=self.name,
                            status=resp.status_code,
                        )

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        usage = data.get("usage") or {}
                        choices = data.get("choices") or []
                        text = ""
                        finish_reason = None
                        if choices:
                            delta = choices[0].get("delta") or {}
                            text = delta.get("content") or ""
                            finish_reason = choices[0].get("finish_reason")

                        # usage 片通常 choices 为空，单独处理
                        if text:
                            yield StreamChunk(text=text)
                        if usage:
                            yield StreamChunk(
                                finished=True,
                                prompt_tokens=usage.get("prompt_tokens"),
                                completion_tokens=usage.get("completion_tokens"),
                            )
                        elif finish_reason:
                            yield StreamChunk(finished=True)
        except httpx.HTTPError as exc:
            raise self._friendly_connect_error(exc, "云端 LLM", self.base_url) from exc

    # ------------------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        if not self.api_key:
            return {
                "ok": False,
                "detail": "云端通道未配置 API Key",
                "models": [],
                "base_url": self.base_url,
            }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=8.0)) as client:
                resp = await client.get(f"{self.base_url}/models", headers=self._headers())
            if resp.status_code >= 400:
                return {
                    "ok": False,
                    "detail": self._explain_http_error(resp.status_code, resp.text),
                    "models": [],
                    "base_url": self.base_url,
                }
            models = [m.get("id", "") for m in (resp.json().get("data") or [])]
            return {
                "ok": True,
                "detail": "云端通道可用",
                "models": models,
                "has_configured_model": self.model in models,
                "base_url": self.base_url,
            }
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "detail": f"无法连接云端通道（{self.base_url}）：{type(exc).__name__}: {exc}",
                "models": [],
                "base_url": self.base_url,
            }
