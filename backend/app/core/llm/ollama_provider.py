"""本地 Ollama 通道。"""

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


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 300.0,
        num_ctx: int | None = None,
        **options: Any,
    ) -> None:
        super().__init__(model, timeout=timeout, **options)
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _timeout(self) -> httpx.Timeout:
        # 连接阶段很快失败才好——服务没起时不该让用户干等 300 秒
        return httpx.Timeout(self.timeout, connect=8.0)

    def _build_payload(
        self,
        messages: Sequence[ChatMessage],
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        if temperature is not None:
            opts["temperature"] = temperature
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        if self.num_ctx:
            opts["num_ctx"] = self.num_ctx

        return {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
            "options": opts,
            # 关掉思维链字段外泄（qwen3 等混合推理模型会返回 thinking）
            "think": False,
        }

    def _explain_http_error(self, status: int, body: str) -> str:
        body_short = body.strip()[:300]
        if status == 404:
            return (
                f"Ollama 中找不到模型 `{self.model}`。"
                f"请先执行：ollama pull {self.model}"
            )
        if status == 400 and "context" in body_short.lower():
            return (
                f"上下文长度超出模型限制，请调小 num_ctx 或减小 final_top_k。原始信息：{body_short}"
            )
        return f"Ollama 返回 HTTP {status}：{body_short}"

    # ------------------------------------------------------------------
    # 非流式
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload = self._build_payload(messages, False, temperature, max_tokens)
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise self._friendly_connect_error(exc, "Ollama", self.base_url) from exc

        if resp.status_code >= 400:
            raise LLMError(
                self._explain_http_error(resp.status_code, resp.text),
                provider=self.name,
                status=resp.status_code,
            )

        data = resp.json()
        if err := data.get("error"):
            raise LLMError(f"Ollama 报错：{err}", provider=self.name)

        return ChatResult(
            content=(data.get("message") or {}).get("content", "") or "",
            provider=self.name,
            model=data.get("model", self.model),
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    # ------------------------------------------------------------------
    # 流式（NDJSON）
    # ------------------------------------------------------------------
    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        payload = self._build_payload(messages, True, temperature, max_tokens)
        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
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
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if err := data.get("error"):
                            raise LLMError(f"Ollama 报错：{err}", provider=self.name)

                        msg = data.get("message") or {}
                        text = msg.get("content") or ""
                        # 有些模型会把思维链塞在 thinking 字段，这里明确丢弃
                        done = bool(data.get("done"))
                        if text or done:
                            yield StreamChunk(
                                text=text,
                                finished=done,
                                prompt_tokens=data.get("prompt_eval_count"),
                                completion_tokens=data.get("eval_count"),
                            )
        except httpx.HTTPError as exc:
            raise self._friendly_connect_error(exc, "Ollama", self.base_url) from exc

    # ------------------------------------------------------------------
    # 运维
    # ------------------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
            if resp.status_code >= 400:
                return {"ok": False, "detail": f"Ollama 返回 HTTP {resp.status_code}", "models": []}
            models = [m.get("name", "") for m in (resp.json().get("models") or [])]
            has_model = any(
                m == self.model or m.split(":")[0] == self.model.split(":")[0] for m in models
            )
            detail = "Ollama 正常"
            if not has_model:
                detail = f"Ollama 正常，但缺少模型 {self.model}（ollama pull {self.model}）"
            return {
                "ok": True,
                "detail": detail,
                "models": models,
                "has_configured_model": has_model,
                "base_url": self.base_url,
            }
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "detail": f"无法连接 Ollama（{self.base_url}）：{type(exc).__name__}: {exc}",
                "models": [],
                "base_url": self.base_url,
            }
