"""Ollama 向量化后端。

使用 /api/embed（批量接口）。相比老的 /api/embeddings（一次一条），
批量接口在建立索引时能省掉大量 HTTP 往返 —— 这是实测中最明显的性能差异。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from app.core.embedding.base import Embedder, EmbeddingError


class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        dim: int | None = None,
        batch_size: int = 16,
        timeout: float = 120.0,
        **options: Any,
    ) -> None:
        super().__init__(
            model, dim=dim, batch_size=batch_size, timeout=timeout, **options
        )
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    def _explain(self, status: int, body: str) -> str:
        body_short = body.strip()[:300]
        if status == 404:
            return (
                f"Ollama 中找不到向量模型 `{self.model}`。"
                f"请先执行：ollama pull {self.model}"
            )
        return f"Ollama 向量化失败（HTTP {status}）：{body_short}"

    async def _embed_raw(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": list(texts)}
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=8.0)
            ) as client:
                resp = await client.post(f"{self.base_url}/api/embed", json=payload)
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"无法连接 Ollama（{self.base_url}）进行向量化："
                f"{type(exc).__name__}: {exc}。请确认 Ollama 服务正在运行。",
                provider=self.name,
            ) from exc

        if resp.status_code >= 400:
            raise EmbeddingError(
                self._explain(resp.status_code, resp.text),
                provider=self.name,
                status=resp.status_code,
            )

        data = resp.json()
        if err := data.get("error"):
            raise EmbeddingError(f"Ollama 向量化报错：{err}", provider=self.name)

        embeddings = data.get("embeddings")
        if not embeddings:
            raise EmbeddingError(
                "Ollama 未返回 embeddings，可能是模型不支持向量化。"
                f"当前模型：{self.model}",
                provider=self.name,
            )
        return [list(map(float, vec)) for vec in embeddings]

    # ------------------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, connect=5.0)
            ) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
            if resp.status_code >= 400:
                return {
                    "ok": False,
                    "detail": f"Ollama 返回 HTTP {resp.status_code}",
                    "models": [],
                }
            models = [m.get("name", "") for m in (resp.json().get("models") or [])]
            has = any(
                m == self.model or m.split(":")[0] == self.model.split(":")[0]
                for m in models
            )
            return {
                "ok": True,
                "detail": f"Ollama 正常，向量模型 {self.model}"
                + ("" if has else "（未安装，请 ollama pull）"),
                "models": models,
                "has_configured_model": has,
                "base_url": self.base_url,
            }
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "detail": f"无法连接 Ollama（{self.base_url}）：{type(exc).__name__}: {exc}",
                "models": [],
                "base_url": self.base_url,
            }
