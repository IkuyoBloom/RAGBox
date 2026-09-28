"""按配置构造 Embedding 后端。

带进程内缓存：Ollama 客户端本身无状态，重复构造没有意义；
更重要的是缓存 `dim`，否则每次查询都要多花一次探测请求。
"""

from __future__ import annotations

from typing import Any

from app.core.embedding.base import Embedder, EmbeddingError
from app.core.embedding.ollama_embedder import OllamaEmbedder

_PROVIDERS = {"ollama": OllamaEmbedder}

_cache: dict[tuple, Embedder] = {}


def build_embedder(
    cfg: dict[str, Any],
    *,
    model: str | None = None,
    provider: str | None = None,
) -> Embedder:
    """model 参数用于「按知识库自己的模型」编码，优先于全局配置。"""
    name = (provider or cfg.get("embedding_provider") or "ollama").lower()
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise EmbeddingError(
            f"未知的 Embedding 后端 `{name}`，可选：{'、'.join(_PROVIDERS)}", provider=name
        )

    resolved_model = model or cfg.get("ollama_embed_model") or "bge-m3"
    batch_size = int(cfg.get("embed_batch_size") or 16)
    base_url = cfg.get("ollama_base_url") or "http://127.0.0.1:11434"
    dim = cfg.get("embed_dim")

    key = (name, resolved_model, base_url, batch_size, dim)
    cached = _cache.get(key)
    if cached is None:
        cached = cls(
            resolved_model,
            base_url=base_url,
            dim=dim,
            batch_size=batch_size,
        )
        _cache[key] = cached
    return cached


def clear_embedder_cache() -> None:
    _cache.clear()


__all__ = ["build_embedder", "clear_embedder_cache", "OllamaEmbedder"]
