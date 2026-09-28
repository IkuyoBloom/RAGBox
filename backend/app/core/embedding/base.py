"""Embedding 抽象层。

为什么单独抽象：知识库一旦建好，向量空间就被模型固定了 ——
换模型等于换语义坐标系，旧向量全部作废。所以：
  * 每个知识库在创建时**冻结**自己的 embedding 模型名与维度；
  * 查询时按知识库的模型去编码，而不是用全局当前设置；
  * 模型不匹配时报明确错误，绝不静默返回垃圾结果。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class EmbeddingError(RuntimeError):
    """向量化失败，message 会展示给前端。"""

    def __init__(self, message: str, *, provider: str = "", status: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status


@dataclass(frozen=True)
class EmbeddingModelInfo:
    provider: str
    model: str
    dim: int


class Embedder(ABC):
    """向量化后端统一接口。"""

    name: str = "base"

    def __init__(
        self,
        model: str,
        *,
        dim: int | None = None,
        batch_size: int = 16,
        timeout: float = 120.0,
        # 非对称检索用的前缀。bge-m3 不需要，但 bge-large-zh 之类需要给查询加指令
        query_prefix: str = "",
        doc_prefix: str = "",
        **options: Any,
    ) -> None:
        self.model = model
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.options = options
        self._dim = dim

    # ------------------------------------------------------------------
    @property
    def dim(self) -> int | None:
        return self._dim

    def set_dim(self, dim: int) -> None:
        self._dim = dim

    @abstractmethod
    async def _embed_raw(self, texts: Sequence[str]) -> list[list[float]]:
        """后端原生批量编码。"""

    # ------------------------------------------------------------------
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [self.doc_prefix + t for t in texts] if self.doc_prefix else list(texts)
        out: list[list[float]] = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            vectors = await self._embed_raw(batch)
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"{self.name} 返回的向量数量（{len(vectors)}）与输入数量（{len(batch)}）不一致",
                    provider=self.name,
                )
            out.extend(vectors)
        if out and self._dim is None:
            self._dim = len(out[0])
        return out

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed_raw([self.query_prefix + text])
        if not vectors:
            raise EmbeddingError(f"{self.name} 未返回查询向量", provider=self.name)
        if self._dim is None:
            self._dim = len(vectors[0])
        return vectors[0]

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """探测可用性，返回 {ok, detail, models}。"""


__all__ = ["Embedder", "EmbeddingError", "EmbeddingModelInfo"]
