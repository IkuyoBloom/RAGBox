"""检索层的公共数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedChunk:
    """融合后返回给上层的检索结果。

    同时保留稠密/稀疏两路的名次与分数，这样：
      * 前端可以展示「哪一路命中的」，答辩时能直观看出混合检索的价值；
      * 评估模块可以单独跑 dense-only / sparse-only / hybrid 三种配置做消融实验。
    """

    chunk_id: str
    doc_id: str
    kb_id: str
    ordinal: int
    content: str
    filename: str
    page: int | None = None

    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float = 0.0

    # 命中来源：dense / sparse / both
    @property
    def source(self) -> str:
        if self.dense_rank is not None and self.sparse_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "dense"
        if self.sparse_rank is not None:
            return "sparse"
        return "none"

    def to_citation(self, index: int) -> dict[str, Any]:
        """转成前端引用卡片 / 落库用的结构。index 从 1 开始，对应正文里的 [n]。"""
        return {
            "index": index,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page": self.page,
            "ordinal": self.ordinal,
            "score": round(self.fused_score, 6),
            "dense_score": None if self.dense_score is None else round(self.dense_score, 6),
            "sparse_score": None if self.sparse_score is None else round(self.sparse_score, 6),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "source": self.source,
            # 预览截断，完整正文另走 /api/chunks/{id}
            "preview": self.content[:400],
        }


@dataclass(slots=True)
class RetrievalTrace:
    """一次检索的完整过程记录 —— 论文里做误差分析靠它。"""

    query: str
    rewritten_query: str | None = None
    dense_hits: int = 0
    sparse_hits: int = 0
    fused_hits: int = 0
    elapsed_ms: int = 0
    mode: str = "hybrid"
    stages: list[dict[str, Any]] = field(default_factory=list)

    def mark(self, stage: str, **info: Any) -> None:
        self.stages.append({"stage": stage, **info})


__all__ = ["RetrievalTrace", "RetrievedChunk"]
