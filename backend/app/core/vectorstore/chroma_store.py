"""Chroma 向量库封装。

关键取舍：**Chroma 里不存正文**。
只放 `向量 + 定位元数据（kb_id / doc_id / ordinal / page）`，
正文由 SQLite 的 `chunks` 表提供。理由：
  1. 单一事实来源，不存在「向量库里的文本和数据库里的文本不一致」；
  2. 向量检索与 BM25 拿到的是同一批 chunk_id，融合时天然对齐，不需要按文本去重；
  3. 将来换向量库（Milvus / Qdrant / pgvector），只要重放 embedding 即可，数据不丢。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from app.core.embedding.base import EmbeddingError

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    pass


class ChromaVectorStore:
    """单例式封装的持久化向量库。"""

    def __init__(self, path: Path) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(
                # 本地离线项目，关掉匿名遥测，避免无网环境下的超时等待
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        # collection 句柄缓存。见 _collection() 的说明。
        self._collections: dict[str, Any] = {}
        self._collection_lock = threading.Lock()
        # Chroma 底层是嵌入式 SQLite，并发写会互相等锁甚至直接报错。
        # 用一把写锁把写入串行化：本地项目的写入量本就不大，
        # 而并发写导致的整体卡死是致命的（这正是之前那次「上传过多卡住」的成因之一）。
        self._write_lock = threading.Lock()

    # ------------------------------------------------------------------
    def _collection(self, name: str):
        """取得 collection 句柄（带缓存）。

        **必须缓存**：`get_or_create_collection` 每次都会读一遍 Chroma 内部的
        SQLite 元数据表。原先每次 add/query/delete 都调一次，批量入库时几十个
        任务反复打同一张元数据表，锁竞争把整个服务拖垮 ——
        实测 111 个文档同时入库时，连 /api/health 都要 60 秒才响应。
        """
        with self._collection_lock:
            cached = self._collections.get(name)
            if cached is not None:
                return cached

        # cosine 空间：文本检索的事实标准。默认是 L2，
        # 对未归一化的向量会得出反直觉的排序。
        # 建句柄本身放锁外：它可能较慢，不该把查询也一起堵住。
        collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        with self._collection_lock:
            # 并发时可能已经有别人放进去了，用先到的那个
            self._collections.setdefault(name, collection)
            return self._collections[name]

    # ------------------------------------------------------------------
    def add(
        self,
        collection_name: str,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        if not (len(ids) == len(embeddings) == len(metadatas)):
            raise VectorStoreError(
                f"add 参数长度不一致：ids={len(ids)} embeddings={len(embeddings)} "
                f"metadatas={len(metadatas)}"
            )
        try:
            with self._write_lock:
                self._collection(collection_name).add(
                    ids=ids, embeddings=embeddings, metadatas=metadatas
                )
        except Exception as exc:  # chroma 抛的异常类型不稳定
            raise VectorStoreError(f"写入向量库失败：{type(exc).__name__}: {exc}") from exc

    def delete(
        self,
        collection_name: str,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        if not ids and not where:
            return
        try:
            with self._write_lock:
                col = self._collection(collection_name)
                if ids:
                    col.delete(ids=ids)
                elif where:
                    col.delete(where=where)
        except Exception as exc:
            # 删除失败不该让整个事务崩掉：SQLite 删除才是权威，向量残留由重建索引兜底
            logger.warning("删除向量失败 collection=%s: %s", collection_name, exc)

    def drop(self, collection_name: str) -> None:
        try:
            with self._write_lock:
                self._client.delete_collection(collection_name)
        except Exception as exc:
            logger.warning("删除 collection 失败 %s: %s", collection_name, exc)
        finally:
            # 句柄已失效，必须从缓存里摘掉；否则重建同名 collection 后
            # 会继续用那个悬空句柄，写入静默丢失。
            with self._collection_lock:
                self._collections.pop(collection_name, None)

    def count(self, collection_name: str) -> int:
        try:
            return self._collection(collection_name).count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    def query(
        self,
        collection_name: str,
        *,
        embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """返回 [(chunk_id, similarity)]，similarity 越大越相关。"""
        if top_k <= 0:
            return []
        try:
            res = self._collection(collection_name).query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where or None,
                include=["distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"向量检索失败：{type(exc).__name__}: {exc}") from exc

        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        out: list[tuple[str, float]] = []
        for i, cid in enumerate(ids):
            dist = distances[i] if i < len(distances) else 0.0
            # cosine 空间下 distance = 1 - cos_sim
            out.append((cid, 1.0 - float(dist)))
        return out


_store: ChromaVectorStore | None = None


def get_vector_store(path: Path | None = None) -> ChromaVectorStore:
    global _store
    if _store is None:
        from app.config import get_settings

        _store = ChromaVectorStore(path or get_settings().chroma_dir)
    return _store


def reset_vector_store() -> None:
    """测试用：丢弃单例。"""
    global _store
    _store = None


__all__ = [
    "ChromaVectorStore",
    "VectorStoreError",
    "get_vector_store",
    "reset_vector_store",
    "EmbeddingError",
]
