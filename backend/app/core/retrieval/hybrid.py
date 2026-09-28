"""混合检索：稠密向量 + BM25 稀疏，用 RRF 融合。

## 为什么是 RRF 而不是加权求和

稠密检索的分数是余弦相似度（0~1 有界），BM25 是无上界的对数加权和。
两者**量纲不同、分布不同**，直接 `α·dense + β·sparse` 需要反复调参，
而且换一个 embedding 模型就得重调。

RRF（Reciprocal Rank Fusion）只看**名次**不看分数：

    score(d) = Σ_r  weight_r / (k + rank_r(d))

  * 天然免疫量纲差异，换模型/换语料都不用重调；
  * k 越大越平滑（头部结果的优势被削弱），论文里常用 k=60；
  * 实现简单、无需训练，是工业界的默认基线。

## 三路可消融

`dense_only` / `sparse_only` / `hybrid` 三种模式都能跑，
评估模块直接复用本模块做消融实验，保证「论文里的对比数据」和
「系统实际行为」出自同一份代码，不会出现"实验代码和系统不一样"的尴尬。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Sequence
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.embedding import build_embedder
from app.core.retrieval.bm25 import BM25Index, BM25Hit, registry, tokenize
from app.core.retrieval.types import RetrievalTrace, RetrievedChunk
from app.core.vectorstore import get_vector_store
from app.db.models import Chunk, Document, KnowledgeBase

logger = logging.getLogger(__name__)

SearchMode = Literal["hybrid", "dense_only", "sparse_only"]


# ----------------------------------------------------------------------
# 纯函数：RRF 融合（单独拎出来是为了能直接单元测试）
# ----------------------------------------------------------------------
def rrf_fuse(
    dense: Sequence[tuple[str, float]],
    sparse: Sequence[BM25Hit],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> dict[str, float]:
    """倒数排名融合。

    dense  : [(chunk_id, similarity)]，调用方保证已按相关性降序
    sparse : [BM25Hit]，rank 字段从 1 开始
    返回   : {chunk_id: fused_score}
    """
    if k < 1:
        k = 1
    fused: dict[str, float] = {}

    for rank, (chunk_id, _score) in enumerate(dense, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + dense_weight / (k + rank)

    for hit in sparse:
        rank = hit.rank if hit.rank > 0 else 1
        fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + sparse_weight / (k + rank)

    return fused


# ----------------------------------------------------------------------
# 单路检索
# ----------------------------------------------------------------------
async def dense_search(
    db: Session,
    cfg: dict[str, Any],
    kb: KnowledgeBase,
    query: str,
    *,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """向量检索。

    注意用的是 `kb.embed_model` 而不是全局配置：向量空间由建库时的模型决定，
    用别的模型编码查询会得到毫无意义的相似度。
    """
    top_k = top_k or int(cfg.get("dense_top_k") or 20)
    embedder = build_embedder(cfg, model=kb.embed_model)
    vector = await embedder.embed_query(query)

    store = get_vector_store()
    # Chroma 的 query 是同步接口，直接在协程里调用会阻塞事件循环 ——
    # 检索高峰期会让整个服务无法响应其他请求。
    return await asyncio.to_thread(
        store.query,
        kb.collection_name,
        embedding=vector,
        top_k=top_k,
        where={"kb_id": kb.id},
    )


def _kb_signature(db: Session, kb_id: str) -> tuple[int, int]:
    row = db.execute(
        select(func.count(Chunk.id), func.count(func.distinct(Chunk.doc_id))).where(
            Chunk.kb_id == kb_id
        )
    ).one()
    return int(row[0] or 0), int(row[1] or 0)


def build_bm25_index(db: Session, kb_id: str) -> BM25Index:
    rows = db.execute(
        select(Chunk.id, Chunk.content)
        .where(Chunk.kb_id == kb_id)
        .order_by(Chunk.doc_id, Chunk.ordinal)
    ).all()
    ids = [r[0] for r in rows]
    corpus = [tokenize(r[1]) for r in rows]
    return BM25Index(ids, corpus)


# 建索引的按库锁。见 get_bm25_index 的说明 ——
# 没有它，并发检索会各自重建一整份索引并同时打满 CPU。
_build_locks: dict[str, threading.Lock] = {}
_build_locks_guard = threading.Lock()


def _build_lock(kb_id: str) -> threading.Lock:
    with _build_locks_guard:
        lock = _build_locks.get(kb_id)
        if lock is None:
            lock = threading.Lock()
            _build_locks[kb_id] = lock
        return lock


def get_bm25_index(db: Session, kb_id: str) -> BM25Index:
    """带缓存的索引获取。语料没变就直接复用，避免每次提问重建整库分词。

    并发保护是必须的：建索引要对整库切片做 jieba 分词（1 万条要十几秒）。
    检索现在跑在线程池里，若多个请求同时发现缓存失效，就会各建一份 ——
    内存与 CPU 瞬间翻几倍，效果等同于把「卡死」换个地方复现。
    这里用一把按知识库粒度的锁，保证同一时刻只有一个在建，其余等它建完直接拿缓存。
    """
    signature = _kb_signature(db, kb_id)
    cached = registry.get(kb_id, signature)
    if cached is not None:
        return cached

    with _build_lock(kb_id):
        # 双检：等锁期间可能已经有别人建好了
        cached = registry.get(kb_id, signature)
        if cached is not None:
            return cached
        index = build_bm25_index(db, kb_id)
        registry.put(kb_id, signature, index)
        return index


def sparse_search(
    db: Session,
    kb: KnowledgeBase,
    query: str,
    *,
    top_k: int | None = None,
) -> list[BM25Hit]:
    top_k = top_k or 20
    index = get_bm25_index(db, kb.id)
    return index.search(query, top_k)


# ----------------------------------------------------------------------
# 融合检索主入口
# ----------------------------------------------------------------------
async def retrieve(
    db: Session,
    cfg: dict[str, Any],
    kb: KnowledgeBase,
    query: str,
    *,
    mode: SearchMode | None = None,
    trace: RetrievalTrace | None = None,
) -> list[RetrievedChunk]:
    """检索主流程。

    mode 为 None 时按配置决定：enable_hybrid=True -> hybrid，否则 dense_only。
    """
    started = time.perf_counter()
    if mode is None:
        mode = "hybrid" if cfg.get("enable_hybrid", True) else "dense_only"

    dense_top_k = int(cfg.get("dense_top_k") or 20)
    sparse_top_k = int(cfg.get("sparse_top_k") or 20)
    final_top_k = int(cfg.get("final_top_k") or 6)
    rrf_k = int(cfg.get("rrf_k") or 60)

    tr = trace or RetrievalTrace(query=query, mode=mode)
    tr.mode = mode

    dense: list[tuple[str, float]] = []
    sparse: list[BM25Hit] = []

    # ---- 稠密路 ----
    if mode in ("hybrid", "dense_only"):
        t0 = time.perf_counter()
        dense = await dense_search(db, cfg, kb, query, top_k=dense_top_k)
        tr.mark("dense", hits=len(dense), ms=int((time.perf_counter() - t0) * 1000))
    tr.dense_hits = len(dense)

    # ---- 稀疏路 ----
    if mode in ("hybrid", "sparse_only"):
        t0 = time.perf_counter()
        # 首次检索要为整个知识库建 BM25 索引（jieba 逐片分词），
        # 语料大时能跑十几秒。放在协程里会直接堵死事件循环 ——
        # 表现就是「提问后界面整个卡住，连知识库列表都刷不出来」。
        sparse = await asyncio.to_thread(
            sparse_search, db, kb, query, top_k=sparse_top_k
        )
        tr.mark("sparse", hits=len(sparse), ms=int((time.perf_counter() - t0) * 1000))
    tr.sparse_hits = len(sparse)

    # ---- 融合 ----
    if mode == "hybrid":
        fused = rrf_fuse(dense, sparse, k=rrf_k)
    elif mode == "dense_only":
        fused = {cid: 1.0 / (rrf_k + rank) for rank, (cid, _) in enumerate(dense, start=1)}
    else:  # sparse_only
        fused = {
            h.chunk_id: 1.0 / (rrf_k + (h.rank if h.rank > 0 else 1)) for h in sparse
        }
    tr.fused_hits = len(fused)

    if not fused:
        tr.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return []

    # ---- 排序取 Top-N ----
    ranked_ids = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:final_top_k]
    ids = [cid for cid, _ in ranked_ids]

    # ---- 回表拿正文（SQLite 是唯一事实来源）----
    rows = db.execute(
        select(Chunk, Document.filename)
        .join(Document, Chunk.doc_id == Document.id)
        .where(Chunk.id.in_(ids))
    ).all()
    by_id: dict[str, tuple[Chunk, str]] = {chunk.id: (chunk, filename) for chunk, filename in rows}

    dense_rank_map = {cid: i for i, (cid, _) in enumerate(dense, start=1)}
    dense_score_map = dict(dense)
    sparse_rank_map = {h.chunk_id: h.rank for h in sparse}
    sparse_score_map = {h.chunk_id: h.score for h in sparse}

    results: list[RetrievedChunk] = []
    for cid, fused_score in ranked_ids:
        found = by_id.get(cid)
        if found is None:
            # 向量库里残留了数据库已删除的 id：忽略，并提示重建索引
            logger.warning("chunk %s 在向量库中存在但数据库中已删除，建议重建索引", cid)
            continue
        chunk, filename = found
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                kb_id=chunk.kb_id,
                ordinal=chunk.ordinal,
                content=chunk.content,
                filename=filename,
                page=chunk.page,
                dense_score=dense_score_map.get(cid),
                sparse_score=sparse_score_map.get(cid),
                dense_rank=dense_rank_map.get(cid),
                sparse_rank=sparse_rank_map.get(cid),
                fused_score=fused_score,
            )
        )

    tr.elapsed_ms = int((time.perf_counter() - started) * 1000)
    tr.mark("hydrate", rows=len(results), total_ms=tr.elapsed_ms)
    return results


__all__ = [
    "SearchMode",
    "build_bm25_index",
    "dense_search",
    "get_bm25_index",
    "retrieve",
    "rrf_fuse",
    "sparse_search",
]
