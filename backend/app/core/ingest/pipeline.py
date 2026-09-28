"""文档入库流水线：解析 -> 切分 -> 向量化 -> 落库。

## 事务边界设计（重要）

朴素写法是把整个流程包在一个事务里，但向量化一批 500 片的文档要几秒到几十秒，
SQLite 会在这段时间里一直持有写锁，界面上别的操作全部阻塞。

所以这里刻意把流程切成 **「短事务读写 - 无事务长计算 - 短事务写入」**：

    ① 短读：取出文档与知识库配置          -> 立即结束事务
    ② 无事务：解析 + 切分 + 调用 Ollama 向量化（耗时都在这里）
    ③ 短写：删除旧切片 -> 插入新切片 -> 更新文档状态与计数 -> 提交
    ④ 事务外：写入向量库（失败则标记文档 failed，可一键重建索引）

向量库写入放在 ③ 之后，是为了保证「数据库里存在的切片」一定先落盘；
即使 Chroma 写失败，也只会出现「有文本没向量」的可恢复状态（重建索引即可），
不会出现「有向量没文本」这种检索时回表拿不到内容的脏状态。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select

from app.core.embedding import build_embedder
from app.core.ingest.loaders import ParseError, detect_type, load_document
from app.core.ingest.splitter import split_blocks
from app.core.retrieval.bm25 import registry as bm25_registry
from app.core.vectorstore import get_vector_store
from app.db.base import session_scope
from app.db.models import Chunk, Document, DocumentStatus, KnowledgeBase, new_id

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 入库并发上限
#
# 为什么必须限制：一次上传 100 个文档会产生 100 个后台任务同时开跑，
# 它们一起抢 Ollama、抢 Chroma 的嵌入式 SQLite、抢事件循环。
# 实测 111 个文档同时入库时整个服务卡死 —— 连 /api/health 都要 60 秒才响应，
# 而 GPU 利用率只有 6%，说明瓶颈根本不在模型推理，而在无节制的并发。
#
# 取 2 是有意的：本机只有一个 Ollama 实例，再高的并发也只是互相排队，
# 反而让状态更难推进、日志更难读。
MAX_CONCURRENT_INGEST = 2

_ingest_gate: asyncio.Semaphore | None = None
_ingest_gate_loop: asyncio.AbstractEventLoop | None = None


def _ingest_semaphore() -> asyncio.Semaphore:
    """取得并发闸门（**按事件循环隔离**）。

    `asyncio.Semaphore` 一旦被 await 就绑定到当时的事件循环，
    换个循环再 await 会直接抛 `bound to a different event loop`。
    所以不能简单地做成模块级单例：测试里每个用例一个新 loop，
    在同一个进程里跑两批异步入库就会炸。

    这里按 loop 缓存，发现循环变了就重建一个。
    重建是安全的：旧循环上的等待者随旧循环一起消失了，
    新循环上的任务用新闸门，限流语义不变。
    """
    global _ingest_gate, _ingest_gate_loop
    loop = asyncio.get_running_loop()
    if _ingest_gate is None or _ingest_gate_loop is not loop:
        _ingest_gate = asyncio.Semaphore(MAX_CONCURRENT_INGEST)
        _ingest_gate_loop = loop
    return _ingest_gate


@dataclass(slots=True)
class IngestStats:
    doc_id: str
    chunk_count: int = 0
    char_count: int = 0
    page_count: int = 0
    parse_ms: int = 0
    split_ms: int = 0
    embed_ms: int = 0
    vector_ms: int = 0
    total_ms: int = 0
    embed_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_file_hash(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    """流式计算 sha256，避免把大文件整个读进内存。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _set_status(doc_id: str, status: str, **fields: Any) -> None:
    """独立的短事务：只更新状态字段，让前端能轮询到中间态。"""
    with session_scope() as db:
        doc = db.get(Document, doc_id)
        if doc is None:
            return
        doc.status = status
        for key, value in fields.items():
            setattr(doc, key, value)


def _purge_vectors(kb: KnowledgeBase, doc_id: str) -> None:
    """删除某文档在向量库里的全部向量。"""
    try:
        get_vector_store().delete(kb.collection_name, where={"doc_id": doc_id})
    except Exception as exc:  # 清理失败不致命
        logger.warning("清理文档向量失败 doc=%s: %s", doc_id, exc)


async def ingest_document(doc_id: str, cfg: dict[str, Any]) -> IngestStats:
    """执行一次完整入库（对外入口）。

    并发受 `MAX_CONCURRENT_INGEST` 限制。等待闸门期间文档保持 PENDING 状态，
    界面上显示「排队中」—— 这正是想传达的语义：文档已经收下了、在排队，
    而不是正在跑。不加这道闸门的话，批量上传会让整个服务失去响应（见常量注释）。
    """
    async with _ingest_semaphore():
        return await _run_ingest(doc_id, cfg)


async def _run_ingest(doc_id: str, cfg: dict[str, Any]) -> IngestStats:
    """真正的入库流程。异常会被转换成文档的 failed 状态，不向外抛。"""
    started = time.perf_counter()
    stats = IngestStats(doc_id=doc_id)

    # ---------------- ① 短读：拿任务快照 ----------------
    with session_scope() as db:
        doc = db.get(Document, doc_id)
        if doc is None:
            raise ValueError(f"文档不存在：{doc_id}")
        kb = db.get(KnowledgeBase, doc.kb_id)
        if kb is None:
            raise ValueError(f"知识库不存在：{doc.kb_id}")
        snapshot = {
            "path": Path(doc.stored_path),
            "filename": doc.filename,
            "file_type": doc.file_type,
            "kb_id": kb.id,
            "collection": kb.collection_name,
            "embed_model": kb.embed_model,
            "chunk_size": kb.chunk_size,
            "chunk_overlap": kb.chunk_overlap,
        }
        doc.status = DocumentStatus.PARSING
        doc.error = ""

    stats.embed_model = snapshot["embed_model"]
    path: Path = snapshot["path"]

    try:
        # ---------------- ② 无事务阶段 ----------------
        if not path.exists():
            raise ParseError(f"原始文件已丢失：{path}")

        t0 = time.perf_counter()
        # 解析是纯 CPU 的同步操作（pypdf / python-docx）。直接在协程里跑会把
        # 事件循环占住 —— 这正是「上传过多就卡死」的成因之一。
        parsed = await asyncio.to_thread(
            load_document, path, snapshot["file_type"] or detect_type(snapshot["filename"])
        )
        stats.parse_ms = int((time.perf_counter() - t0) * 1000)
        stats.page_count = parsed.page_count

        t0 = time.perf_counter()
        # 切分同样是同步 CPU 操作，交给线程池
        pieces = await asyncio.to_thread(
            split_blocks,
            parsed.blocks,
            chunk_size=snapshot["chunk_size"],
            chunk_overlap=snapshot["chunk_overlap"],
        )
        stats.split_ms = int((time.perf_counter() - t0) * 1000)

        if not pieces:
            raise ParseError("切分后没有得到任何文本片段，请检查文档内容。")

        # 本地先生成 id，这样向量化完成前就能确定每个片的身份
        rows: list[dict[str, Any]] = []
        for ordinal, piece in enumerate(pieces):
            rows.append(
                {
                    "id": new_id(),
                    "ordinal": ordinal,
                    "content": piece.text,
                    "char_count": len(piece.text),
                    "page": piece.page,
                }
            )
        stats.chunk_count = len(rows)
        stats.char_count = sum(r["char_count"] for r in rows)

        # 状态推进到「向量化中」，长耗时在这里
        _set_status(doc_id, DocumentStatus.EMBEDDING, chunk_count=0)

        embedder = build_embedder(cfg, model=snapshot["embed_model"])
        t0 = time.perf_counter()
        vectors = await embedder.embed_documents([r["content"] for r in rows])
        stats.embed_ms = int((time.perf_counter() - t0) * 1000)

        if len(vectors) != len(rows):
            raise ParseError(
                f"向量数量（{len(vectors)}）与切片数量（{len(rows)}）不一致，已中止入库"
            )

        # ---------------- ③ 短写：落盘 ----------------
        with session_scope() as db:
            # 重建索引场景：先清掉旧切片
            db.execute(delete(Chunk).where(Chunk.doc_id == doc_id))

            db.add_all(
                [
                    Chunk(
                        id=r["id"],
                        kb_id=snapshot["kb_id"],
                        doc_id=doc_id,
                        ordinal=r["ordinal"],
                        content=r["content"],
                        char_count=r["char_count"],
                        page=r["page"],
                    )
                    for r in rows
                ]
            )

            doc = db.get(Document, doc_id)
            if doc is None:
                raise ValueError(f"文档在入库过程中被删除：{doc_id}")
            doc.status = DocumentStatus.READY
            doc.chunk_count = len(rows)
            doc.char_count = stats.char_count
            doc.page_count = stats.page_count
            doc.error = ""
            # 这里先写一份「阶段性」统计；向量库写完后再回写完整版（见下方）
            meta = dict(doc.meta or {})
            meta["ingest_partial"] = stats.to_dict()
            meta["parse_meta"] = parsed.meta
            doc.meta = meta

            # 知识库层面的文档数/切片数一律在读接口实时聚合，不冗余存储，
            # 从根上避免计数器与真实数据不一致。
            kb = db.get(KnowledgeBase, snapshot["kb_id"])
            if kb is not None and kb.embed_dim is None and embedder.dim:
                kb.embed_dim = embedder.dim

        # ---------------- ④ 事务外：写向量库 ----------------
        t0 = time.perf_counter()
        store = get_vector_store()
        # Chroma 的接口都是同步的，直接在协程里调用会阻塞事件循环。
        # 交给线程池执行；向量库内部还有一把写锁保证并发安全。
        # 先按 doc_id 清掉可能残留的旧向量（重建索引时必需）
        await asyncio.to_thread(store.delete, snapshot["collection"], where={"doc_id": doc_id})
        await asyncio.to_thread(
            store.add,
            snapshot["collection"],
            ids=[r["id"] for r in rows],
            embeddings=vectors,
            metadatas=[
                {
                    "kb_id": snapshot["kb_id"],
                    "doc_id": doc_id,
                    "ordinal": r["ordinal"],
                    "page": r["page"] if r["page"] is not None else -1,
                    "filename": snapshot["filename"],
                }
                for r in rows
            ],
        )
        stats.vector_ms = int((time.perf_counter() - t0) * 1000)

        # 语料变了，BM25 索引必须失效
        bm25_registry.invalidate(snapshot["kb_id"])

        stats.total_ms = int((time.perf_counter() - started) * 1000)

        # 此时统计才齐全，回写一次 meta。
        # 上面那次写入发生在 vector_ms/total_ms 计算之前，序列化出来的这两个字段是 0 ——
        # 单行 UPDATE 的代价可以忽略，换来的是论文里能直接引用的真实耗时。
        with session_scope() as db:
            doc = db.get(Document, doc_id)
            if doc is not None:
                meta = dict(doc.meta or {})
                meta["ingest"] = stats.to_dict()
                doc.meta = meta

        logger.info(
            "入库完成 doc=%s chunks=%d parse=%dms split=%dms embed=%dms vector=%dms total=%dms",
            doc_id,
            stats.chunk_count,
            stats.parse_ms,
            stats.split_ms,
            stats.embed_ms,
            stats.vector_ms,
            stats.total_ms,
        )
        return stats

    except Exception as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        message = str(exc) if isinstance(exc, (ParseError, ValueError)) else f"{type(exc).__name__}: {exc}"
        logger.exception("入库失败 doc=%s", doc_id)
        # 失败时清掉半成品，避免留下「有切片但状态 failed」的困惑状态
        try:
            with session_scope() as db:
                db.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
                doc = db.get(Document, doc_id)
                if doc is not None:
                    doc.status = DocumentStatus.FAILED
                    doc.error = message[:2000]
                    doc.chunk_count = 0
        except Exception:
            logger.exception("回滚入库状态也失败了 doc=%s", doc_id)
        try:
            with session_scope() as db:
                kb = db.get(KnowledgeBase, snapshot["kb_id"])
                if kb is not None:
                    _purge_vectors(kb, doc_id)
        except Exception:
            pass
        stats.total_ms = elapsed
        raise ParseError(message) from exc


async def reindex_document(doc_id: str, cfg: dict[str, Any]) -> IngestStats:
    """重建索引：直接重跑入库流程（会先删旧切片与旧向量）。"""
    return await ingest_document(doc_id, cfg)


# ----------------------------------------------------------------------
# 启动恢复
# ----------------------------------------------------------------------
# 后台任务的强引用集合。asyncio 只持弱引用，不留一份的话
# 任务可能在跑完前被垃圾回收 —— 表现为「文档永远停在向量化中」。
_background_tasks: set[asyncio.Task[Any]] = set()


def _spawn(coro: Any) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _safe_ingest(doc_id: str, cfg: dict[str, Any]) -> None:
    try:
        await ingest_document(doc_id, cfg)
    except Exception as exc:  # ingest_document 内部已记状态，这里只兜底日志
        logger.warning("恢复入库失败 doc=%s: %s", doc_id, exc)


def spawn_ingest(doc_id: str, cfg: dict[str, Any]) -> None:
    """把入库排进后台，**不绑定调用方的请求生命周期**。

    ## 为什么不能用 FastAPI 的 BackgroundTasks

    这是「批量上传把连接池打满」的真正根因。FastAPI 里 `yield` 型依赖
    （`get_db`）的清理发生在**后台任务跑完之后**，所以每个上传请求会把它的
    数据库连接占住整整一批文件的入库时间 —— 一批 8 个文件顺序入库约
    8~10 秒，连接就被占用这么久。

    批量导入时几十个上传请求一叠加，再算上界面的轮询请求，
    连接池瞬间被占光。**把池子从 15 扩到 60 照样被打满**，
    报 `QueuePool limit ... reached`，而且失败会级联：
    入库拿不到连接 -> 失败 -> 重试逻辑又要连接 -> 继续失败。

    改成独立的 asyncio 任务后：请求一返回，依赖清理立即执行、连接立即释放；
    入库用自己短生命周期的 `session_scope()` 会话；并发仍由 `_ingest_semaphore()`
    控制。三件事各归各位，连接占用时间从「一批文件的入库时长」降到「一次短查询」。
    """
    _spawn(_safe_ingest(doc_id, cfg))


async def requeue_unfinished(*, max_docs: int = 5000) -> int:
    """恢复上次没跑完的文档，并重新排队入库。

    两种场景会在库里留下「中间态」：

      * **进程在入库途中退出** —— 后台任务随进程一起消失，但数据库里的状态
        停在 PARSING / EMBEDDING。界面上表现为一份文档永远「向量化中」，
        用户既不能重试（重建索引按钮对它没用）也搞不懂发生了什么。
      * **上传后服务立刻重启** —— PENDING 的文档没有接手方，同样永远停着。

    实测：一次上传 111 份文档把服务压垮并重启后，库里积了 92 份这种文档。
    启动时做一次恢复，让它们自动跑完，而不是把烂摊子留给用户。
    """
    from app.core.settings_service import SettingsService

    with session_scope() as db:
        cfg = SettingsService(db).effective()
        rows = db.execute(
            select(Document.id)
            .where(
                Document.status.in_(
                    [DocumentStatus.PARSING, DocumentStatus.EMBEDDING, DocumentStatus.PENDING]
                )
            )
            .order_by(Document.created_at)
            .limit(max_docs)
        ).all()
        doc_ids = [r[0] for r in rows]

        for doc_id in doc_ids:
            doc = db.get(Document, doc_id)
            if doc is not None:
                # 统一回到 PENDING，让它们从干净的状态重新开始
                doc.status = DocumentStatus.PENDING
                doc.error = ""

    if not doc_ids:
        return 0

    # 全部丢进后台：入库闸门会自动把并发压到 MAX_CONCURRENT_INGEST，
    # 所以这里一次性提交几百个也不会重演「并发压垮服务」。
    for doc_id in doc_ids:
        _spawn(_safe_ingest(doc_id, cfg))

    logger.info("恢复了 %d 份上次未完成的文档，已重新排队", len(doc_ids))
    return len(doc_ids)


__all__ = [
    "IngestStats",
    "compute_file_hash",
    "ingest_document",
    "reindex_document",
]
