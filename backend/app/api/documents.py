"""文档上传、索引与切片查询接口。"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Response, UploadFile
from sqlalchemy import select

from app.api.deps import DbSession, SettingsDep, get_kb_or_404, problem
from app.core.ingest import (
    SUPPORTED_TYPES,
    detect_type,
    spawn_ingest,
)
from app.core.retrieval.bm25 import registry as bm25_registry
from app.core.vectorstore import get_vector_store
from app.db.models import Chunk, Document, DocumentStatus, new_id
from app.schemas import ChunkOut, DocumentOut, DocumentUploadResult

logger = logging.getLogger(__name__)
router = APIRouter(tags=["文档"])

MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MB
# 展示名上限。数据库列是 String(512)，留出余量。
MAX_NAME_CHARS = 480


def _safe_display_name(raw: str) -> str:
    """把客户端传来的文件名规范成安全的展示名。

    「选择文件夹」导入时浏览器会带上相对路径（如 `论文/2024/注意力机制.pdf`）。
    这个路径是有价值的信息 —— 它能区分不同子目录下的同名文件，
    而一个文件夹树里出现多个 `README.md` 实在太常见了。所以保留它，
    但必须做安全处理：

      * 统一分隔符（Windows 客户端可能传反斜杠）；
      * 去掉盘符与开头的斜杠，避免被当成绝对路径；
      * `..` 按真实路径语义**弹出上一级**，已在根上则丢弃 ——
        单纯丢弃会得到误导性的名字：`docs/../../secret.txt` 若变成
        `docs/secret.txt`，看名字会以为文件在 docs 里，实际它来自别处。
      * 限制总长度，超长时保留尾部（文件名与最贴近的目录比顶层目录更有用）。

    注意：真正落盘用的是 `<doc_id><后缀>`，与展示名无关，
    所以保留路径**不会**在上传目录里造出嵌套结构。
    """
    name = (raw or "").replace("\\", "/").strip()
    name = re.sub(r"^[A-Za-z]:", "", name)  # C:/xxx -> /xxx

    resolved: list[str] = []
    for part in name.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            # 能退就退一级；已经在根上则直接丢弃，绝不生成上级引用
            if resolved:
                resolved.pop()
            continue
        resolved.append(part)

    if not resolved:
        return "unnamed"

    joined = "/".join(resolved)
    if len(joined) > MAX_NAME_CHARS:
        joined = "…" + joined[-(MAX_NAME_CHARS - 1) :]
    return joined


async def _save_upload(upload: UploadFile, dest: Path) -> tuple[int, str]:
    """流式落盘并同时算 sha256，避免把大文件整个读进内存。"""
    digest = hashlib.sha256()
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as fh:
            while True:
                block = await upload.read(1 << 20)
                if not block:
                    break
                size += len(block)
                if size > MAX_FILE_BYTES:
                    raise ValueError(
                        f"文件超过 {MAX_FILE_BYTES // 1024 // 1024} MB 上限"
                    )
                digest.update(block)
                fh.write(block)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return size, digest.hexdigest()


def _run_ingest(doc_id: str, cfg: dict) -> None:
    """把入库交给后台调度。

    **必须是同步函数**。这里只调用 spawn_ingest（它内部用 asyncio.create_task
    起独立任务）。曾经把它写成 async def 却在端点里不 await 地调用 ——
    结果是协程被创建但从未执行，新上传的文档永远停在 pending，
    只在日志里留一行 RuntimeWarning，极难发现。

    也**不**再注册成 FastAPI 的 BackgroundTask：那样会让上传请求把数据库
    连接占到自己这批文件全部入库结束（详见 spawn_ingest 的说明）。
    """
    spawn_ingest(doc_id, cfg)


@router.post(
    "/kb/{kb_id}/documents",
    response_model=list[DocumentUploadResult],
    summary="上传文档（支持多文件，后台异步建索引）",
)
async def upload_documents(
    kb_id: str,
    db: DbSession,
    settings: SettingsDep,
    files: list[UploadFile] = File(..., description="支持 PDF / DOCX / MD / TXT"),
) -> list[DocumentUploadResult]:
    get_kb_or_404(db, kb_id)
    if not files:
        raise problem(422, "没有收到任何文件")

    cfg = settings.effective()
    upload_root = settings.base.upload_dir / kb_id
    results: list[DocumentUploadResult] = []

    for upload in files:
        # 保留相对路径（文件夹导入时用来区分同名文件），同时做路径穿越防护
        original = _safe_display_name(upload.filename or "unnamed")
        file_type = detect_type(original)

        if file_type is None:
            results.append(
                DocumentUploadResult(
                    filename=original,
                    ok=False,
                    error=(
                        f"不支持的文件类型：{Path(original).suffix or '(无扩展名)'}。"
                        f"当前支持：{'、'.join(sorted(SUPPORTED_TYPES))}"
                    ),
                )
            )
            continue

        doc_id = new_id()
        dest = upload_root / f"{doc_id}{Path(original).suffix.lower()}"

        try:
            size, content_hash = await _save_upload(upload, dest)
        except ValueError as exc:
            results.append(DocumentUploadResult(filename=original, ok=False, error=str(exc)))
            continue
        except Exception as exc:
            logger.exception("保存上传文件失败：%s", original)
            results.append(
                DocumentUploadResult(
                    filename=original, ok=False, error=f"保存文件失败：{type(exc).__name__}: {exc}"
                )
            )
            continue

        if size == 0:
            dest.unlink(missing_ok=True)
            results.append(DocumentUploadResult(filename=original, ok=False, error="文件内容为空"))
            continue

        # 同库同内容去重（秒传）：既省索引时间，也避免同一份资料被检索到两次
        existing = db.execute(
            select(Document).where(
                Document.kb_id == kb_id,
                Document.content_hash == content_hash,
                Document.status == DocumentStatus.READY,
            )
        ).scalars().first()
        if existing is not None:
            dest.unlink(missing_ok=True)
            results.append(
                DocumentUploadResult(
                    filename=original,
                    ok=True,
                    document=DocumentOut.model_validate(existing),
                    duplicated=True,
                )
            )
            continue

        # 同名或同哈希但状态不是 ready 的旧记录（例如上次失败）先清掉，避免堆积
        stale = db.execute(
            select(Document).where(
                Document.kb_id == kb_id,
                Document.content_hash == content_hash,
            )
        ).scalars().all()
        for old in stale:
            Path(old.stored_path).unlink(missing_ok=True)
            db.delete(old)
        db.flush()

        doc = Document(
            id=doc_id,
            kb_id=kb_id,
            filename=original,
            stored_path=str(dest),
            file_type=file_type,
            size_bytes=size,
            content_hash=content_hash,
            status=DocumentStatus.PENDING,
            meta={"uploaded_filename": original},
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        results.append(DocumentUploadResult(filename=original, ok=True, document=DocumentOut.model_validate(doc)))
        # 索引进后台：向量化几百页可能要几十秒，不该让上传请求一直挂着。
        # 用 spawn_ingest 而不是 BackgroundTasks —— 后者会让本请求的连接
        # 一直占到自己这批文件全部入库完成（详见 spawn_ingest 的说明）。
        _run_ingest(doc_id, cfg)

    return results


@router.get("/documents/{doc_id}", response_model=DocumentOut, summary="文档详情（含索引进度）")
def get_document(doc_id: str, db: DbSession) -> DocumentOut:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise problem(404, f"文档不存在：{doc_id}")
    return DocumentOut.model_validate(doc)


@router.post("/documents/{doc_id}/reindex", response_model=DocumentOut, summary="重建索引")
async def reindex(doc_id: str, db: DbSession, settings: SettingsDep) -> DocumentOut:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise problem(404, f"文档不存在：{doc_id}")
    if not Path(doc.stored_path).exists():
        raise problem(409, "原始文件已丢失，无法重建索引，请重新上传。")

    doc.status = DocumentStatus.PENDING
    doc.error = ""
    db.commit()
    db.refresh(doc)

    _run_ingest(doc_id, settings.effective())
    return DocumentOut.model_validate(doc)


@router.delete("/documents/{doc_id}", status_code=204, summary="删除文档（含向量与文件）")
def delete_document(doc_id: str, db: DbSession) -> Response:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise problem(404, f"文档不存在：{doc_id}")

    kb_id = doc.kb_id
    stored_path = Path(doc.stored_path)

    db.delete(doc)
    db.commit()

    # 向量库按 doc_id 批量清理
    from app.db.models import KnowledgeBase

    kb = db.get(KnowledgeBase, kb_id)
    if kb is not None:
        try:
            get_vector_store().delete(kb.collection_name, where={"doc_id": doc_id})
        except Exception:
            logger.warning("清理文档向量失败 doc=%s", doc_id, exc_info=True)

    stored_path.unlink(missing_ok=True)
    bm25_registry.invalidate(kb_id)
    return Response(status_code=204)


@router.get("/documents/{doc_id}/chunks", response_model=list[ChunkOut], summary="查看文档切片")
def list_chunks(
    doc_id: str,
    db: DbSession,
    offset: int = 0,
    limit: int = 50,
) -> list[ChunkOut]:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise problem(404, f"文档不存在：{doc_id}")
    limit = max(1, min(limit, 500))

    rows = db.execute(
        select(Chunk)
        .where(Chunk.doc_id == doc_id)
        .order_by(Chunk.ordinal)
        .offset(max(0, offset))
        .limit(limit)
    ).scalars().all()

    out: list[ChunkOut] = []
    for chunk in rows:
        item = ChunkOut.model_validate(chunk)
        item.filename = doc.filename
        out.append(item)
    return out


@router.get("/chunks/{chunk_id}", response_model=ChunkOut, summary="读取单个切片正文（引用溯源用）")
def get_chunk(chunk_id: str, db: DbSession) -> ChunkOut:
    chunk = db.get(Chunk, chunk_id)
    if chunk is None:
        raise problem(404, f"切片不存在：{chunk_id}")
    doc = db.get(Document, chunk.doc_id)
    item = ChunkOut.model_validate(chunk)
    item.filename = doc.filename if doc else ""
    return item
