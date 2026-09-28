"""知识库管理接口。"""

from __future__ import annotations

import logging
import shutil

from fastapi import APIRouter, Response
from sqlalchemy import select

from app.api.deps import DbSession, SettingsDep, get_kb_or_404, kb_counts, problem
from app.core.retrieval.bm25 import registry as bm25_registry
from app.core.vectorstore import get_vector_store
from app.db.models import Document, KnowledgeBase, new_id
from app.schemas import (
    DocumentOut,
    KnowledgeBaseCreate,
    KnowledgeBaseDetail,
    KnowledgeBaseOut,
    KnowledgeBaseUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/kb", tags=["知识库"])


def _to_out(db, kb: KnowledgeBase, counts: dict[str, int] | None = None) -> KnowledgeBaseOut:
    data = KnowledgeBaseOut.model_validate(kb)
    if counts:
        data.document_count = counts.get("document_count", 0)
        data.ready_document_count = counts.get("ready_document_count", 0)
        data.chunk_count = counts.get("chunk_count", 0)
    return data


@router.get("", response_model=list[KnowledgeBaseOut], summary="知识库列表")
def list_kbs(db: DbSession) -> list[KnowledgeBaseOut]:
    kbs = db.execute(
        select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc())
    ).scalars().all()
    counts = kb_counts(db, [kb.id for kb in kbs])
    return [_to_out(db, kb, counts.get(kb.id)) for kb in kbs]


@router.post("", response_model=KnowledgeBaseOut, status_code=201, summary="新建知识库")
def create_kb(payload: KnowledgeBaseCreate, db: DbSession, settings: SettingsDep) -> KnowledgeBaseOut:
    name = payload.name.strip()
    if not name:
        raise problem(422, "知识库名称不能为空")

    exists = db.execute(
        select(KnowledgeBase.id).where(KnowledgeBase.name == name)
    ).first()
    if exists:
        raise problem(409, f"已存在同名知识库：{name}")

    cfg = settings.effective()

    # 向量模型在建库时冻结：之后换模型只影响新库，
    # 因为旧库的向量已经存在于另一个语义空间里。
    embed_model = (payload.embed_model or cfg.get("ollama_embed_model") or "bge-m3").strip()
    chunk_size = int(payload.chunk_size or cfg.get("chunk_size") or 500)
    chunk_overlap = (
        int(payload.chunk_overlap) if payload.chunk_overlap is not None else int(cfg.get("chunk_overlap") or 80)
    )
    if chunk_overlap >= chunk_size:
        raise problem(422, f"切片重叠（{chunk_overlap}）必须小于切片长度（{chunk_size}）")

    kb_id = new_id()
    kb = KnowledgeBase(
        id=kb_id,
        name=name,
        description=payload.description or "",
        embed_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection_name=f"kb_{kb_id}",
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)

    # 提前建好 collection，便于「没有文档时也能看到向量库结构」
    try:
        get_vector_store()._collection(kb.collection_name)  # noqa: SLF001
    except Exception:
        logger.warning("预创建 collection 失败（不影响使用）", exc_info=True)

    return _to_out(db, kb)


@router.get("/{kb_id}", response_model=KnowledgeBaseDetail, summary="知识库详情（含文档列表）")
def get_kb(kb_id: str, db: DbSession) -> KnowledgeBaseDetail:
    kb = get_kb_or_404(db, kb_id)
    base = _to_out(db, kb, kb_counts(db, [kb.id]).get(kb.id))
    docs = db.execute(
        select(Document).where(Document.kb_id == kb_id).order_by(Document.created_at.desc())
    ).scalars().all()
    detail = KnowledgeBaseDetail(**base.model_dump())
    detail.documents = [DocumentOut.model_validate(d) for d in docs]
    return detail


@router.patch("/{kb_id}", response_model=KnowledgeBaseOut, summary="重命名 / 改描述")
def update_kb(kb_id: str, payload: KnowledgeBaseUpdate, db: DbSession) -> KnowledgeBaseOut:
    kb = get_kb_or_404(db, kb_id)

    if payload.name is not None:
        name = payload.name.strip()
        if not name:
            raise problem(422, "知识库名称不能为空")
        clash = db.execute(
            select(KnowledgeBase.id).where(
                KnowledgeBase.name == name, KnowledgeBase.id != kb_id
            )
        ).first()
        if clash:
            raise problem(409, f"已存在同名知识库：{name}")
        kb.name = name

    if payload.description is not None:
        kb.description = payload.description

    db.commit()
    db.refresh(kb)
    return _to_out(db, kb, kb_counts(db, [kb.id]).get(kb.id))


@router.delete("/{kb_id}", status_code=204, summary="删除知识库（连同文档与向量）")
def delete_kb(kb_id: str, db: DbSession, settings: SettingsDep) -> Response:
    kb = get_kb_or_404(db, kb_id)
    collection = kb.collection_name
    upload_dir = settings.base.upload_dir / kb_id

    db.delete(kb)  # 级联删除 documents / chunks
    db.commit()

    # 数据库删干净后再清外部资源；失败只记日志，因为数据库才是权威
    try:
        get_vector_store().drop(collection)
    except Exception:
        logger.warning("删除 collection 失败：%s", collection, exc_info=True)
    try:
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
    except Exception:
        logger.warning("删除上传目录失败：%s", upload_dir, exc_info=True)

    bm25_registry.invalidate(kb_id)
    return Response(status_code=204)
