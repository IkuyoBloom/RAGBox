"""FastAPI 依赖项。"""

from __future__ import annotations

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.core.settings_service import SettingsService
from app.db.base import get_db
from app.db.models import Chunk, Document, DocumentStatus, KnowledgeBase

DbSession = Annotated[Session, Depends(get_db)]


def get_settings_service(db: DbSession) -> SettingsService:
    return SettingsService(db)


SettingsDep = Annotated[SettingsService, Depends(get_settings_service)]


def problem(status_code: int, detail: str):
    """统一的错误返回，前端只需看 detail 就能给人话提示。"""
    from fastapi import HTTPException

    return HTTPException(status_code=status_code, detail=detail)


# ----------------------------------------------------------------------
# 知识库统计（实时聚合，不落库，杜绝计数器漂移）
# ----------------------------------------------------------------------
def kb_counts(db: Session, kb_ids: list[str]) -> dict[str, dict[str, int]]:
    if not kb_ids:
        return {}

    out: dict[str, dict[str, int]] = {
        kb_id: {"document_count": 0, "ready_document_count": 0, "chunk_count": 0}
        for kb_id in kb_ids
    }

    doc_rows = db.execute(
        select(
            Document.kb_id,
            func.count(Document.id),
            func.sum(case((Document.status == DocumentStatus.READY, 1), else_=0)),
        )
        .where(Document.kb_id.in_(kb_ids))
        .group_by(Document.kb_id)
    ).all()
    for kb_id, total, ready in doc_rows:
        if kb_id in out:
            out[kb_id]["document_count"] = int(total or 0)
            out[kb_id]["ready_document_count"] = int(ready or 0)

    chunk_rows = db.execute(
        select(Chunk.kb_id, func.count(Chunk.id))
        .where(Chunk.kb_id.in_(kb_ids))
        .group_by(Chunk.kb_id)
    ).all()
    for kb_id, total in chunk_rows:
        if kb_id in out:
            out[kb_id]["chunk_count"] = int(total or 0)

    return out


def get_kb_or_404(db: Session, kb_id: str) -> KnowledgeBase:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise problem(404, f"知识库不存在：{kb_id}")
    return kb


__all__ = [
    "DbSession",
    "SettingsDep",
    "get_kb_or_404",
    "get_settings_service",
    "kb_counts",
    "problem",
]
