"""ORM 模型。

设计要点（可写进论文的「系统设计」章节）：

1. **SQLite 是唯一事实来源（single source of truth）**，Chroma 只存
   「向量 + 最小定位元数据」，文本正文不放进去。这样：
     - 不存在两处正文不同步的问题；
     - 换向量库时数据不丢，重建索引即可；
     - BM25 与向量检索共用同一份 chunk 表，融合时天然对齐。
2. chunk 主键直接用作向量库的 id，省掉一层映射表。
3. 消息表把 citations 存成 JSON：一次问答命中哪些片段、分数多少，
   既能前端回显，也是评估模块的原始数据。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def new_id() -> str:
    """32 位十六进制 UUID，够短、够唯一，还能直接当向量库 id 用。"""
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )


class KnowledgeBase(TimestampMixin, Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # 每个知识库固定一套向量化配置：模型不同 → 向量空间不同 → 绝不能混库
    embed_model: Mapped[str] = mapped_column(String(128), nullable=False)
    # 记录建库时的维度，用于换模型后给出明确报错而不是静默出错
    embed_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)

    chunk_size: Mapped[int] = mapped_column(Integer, default=500, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=80, nullable=False)

    # Chroma collection 名，创建时生成后不再改
    collection_name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    documents: Mapped[list["Document"]] = relationship(
        back_populates="kb", cascade="all, delete-orphan", passive_deletes=True
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="kb", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentStatus:
    PENDING = "pending"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 内容指纹（sha256），用于同一知识库内秒传 / 重复检测
    content_hash: Mapped[str] = mapped_column(String(64), default="", nullable=False, index=True)

    status: Mapped[str] = mapped_column(
        String(16), default=DocumentStatus.PENDING, nullable=False, index=True
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # 解析阶段统计、耗时等，方便论文里做性能分析
    meta: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    kb: Mapped[KnowledgeBase] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (Index("ix_documents_kb_hash", "kb_id", "content_hash"),)


class Chunk(Base):
    __tablename__ = "chunks"

    # 主键 = 向量库 id
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)  # 文档内第几块
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 一-based 页码，PDF 有值；其他格式为 None
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)

    kb: Mapped[KnowledgeBase] = relationship(back_populates="chunks")
    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (UniqueConstraint("doc_id", "ordinal", name="uq_chunk_doc_ordinal"),)


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    kb_id: Mapped[str | None] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(256), default="新对话", nullable=False)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 检索阶段：改写后的查询、命中的片段与分数
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    # 生成阶段：用了哪条通道、耗时、用量 —— 论文「双通道对比」的数据来源
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retrieval_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class AppSetting(Base):
    """界面可改的运行时配置覆盖项（key -> JSON 值）。"""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )
