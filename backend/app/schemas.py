"""API 数据契约（Pydantic 模型）。

约定：
  * 请求体用 `...Create` / `...Update`，响应体用 `...Out`；
  * 所有时间字段统一序列化成**带 Z 的 UTC ISO 串**。
    SQLite 的 DateTime 不带时区，直接吐出去前端会当成本地时间，
    差 8 小时这种 bug 极难排查，所以在契约层一次性钉死。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer


def _iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


UtcDatetime = Annotated[datetime, PlainSerializer(_iso_utc, return_type=str)]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ======================================================================
# 知识库
# ======================================================================
class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128, description="知识库名称，全局唯一")
    description: str = Field(default="", max_length=1000)
    # 留空则用全局配置里的向量模型
    embed_model: str | None = Field(default=None, max_length=128)
    chunk_size: int | None = Field(default=None, ge=100, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=1000)
    # 注意：embed_model / chunk_size 建库后不可改 ——
    # 改了会让已有向量与新查询不在同一语义空间。要换就新建知识库。


class KnowledgeBaseOut(ORMModel):
    id: str
    name: str
    description: str
    embed_model: str
    embed_dim: int | None
    chunk_size: int
    chunk_overlap: int
    collection_name: str
    created_at: UtcDatetime
    updated_at: UtcDatetime
    # 下面三项在读接口里实时聚合，不落库
    document_count: int = 0
    ready_document_count: int = 0
    chunk_count: int = 0


class KnowledgeBaseDetail(KnowledgeBaseOut):
    documents: list["DocumentOut"] = Field(default_factory=list)


# ======================================================================
# 文档
# ======================================================================
DocumentStatusLiteral = Literal["pending", "parsing", "embedding", "ready", "failed"]


class DocumentOut(ORMModel):
    id: str
    kb_id: str
    filename: str
    file_type: str
    size_bytes: int
    status: DocumentStatusLiteral | str
    chunk_count: int
    char_count: int
    page_count: int
    error: str
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class ChunkOut(ORMModel):
    id: str
    doc_id: str
    kb_id: str
    ordinal: int
    content: str
    char_count: int
    page: int | None
    filename: str = ""


class DocumentUploadResult(BaseModel):
    """批量上传的逐文件结果。部分失败不应该让整个请求失败。"""

    filename: str
    ok: bool
    document: DocumentOut | None = None
    error: str | None = None
    duplicated: bool = False


# ======================================================================
# 对话
# ======================================================================
class ConversationCreate(BaseModel):
    kb_id: str | None = None
    title: str = Field(default="新对话", max_length=256)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=256)


class Citation(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    filename: str
    page: int | None = None
    ordinal: int
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    source: str = "none"
    preview: str = ""


class MessageOut(ORMModel):
    id: str
    conversation_id: str
    role: str
    content: str
    rewritten_query: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    retrieval_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: UtcDatetime


class ConversationOut(ORMModel):
    id: str
    kb_id: str | None
    title: str
    created_at: UtcDatetime
    updated_at: UtcDatetime
    message_count: int = 0


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """流式问答请求。

    message        : 本轮提问
    conversation_id: 为空则自动新建会话
    kb_id          : 新建会话时必须给；已有会话沿用会话自己的知识库
    provider       : 单次覆盖生成通道（ollama / openai），用于双通道对比
    mode           : 单次覆盖检索模式（hybrid / dense_only / sparse_only），用于消融实验
    """

    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None
    kb_id: str | None = None
    provider: Literal["ollama", "openai"] | None = None
    mode: Literal["hybrid", "dense_only", "sparse_only"] | None = None


class ChatPrepared(BaseModel):
    """流式接口在开始推流前的准备结果。"""

    conversation_id: str
    user_message_id: str
    kb_id: str


# ======================================================================
# 设置
# ======================================================================
class SettingDependency(BaseModel):
    """依赖关系：只有当 `key` 取 `values` 中的值时，本项才真正生效。"""

    key: str
    values: list[str] = Field(default_factory=list)


class SettingItem(BaseModel):
    key: str
    value: Any
    configured: bool = True
    overridden: bool = False
    label: str
    group: str
    # 二级分组。空字符串表示「本组主控项」，界面会置顶并突出展示。
    section: str = ""
    description: str = ""
    type: str
    secret: bool = False
    choices: list[str] | None = None
    # 选项显示名 {值: 显示文本}，供下拉框与依赖提示使用
    choice_labels: dict[str, str] | None = None
    # 动态选项来源（如 ollama_embedding_models）。
    # 非空表示 `choices` 是**运行时建议**而非强约束枚举，界面应渲染成
    # 「可自由输入 + 下拉建议」——Ollama 没启动时建议列表会是空的，
    # 此时若渲染成严格下拉框，用户连模型名都改不了。
    choices_source: str | None = None
    # 动态候选是否真的取到了。False 表示查询失败（如 Ollama 未启动），
    # 此时 `choices` 只含当前值，界面应把它当作「未知」而不是「没装这个模型」。
    choices_loaded: bool = True
    minimum: float | None = None
    maximum: float | None = None
    # 生效范围提示，例如「仅对新建知识库生效」
    scope: str = ""
    depends_on: SettingDependency | None = None


class SettingGroup(BaseModel):
    """一级分组：顺序与一句话定位由后端声明，前端只负责渲染。"""

    name: str
    summary: str = ""
    order: int = 0


class SettingsOut(BaseModel):
    items: list[SettingItem]
    groups: list[SettingGroup] = Field(default_factory=list)
    readonly: dict[str, Any] = Field(default_factory=dict)
    overridden_keys: list[str] = Field(default_factory=list)


class SettingsUpdate(BaseModel):
    values: dict[str, Any]


class SettingsReset(BaseModel):
    keys: list[str] | None = None


# ======================================================================
# 系统 / 健康
# ======================================================================
class ProviderHealth(BaseModel):
    name: str
    ok: bool
    detail: str = ""
    model: str = ""
    models: list[str] = Field(default_factory=list)
    base_url: str = ""


class HealthOut(BaseModel):
    status: str
    version: str
    database: bool
    vector_store: bool
    active_provider: str
    active_model: str
    embed_model: str
    providers: list[ProviderHealth] = Field(default_factory=list)


KnowledgeBaseDetail.model_rebuild()
