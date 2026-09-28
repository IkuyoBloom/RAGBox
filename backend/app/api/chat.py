"""对话接口：SSE 流式问答 + 会话管理。"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from app.api.deps import DbSession, SettingsDep, get_kb_or_404, problem
from app.core.rag import stream_answer
from app.db.models import Conversation, Document, DocumentStatus, Message
from app.schemas import (
    ChatRequest,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ConversationUpdate,
    MessageOut,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["对话"])


def _sse(payload: dict) -> str:
    """SSE 帧。

    只发 `data:` 行，事件类型放在 JSON 的 `type` 字段里 ——
    这样前端一个 onmessage 就能处理全部事件，不用为每种类型注册监听器。
    中文必须 ensure_ascii=False，否则前端拿到的是转义串，调试时很难看。
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ======================================================================
# 流式问答
# ======================================================================
@router.post("/chat/stream", summary="流式问答（Server-Sent Events）")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    db: DbSession,
    settings: SettingsDep,
) -> StreamingResponse:
    # ---------------- 解析 / 新建会话 ----------------
    if payload.conversation_id:
        convo = db.get(Conversation, payload.conversation_id)
        if convo is None:
            raise problem(404, f"会话不存在：{payload.conversation_id}")
        kb_id = convo.kb_id
        if not kb_id:
            raise problem(409, "该会话没有绑定知识库，请新建会话。")
    else:
        if not payload.kb_id:
            raise problem(422, "新建会话时必须提供 kb_id")
        get_kb_or_404(db, payload.kb_id)
        convo = Conversation(
            kb_id=payload.kb_id,
            title=payload.message.strip()[:40] or "新对话",
        )
        db.add(convo)
        db.commit()
        db.refresh(convo)
        kb_id = convo.kb_id

    get_kb_or_404(db, kb_id)

    # 提前把用户消息落库：即使生成中途失败，历史里也留得下这次提问
    user_msg = Message(
        conversation_id=convo.id,
        role="user",
        content=payload.message,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    conversation_id = convo.id
    user_message_id = user_msg.id
    cfg = settings.effective()
    provider_name = payload.provider
    mode = payload.mode

    async def event_stream():
        try:
            async for event in stream_answer(
                conversation_id=conversation_id,
                kb_id=kb_id,
                question=payload.message,
                user_message_id=user_message_id,
                provider_name=provider_name,
                mode=mode,
            ):
                # 前端断开（关标签页 / 点停止）时尽快收工，别继续烧 GPU
                if await request.is_disconnected():
                    logger.info("客户端已断开，中止生成 conversation=%s", conversation_id)
                    break
                yield _sse(event)
        except Exception as exc:  # pragma: no cover - 兜底
            logger.exception("流式问答异常")
            yield _sse(
                {"type": "error", "message": f"服务端异常：{type(exc).__name__}: {exc}", "provider": ""}
            )
        finally:
            yield _sse({"type": "eos"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 万一前面挂了 nginx，这个头能阻止它缓冲整个响应
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/preflight", summary="提问前检查：知识库是否已有可检索内容")
def preflight(kb_id: str, db: DbSession, settings: SettingsDep) -> dict:
    from app.db.models import Chunk

    kb = get_kb_or_404(db, kb_id)
    cfg = settings.effective()

    ready_docs = db.execute(
        select(func.count(Document.id)).where(
            Document.kb_id == kb_id, Document.status == DocumentStatus.READY
        )
    ).scalar_one()
    chunk_count = db.execute(
        select(func.count(Chunk.id)).where(Chunk.kb_id == kb_id)
    ).scalar_one()

    warnings: list[str] = []
    if chunk_count == 0:
        warnings.append("该知识库还没有已索引的内容，先上传文档并等待索引完成。")
    if kb.embed_model != cfg.get("ollama_embed_model"):
        warnings.append(
            f"注意：该知识库使用向量模型 `{kb.embed_model}`，"
            f"与当前全局设置 `{cfg.get('ollama_embed_model')}` 不一致。"
            f"提问时会自动使用知识库自己的模型，这是正确行为。"
        )

    return {
        "kb_id": kb_id,
        "kb_name": kb.name,
        "embed_model": kb.embed_model,
        "ready_documents": int(ready_docs or 0),
        "chunk_count": int(chunk_count or 0),
        "search_mode": "hybrid" if cfg.get("enable_hybrid", True) else "dense_only",
        "warnings": warnings,
    }


# ======================================================================
# 会话管理
# ======================================================================
@router.get("/conversations", response_model=list[ConversationOut], summary="会话列表")
def list_conversations(
    db: DbSession,
    kb_id: str | None = None,
    limit: int = 100,
) -> list[ConversationOut]:
    stmt = select(Conversation).order_by(Conversation.updated_at.desc()).limit(max(1, min(limit, 500)))
    if kb_id:
        stmt = stmt.where(Conversation.kb_id == kb_id)
    convos = db.execute(stmt).scalars().all()

    counts: dict[str, int] = {}
    if convos:
        rows = db.execute(
            select(Message.conversation_id, func.count(Message.id))
            .where(Message.conversation_id.in_([c.id for c in convos]))
            .group_by(Message.conversation_id)
        ).all()
        counts = {cid: int(n or 0) for cid, n in rows}

    out: list[ConversationOut] = []
    for convo in convos:
        item = ConversationOut.model_validate(convo)
        item.message_count = counts.get(convo.id, 0)
        out.append(item)
    return out


@router.post("/conversations", response_model=ConversationOut, status_code=201, summary="新建会话")
def create_conversation(payload: ConversationCreate, db: DbSession) -> ConversationOut:
    if payload.kb_id:
        get_kb_or_404(db, payload.kb_id)
    convo = Conversation(kb_id=payload.kb_id, title=payload.title or "新对话")
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return ConversationOut.model_validate(convo)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail, summary="会话详情（含消息）")
def get_conversation(conversation_id: str, db: DbSession) -> ConversationDetail:
    convo = db.get(Conversation, conversation_id)
    if convo is None:
        raise problem(404, f"会话不存在：{conversation_id}")

    msgs = db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    ).scalars().all()

    detail = ConversationDetail.model_validate(convo)
    detail.message_count = len(msgs)
    detail.messages = [MessageOut.model_validate(m) for m in msgs]
    return detail


@router.patch("/conversations/{conversation_id}", response_model=ConversationOut, summary="重命名会话")
def rename_conversation(
    conversation_id: str, payload: ConversationUpdate, db: DbSession
) -> ConversationOut:
    convo = db.get(Conversation, conversation_id)
    if convo is None:
        raise problem(404, f"会话不存在：{conversation_id}")
    convo.title = payload.title
    db.commit()
    db.refresh(convo)
    return ConversationOut.model_validate(convo)


@router.delete("/conversations/{conversation_id}", status_code=204, summary="删除会话")
def delete_conversation(conversation_id: str, db: DbSession) -> Response:
    convo = db.get(Conversation, conversation_id)
    if convo is None:
        raise problem(404, f"会话不存在：{conversation_id}")
    db.delete(convo)
    db.commit()
    return Response(status_code=204)
