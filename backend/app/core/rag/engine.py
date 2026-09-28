"""RAG 引擎：把「改写 -> 检索 -> 组装 -> 流式生成 -> 落库」串成一条链。

## 为什么引擎自己管数据库会话，而不接收上层传进来的 Session

对话接口返回的是 SSE 流式响应。FastAPI 的依赖注入会在**响应结束后**才关闭
`Depends(get_db)` 的会话，也就是说生成过程中会话一直挂着；而生成一次要几十秒，
这期间如果还持有 SQLite 的读锁/写锁，其他请求会被拖住。

所以引擎采用「短会话」模式：每一步用完即关。
副作用是引擎完全不依赖请求上下文，**评估模块和离线脚本可以直接复用同一份代码** ——
这保证了「论文里的实验数据」和「系统在线行为」出自同一条实现，不会出现两套逻辑对不上。

## 事件协议（与前端 / SSE 一一对应）

    {"type": "start",     "conversation_id", "kb_id", "question"}
    {"type": "rewrite",   "query", "original"}
    {"type": "citations", "citations": [...]}
    {"type": "trace",     "mode", "dense_hits", "sparse_hits", "elapsed_ms", "stages"}
    {"type": "delta",     "text"}
    {"type": "done",      "message_id", "provider", "model", "latency_ms",
                          "retrieval_ms", "prompt_tokens", "completion_tokens", "finish_reason"}
    {"type": "error",     "message", "provider"}
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.llm import ChatMessage, LLMError, build_provider
from app.core.rag.postprocess import TrailingCitationFilter
from app.core.rag.prompt import (
    NO_CONTEXT_ANSWER,
    _EMPTY_KB_ANSWER,
    build_messages,
)
from app.core.retrieval import RetrievalTrace, RetrievedChunk, SearchMode, retrieve
from app.core.settings_service import SettingsService
from app.db.base import session_scope
from app.db.models import (
    Chunk,
    Conversation,
    DocumentStatus,
    KnowledgeBase,
    Message,
    utcnow,
)

logger = logging.getLogger(__name__)

# 历史对话保留轮数（一问一答算两轮）
HISTORY_WINDOW = 6


# ----------------------------------------------------------------------
# 内部小工具：每个都是一次独立短事务
# ----------------------------------------------------------------------
def _load_config() -> dict[str, Any]:
    with session_scope() as db:
        return SettingsService(db).effective()


def _load_kb(db: Session, kb_id: str) -> KnowledgeBase:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise ValueError(f"知识库不存在：{kb_id}")
    return kb


def _load_history(db: Session, conversation_id: str, question: str) -> list[ChatMessage]:
    rows = (
        db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(HISTORY_WINDOW + 2)
        )
        .scalars()
        .all()
    )
    rows = list(reversed(rows))

    # 本轮提问已经由接口层落库，它不该作为「历史」再喂给模型
    if rows and rows[-1].role == "user" and rows[-1].content.strip() == question.strip():
        rows = rows[:-1]

    out: list[ChatMessage] = []
    for row in rows[-HISTORY_WINDOW:]:
        if row.role not in ("user", "assistant"):
            continue
        content = row.content.strip()
        if not content:
            continue
        out.append(ChatMessage(role=row.role, content=content))
    return out


def _ready_chunk_exists(db: Session, kb_id: str) -> bool:
    row = db.execute(
        select(Chunk.id).where(Chunk.kb_id == kb_id).limit(1)
    ).first()
    return row is not None


def _save_assistant_message(
    conversation_id: str,
    *,
    content: str,
    citations: list[dict[str, Any]],
    rewritten_query: str | None,
    provider: str,
    model: str,
    latency_ms: int,
    retrieval_ms: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    meta: dict[str, Any] | None = None,
) -> str:
    with session_scope() as db:
        msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            citations=citations,
            rewritten_query=rewritten_query,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            retrieval_ms=retrieval_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            meta=meta or {},
        )
        db.add(msg)
        db.flush()
        convo = db.get(Conversation, conversation_id)
        if convo is not None:
            convo.updated_at = utcnow()
        return msg.id


def _stamp_rewrite(user_message_id: str | None, rewritten: str | None) -> None:
    if not user_message_id or not rewritten:
        return
    with session_scope() as db:
        msg = db.get(Message, user_message_id)
        if msg is not None:
            msg.rewritten_query = rewritten


def _auto_title(conversation_id: str, question: str) -> None:
    """首轮提问自动作为会话标题，省掉用户手动命名的麻烦。"""
    with session_scope() as db:
        convo = db.get(Conversation, conversation_id)
        if convo is None:
            return
        if convo.title and convo.title != "新对话":
            return
        convo.title = question.strip()[:40] or "新对话"


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
async def stream_answer(
    *,
    conversation_id: str,
    kb_id: str,
    question: str,
    user_message_id: str | None = None,
    provider_name: str | None = None,
    mode: SearchMode | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """流式回答。逐条 yield 事件字典，由接口层序列化成 SSE。"""
    cfg = _load_config()

    yield {
        "type": "start",
        "conversation_id": conversation_id,
        "kb_id": kb_id,
        "question": question,
    }

    # ---------------- 准备：知识库 / 历史 ----------------
    try:
        with session_scope() as db:
            kb = _load_kb(db, kb_id)
            # 把 ORM 对象转成普通值，出事务后继续用；expire_on_commit 会失效属性
            kb_snapshot = {
                "id": kb.id,
                "name": kb.name,
                "collection_name": kb.collection_name,
                "embed_model": kb.embed_model,
            }
            history = _load_history(db, conversation_id, question)
            has_chunks = _ready_chunk_exists(db, kb.id)
    except ValueError as exc:
        yield {"type": "error", "message": str(exc), "provider": ""}
        return

    if not has_chunks:
        _auto_title(conversation_id, question)
        message_id = _save_assistant_message(
            conversation_id,
            content=_EMPTY_KB_ANSWER,
            citations=[],
            rewritten_query=None,
            provider="",
            model="",
            latency_ms=0,
            retrieval_ms=0,
            prompt_tokens=None,
            completion_tokens=None,
            meta={"reason": "empty_kb"},
        )
        yield {"type": "citations", "citations": []}
        yield {"type": "delta", "text": _EMPTY_KB_ANSWER}
        yield {
            "type": "done",
            "message_id": message_id,
            "provider": "",
            "model": "",
            "latency_ms": 0,
            "retrieval_ms": 0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": "empty_kb",
        }
        return

    # ---------------- 生成通道 ----------------
    try:
        provider = build_provider(cfg, provider=provider_name)
    except LLMError as exc:
        yield {"type": "error", "message": str(exc), "provider": exc.provider}
        return

    # ---------------- ① 查询改写 ----------------
    rewritten: str | None = None
    if history and cfg.get("enable_query_rewrite", True):
        from app.core.rag.rewrite import rewrite_query

        rewritten = await rewrite_query(provider, question, history, enabled=True)
        if rewritten:
            _stamp_rewrite(user_message_id, rewritten)
            yield {"type": "rewrite", "query": rewritten, "original": question}

    search_query = rewritten or question

    # ---------------- ② 检索 ----------------
    trace = RetrievalTrace(query=search_query, rewritten_query=rewritten)
    try:
        with session_scope() as db:
            kb = _load_kb(db, kb_snapshot["id"])
            chunks: list[RetrievedChunk] = await retrieve(
                db, cfg, kb, search_query, mode=mode, trace=trace
            )
    except Exception as exc:
        logger.exception("检索失败")
        yield {
            "type": "error",
            "message": f"检索失败：{type(exc).__name__}: {exc}",
            "provider": "",
        }
        return

    citations = [c.to_citation(i) for i, c in enumerate(chunks, start=1)]
    yield {"type": "citations", "citations": citations}
    yield {
        "type": "trace",
        "mode": trace.mode,
        "dense_hits": trace.dense_hits,
        "sparse_hits": trace.sparse_hits,
        "fused_hits": trace.fused_hits,
        "elapsed_ms": trace.elapsed_ms,
        "stages": trace.stages,
    }

    _auto_title(conversation_id, question)

    # ---------------- ③ 没检索到 -> 直接给兜底回答 ----------------
    if not chunks:
        message_id = _save_assistant_message(
            conversation_id,
            content=NO_CONTEXT_ANSWER,
            citations=[],
            rewritten_query=rewritten,
            provider=provider.name,
            model=provider.model,
            latency_ms=0,
            retrieval_ms=trace.elapsed_ms,
            prompt_tokens=None,
            completion_tokens=None,
            meta={"reason": "no_retrieval_hit", "trace": _trace_dict(trace)},
        )
        yield {"type": "delta", "text": NO_CONTEXT_ANSWER}
        yield {
            "type": "done",
            "message_id": message_id,
            "provider": provider.name,
            "model": provider.model,
            "latency_ms": 0,
            "retrieval_ms": trace.elapsed_ms,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": "no_retrieval_hit",
        }
        return

    # ---------------- ④ 流式生成 ----------------
    messages = build_messages(
        question,
        chunks,
        history=history,
        strict=bool(cfg.get("strict_grounding", True)),
    )

    started = time.perf_counter()
    pieces: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason = "stop"

    # 输出侧过滤器：扣住末尾的引用编号，等回答结束再裁决是不是堆砌。
    # 为什么必须在流式里做：文本一旦作为 delta 发出去就已经渲染了，事后删不掉。
    citation_filter = TrailingCitationFilter()

    try:
        async for chunk in provider.stream(
            messages,
            temperature=float(cfg.get("temperature") or 0.2),
            max_tokens=int(cfg.get("max_tokens") or 1024),
        ):
            if chunk.text:
                visible = citation_filter.feed(chunk.text)
                if visible:
                    pieces.append(visible)
                    yield {"type": "delta", "text": visible}
            if chunk.finished:
                prompt_tokens = chunk.prompt_tokens
                completion_tokens = chunk.completion_tokens
    except LLMError as exc:
        # 已经吐了一部分就保留，告诉前端是「中断」而不是「空回答」。
        # 被扣住的尾巴也要放出来 —— 中断时它是被截断的正文，不是堆砌。
        tail = citation_filter.flush()
        if tail:
            pieces.append(tail)
            yield {"type": "delta", "text": tail}
        yield {
            "type": "error",
            "message": str(exc),
            "provider": exc.provider,
            "partial": "".join(pieces),
        }
        return
    except Exception as exc:  # pragma: no cover - 兜底
        logger.exception("生成阶段异常")
        yield {
            "type": "error",
            "message": f"生成失败：{type(exc).__name__}: {exc}",
            "provider": provider.name,
            "partial": "".join(pieces),
        }
        return

    # 生成正常结束：对被扣住的尾巴做最终裁决 ——
    # 数量达到阈值就判定为「末尾引用堆砌」并丢弃，否则原样补发。
    tail = citation_filter.flush()
    if tail:
        pieces.append(tail)
        yield {"type": "delta", "text": tail}

    answer = "".join(pieces).strip()
    latency_ms = int((time.perf_counter() - started) * 1000)
    if not answer:
        finish_reason = "empty"
        answer = "（模型没有返回任何内容。可以尝试降低温度、减少检索片段数，或换一个模型。）"

    message_id = _save_assistant_message(
        conversation_id,
        content=answer,
        citations=citations,
        rewritten_query=rewritten,
        provider=provider.name,
        model=provider.model,
        latency_ms=latency_ms,
        retrieval_ms=trace.elapsed_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        meta={
            "trace": _trace_dict(trace),
            "strict": bool(cfg.get("strict_grounding", True)),
            "search_query": search_query,
        },
    )

    yield {
        "type": "done",
        "message_id": message_id,
        "provider": provider.name,
        "model": provider.model,
        "latency_ms": latency_ms,
        "retrieval_ms": trace.elapsed_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
    }


def _trace_dict(trace: RetrievalTrace) -> dict[str, Any]:
    return {
        "query": trace.query,
        "rewritten_query": trace.rewritten_query,
        "mode": trace.mode,
        "dense_hits": trace.dense_hits,
        "sparse_hits": trace.sparse_hits,
        "fused_hits": trace.fused_hits,
        "elapsed_ms": trace.elapsed_ms,
        "stages": trace.stages,
    }


# ----------------------------------------------------------------------
# 非流式封装：评估模块 / 脚本用
# ----------------------------------------------------------------------
async def answer_once(
    *,
    conversation_id: str,
    kb_id: str,
    question: str,
    provider_name: str | None = None,
    mode: SearchMode | None = None,
) -> dict[str, Any]:
    """跑完整条链路并收集结果，返回最终答案与全部元数据。"""
    result: dict[str, Any] = {
        "answer": "",
        "citations": [],
        "rewritten_query": None,
        "trace": None,
        "provider": "",
        "model": "",
        "latency_ms": 0,
        "retrieval_ms": 0,
        "error": None,
    }
    async for event in stream_answer(
        conversation_id=conversation_id,
        kb_id=kb_id,
        question=question,
        provider_name=provider_name,
        mode=mode,
    ):
        etype = event.get("type")
        if etype == "rewrite":
            result["rewritten_query"] = event["query"]
        elif etype == "citations":
            result["citations"] = event["citations"]
        elif etype == "trace":
            result["trace"] = event
        elif etype == "delta":
            result["answer"] += event["text"]
        elif etype == "done":
            result.update(
                {
                    "provider": event["provider"],
                    "model": event["model"],
                    "latency_ms": event["latency_ms"],
                    "retrieval_ms": event["retrieval_ms"],
                }
            )
        elif etype == "error":
            result["error"] = event["message"]
    return result


__all__ = ["HISTORY_WINDOW", "answer_once", "stream_answer"]
