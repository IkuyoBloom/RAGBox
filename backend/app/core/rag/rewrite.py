"""多轮对话的查询改写。

## 要解决的问题

用户第二轮问「那它的时间复杂度呢？」，直接拿这句话去检索必然失效 ——
向量库里没有「它」的指代对象。同理 BM25 也只会匹配到「时间复杂度」。

改写就是把 `(历史对话, 追问)` 压缩成 `(可独立检索的完整问题)`：

    历史：Q. 什么是快速排序？  A. 快速排序是一种分治排序算法…
    追问：那它的时间复杂度呢？
    ->   快速排序的时间复杂度是多少

## 工程上的三个细节

1. **用非流式、低温、短输出**：改写是中间步骤，用户看不到，
   速度优先（`max_tokens` 限死 96，避免模型写一段解释）。
2. **输出必须清洗**：小模型很爱输出「改写后的问题是：…」这种前后缀，
   这里做去前缀 + 去引号 + 单行化处理。
3. **失败必须无害降级**：改写失败或结果异常（过长、为空、包含换行）时，
   静默回退到原始问题。绝不能让一个可选的优化环节把主流程搞挂。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.core.llm import ChatMessage, LLMError, LLMProvider

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = """你是一个检索查询改写器。

任务：根据对话历史，把用户的最新提问改写成一个**可以独立检索**的完整问题。

规则：
1. 把代词（它、这个、该方法、上面提到的）替换成历史中明确指代的对象。
2. 补全被省略的主语、研究对象、限定条件。
3. 只输出改写后的问题本身，不要输出任何解释、前缀、引号或标点之外的符号。
4. 不要回答问题。
5. 如果最新提问本身已经完整、不依赖历史，原样输出即可。"""

# 模型爱加的前缀，逐个剥掉
_PREFIX_PATTERNS = [
    r"^改写后(的)?(问题|查询|问句)?\s*[:：]\s*",
    r"^改写\s*[:：]\s*",
    r"^完整(的)?(问题|查询)\s*[:：]\s*",
    r"^独立(的)?(问题|查询)\s*[:：]\s*",
    r"^查询\s*[:：]\s*",
    r"^问题\s*[:：]\s*",
    r"^输出\s*[:：]\s*",
]

MAX_REWRITE_CHARS = 200
MAX_HISTORY_TURNS = 4


def _clean_rewrite(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    # 只取第一行（模型偶尔会附上解释）
    text = text.splitlines()[0].strip()
    for pattern in _PREFIX_PATTERNS:
        new = re.sub(pattern, "", text, flags=re.IGNORECASE)
        if new != text:
            text = new.strip()
    # 剥掉整体包裹的引号 / 书名号外层
    text = text.strip().strip('"\'“”‘’`')
    return text.strip()


def _looks_sane(rewritten: str, original: str) -> bool:
    """改写结果的合理性检查。宁可不用，也不能用坏结果去检索。"""
    if not rewritten:
        return False
    if len(rewritten) > MAX_REWRITE_CHARS:
        return False
    # 改写通常比原问题长；但如果暴涨 4 倍以上，多半是模型开始自问自答了
    if len(rewritten) > max(40, len(original) * 4):
        return False
    # 明显是解释性句子，而不是问题
    if rewritten.count("。") > 2:
        return False
    return True


async def rewrite_query(
    provider: LLMProvider,
    question: str,
    history: Sequence[ChatMessage],
    *,
    enabled: bool = True,
) -> str | None:
    """返回改写后的查询；不需要改写或改写失败时返回 None。"""
    if not enabled or not history:
        return None

    # 只保留最近若干轮，并且只取纯文本，避免把上一轮的参考资料也带上（会浪费上下文）
    recent = list(history)[-MAX_HISTORY_TURNS * 2 :]
    transcript_lines: list[str] = []
    for msg in recent:
        role = "用户" if msg.role == "user" else "助手"
        content = msg.content.strip()
        if len(content) > 300:
            content = content[:300] + "…"
        transcript_lines.append(f"{role}：{content}")

    user_content = (
        "【对话历史】\n"
        + "\n".join(transcript_lines)
        + f"\n\n【最新提问】\n{question}\n\n请输出改写后的问题："
    )

    messages = [
        ChatMessage(role="system", content=REWRITE_SYSTEM),
        ChatMessage(role="user", content=user_content),
    ]

    try:
        result = await provider.chat(
            messages, temperature=0.0, max_tokens=96
        )
    except LLMError as exc:
        logger.warning("查询改写失败，回退到原始问题：%s", exc)
        return None
    except Exception as exc:  # pragma: no cover - 兜底
        logger.warning("查询改写异常，回退到原始问题：%s", exc)
        return None

    cleaned = _clean_rewrite(result.content)
    if not _looks_sane(cleaned, question):
        logger.debug("查询改写结果不合理，忽略：%r", result.content[:200])
        return None
    if cleaned == question.strip():
        return None  # 没变化，不必记录
    return cleaned


__all__ = ["MAX_HISTORY_TURNS", "REWRITE_SYSTEM", "rewrite_query"]
