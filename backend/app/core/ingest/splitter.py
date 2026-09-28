"""中文友好的递归字符切分。

为什么不用 LangChain 的 `RecursiveCharacterTextSplitter`：
  * 它的默认分隔符是英文句号 `\\n\\n` / `\\n` / ` ` / `""`，
    中文文档用它会退化成「按空格切」然后直接硬切，断句质量很差；
  * 本课题需要把这个算法讲清楚，自己实现反而更好写进论文；
  * 少一个大依赖，安装体积和版本冲突风险都小。

算法（与 LangChain 同源，但分隔符换成中文标点优先级）：
  1. 按优先级从高到低尝试分隔符；
  2. 用当前分隔符把文本切成小片段，贪心地往缓冲里塞，直到超过 chunk_size；
  3. 超长片段（本身 > chunk_size）降级用下一级分隔符递归处理；
  4. 缓冲满时从**头部**弹出片段，直到剩余长度 ≤ chunk_overlap，实现重叠。

这样切出来的片：优先在段落/句子边界断开，尺寸受控，且相邻片有语义重叠，
不会出现「答案正好被切在两片之间」的情况。
"""

from __future__ import annotations

from dataclasses import dataclass

# 优先级从高到低：段落 -> 换行 -> 中文句末 -> 中文分句 -> 空格 -> 硬切
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "!",
    "?",
    ";",
    "……",
    "…",
    "”",
    "）",
    "】",
    "，",
    "、",
    ",",
    " ",
    "",
)

# 太短的片没有检索价值，反而稀释 Top-K
MIN_CHUNK_CHARS = 20


@dataclass(slots=True)
class SplitResult:
    text: str
    page: int | None


def _split_keep_separator(text: str, separator: str) -> list[str]:
    """按 separator 切分，并把分隔符粘回**前一段**尾部。

    丢掉标点会破坏语义（「什么是过拟合」变成「什么是过拟合」+「它指…」，
    前一片没了问号，BM25 里也就少了线索）。
    """
    if separator == "":
        return list(text)
    parts = text.split(separator)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            out.append(part + separator)
        else:
            out.append(part)
    return [p for p in out if p]


def _recursive_split(
    text: str,
    chunk_size: int,
    separators: tuple[str, ...],
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # 没有任何分隔符可用（例如一整段无标点的连续中文）：只能硬切
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    rest = separators[1:]
    pieces = _split_keep_separator(text, separator)

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for piece in pieces:
        plen = len(piece)
        if plen > chunk_size and rest:
            # 这一片自己就超长，交给下一级分隔符
            if buffer:
                chunks.append("".join(buffer))
                buffer, buffer_len = [], 0
            chunks.extend(_recursive_split(piece, chunk_size, rest))
            continue

        if buffer_len + plen > chunk_size and buffer:
            chunks.append("".join(buffer))
            buffer, buffer_len = [], 0

        buffer.append(piece)
        buffer_len += plen

    if buffer:
        chunks.append("".join(buffer))

    return chunks


def _merge_with_overlap(
    splits: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """把碎片合并成最终切片，并制造重叠。"""
    docs: list[str] = []
    current: list[str] = []
    total = 0

    for piece in splits:
        plen = len(piece)

        if current and total + plen > chunk_size:
            docs.append("".join(current).strip())
            # 从头部弹出，保留尾部约 chunk_overlap 个字符作为重叠
            while current and total > chunk_overlap:
                total -= len(current[0])
                current.pop(0)
            # 若弹出后仍塞不下这一片（片本身很大），继续弹到能放下
            while current and total + plen > chunk_size:
                total -= len(current[0])
                current.pop(0)

        current.append(piece)
        total += plen

    if current:
        docs.append("".join(current).strip())

    return [d for d in docs if d]


def split_text(
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:
    """切分纯文本。"""
    if not text or not text.strip():
        return []

    chunk_size = max(50, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))

    # ① 先按段落切出「硬边界」片段：段落是作者划定的语义单元，
    #    能整段放下就不要从中间劈开。
    pieces: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            pieces.append(para)
        else:
            # 超长段落交给递归切分器按标点优先级降级处理
            pieces.extend(_recursive_split(para, chunk_size, separators))

    # ② 再把碎片贪心打包到 chunk_size 并制造重叠。
    #    少了这一步，一篇由大量短段落组成的文档会切出几十个十几字的碎块，
    #    检索时 Top-K 全被这些碎片占满，真正的上下文反而进不来。
    chunks = _merge_with_overlap(pieces, chunk_size, chunk_overlap)

    # ③ 末尾不足 min_chunk_chars 的残片并回上一片，避免出现无检索价值的尾巴
    if len(chunks) > 1 and len(chunks[-1]) < min_chunk_chars:
        merged_tail = chunks[-2] + "\n" + chunks[-1]
        if len(merged_tail) <= chunk_size * 2:
            chunks[-2] = merged_tail
            chunks.pop()

    return [c for c in chunks if c.strip()]


def split_blocks(
    blocks: list,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[SplitResult]:
    """按块（通常一页一块）切分，保留每片的来源页码。"""
    results: list[SplitResult] = []
    for block in blocks:
        text = getattr(block, "text", "")
        page = getattr(block, "page", None)
        for piece in split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            results.append(SplitResult(text=piece, page=page))
    return results


__all__ = [
    "DEFAULT_SEPARATORS",
    "MIN_CHUNK_CHARS",
    "SplitResult",
    "split_blocks",
    "split_text",
]
