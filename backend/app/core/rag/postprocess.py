"""生成结果的输出侧后处理。

## 为什么需要它

提示词能约束大部分行为，但有些毛病光靠提示词按不住 —— 最典型的是
**末尾引用堆砌**。即使系统提示里明确写了「不要把编号堆在末尾」，
7B 量级的模型依然会在回答最后补一串：

    …综上所述，这些资料主要围绕任务与角色展开。[1][2][3][4][5][6]

这不是它真的依据了全部六条，而是把「要标注来源」这条规则满足得
「彻底」了一点 —— 给一个总结句配六个引用，看起来像在凑数。
实测提示词写了也拦不住，所以只能在输出侧兜一道。

## 为什么必须在流式里做

文本一旦作为 delta 推给前端就已经渲染出来了，**事后删不掉**。
所以不能等回答结束再清理，必须把「可能是末尾编号」的尾巴**先扣住不发**，
等下一个字符到了再决定：

  * 后面还有正文  -> 那不是末尾，把扣住的连同正文一起放行；
  * 一直没有正文  -> 确实是末尾，且数量达到阈值，判定为堆砌并丢弃。

扣住的窗口极小（只有形如 `[12]` 的字符），所以不影响流式的观感。
"""

from __future__ import annotations

import re

# 末尾可能是引用编号、**或没写完的引用编号片段**。
#
# 必须容忍「没写完」：流式下 `[1]` 未必整块到达 —— 逐 token 推送时
# 很可能是 `[`、`1`、`]` 三个块。只匹配完整的 `[1]` 会导致过滤器
# 在真实流式场景下完全失效（本次测试就抓到了这个漏洞）。
#
# 代价是：正文末尾若恰好以一个孤立的 `[` 结尾，会多扣住一个字符，
# 等下一个字符到达立刻放行，肉眼无感。
_TRAILING_CITATIONS = re.compile(r"(?:\s*\[\d{0,2}\]?)+$")

_MARKER = re.compile(r"\[\d{1,2}\]")

# 少于这个数量不算「堆砌」—— 句子末尾标一两个来源是正常的溯源行为，
# 不该被当作噪声删掉。
MIN_DUMP_MARKERS = 3


def count_markers(text: str) -> int:
    """数一段文本里有几个引用编号。"""
    return len(_MARKER.findall(text))


def strip_trailing_citation_dump(text: str, *, min_markers: int = MIN_DUMP_MARKERS) -> str:
    """一次性版本：直接清理已经生成完的整段文本。

    流式路径用下面的 TrailingCitationFilter；这个函数供非流式调用与测试使用。
    """
    match = _TRAILING_CITATIONS.search(text)
    if match is None:
        return text
    if count_markers(match.group(0)) < min_markers:
        return text
    return text[: match.start()].rstrip()


class TrailingCitationFilter:
    """流式过滤器：扣住末尾的引用编号，确认是堆砌就丢掉。

    用法：

        f = TrailingCitationFilter()
        async for chunk in provider.stream(...):
            visible = f.feed(chunk.text)
            if visible:
                yield delta(visible)
        tail = f.flush()          # 回答结束，做最终裁决
        if tail:
            yield delta(tail)
    """

    def __init__(self, min_markers: int = MIN_DUMP_MARKERS) -> None:
        self._min = min_markers
        self._pending = ""

    def feed(self, text: str) -> str:
        """喂入新增文本，返回此刻可以安全发给前端的部分。"""
        if not text:
            return ""
        self._pending += text

        match = _TRAILING_CITATIONS.search(self._pending)
        if match is None:
            # 尾部不是引用编号，全部可以发出去
            out, self._pending = self._pending, ""
            return out

        # 尾部像引用编号，先扣住；它前面的正文照常发出
        out = self._pending[: match.start()]
        self._pending = self._pending[match.start() :]
        return out

    def flush(self) -> str:
        """回答结束。返回仍需补发的文本；判定为堆砌则返回空串。"""
        held, self._pending = self._pending, ""
        if count_markers(held) >= self._min:
            return ""
        return held


__all__ = [
    "MIN_DUMP_MARKERS",
    "TrailingCitationFilter",
    "count_markers",
    "strip_trailing_citation_dump",
]
