"""末尾引用堆砌过滤器的测试。

对应「回答过于死板」那次反馈的后半段：提示词里已经写明
「不要把编号堆在末尾」，7B 模型依然照堆（实测 `[1][2][3][4][5][6]`），
所以不得不在输出侧兜一道。

这里重点测两件容易搞错的事：

  1. **别误伤正常的行内引用** —— 句中的 `[1]` 是真正有用的溯源信息，
     删掉就把 RAG 的核心价值删了；
  2. **流式扣尾要能正确放行** —— 扣住的字符后面一旦出现正文，
     必须原样补发，否则回答会缺字。
"""

from __future__ import annotations

from app.core.rag.postprocess import (
    MIN_DUMP_MARKERS,
    TrailingCitationFilter,
    count_markers,
    strip_trailing_citation_dump,
)


class TestCountMarkers:
    def test_数出编号个数(self):
        assert count_markers("[1][2][3]") == 3

    def test_多个数字也认得(self):
        assert count_markers("[12]") == 1

    def test_没有编号时为零(self):
        assert count_markers("普通文本") == 0

    def test_三位数不算编号(self):
        """编号上限两位 —— 检索片段数不可能到三位，三位更可能是年份 [2024]。"""
        assert count_markers("[2024]") == 0


class TestStripTrailingDump:
    def test_删掉末尾堆砌(self):
        assert strip_trailing_citation_dump("结论就是这样。[1][2][3]") == "结论就是这样。"

    def test_带空格也删(self):
        assert strip_trailing_citation_dump("结论。[1] [2] [3]") == "结论。"

    def test_少于阈值不删(self):
        """句尾标一两个来源是正常的，不该动。"""
        assert strip_trailing_citation_dump("结论。[1]") == "结论。[1]"
        assert strip_trailing_citation_dump("结论。[1][2]") == "结论。[1][2]"

    def test_不影响行内引用(self):
        text = "因为 X[1]，所以 Y[2]，最终得出 Z。"
        assert strip_trailing_citation_dump(text) == text

    def test_末尾堆砌后还有其他内容时不删(self):
        text = "结论。[1][2][3] 补充说明在后面。"
        assert strip_trailing_citation_dump(text) == text

    def test_没有引用时原样返回(self):
        assert strip_trailing_citation_dump("就是一句话。") == "就是一句话。"

    def test_阈值可调(self):
        assert strip_trailing_citation_dump("结论。[1][2]", min_markers=2) == "结论。"

    def test_全篇只有引用时删成空(self):
        assert strip_trailing_citation_dump("[1][2][3]") == ""


class TestStreamingFilter:
    def _run(self, chunks: list[str]) -> tuple[str, str]:
        """按块喂入，返回 (实际发给前端的内容, 过滤器扣下未发的最终裁决)。"""
        f = TrailingCitationFilter()
        out: list[str] = []
        for c in chunks:
            out.append(f.feed(c))
        tail = f.flush()
        return "".join(out), tail

    def test_普通文本逐块放行(self):
        sent, tail = self._run(["这是", "一句", "话。"])
        assert sent + tail == "这是一句话。"

    def test_末尾堆砌被拦下(self):
        sent, tail = self._run(["结论。", "[1]", "[2]", "[3]"])
        assert sent + tail == "结论。"
        assert "[1]" not in sent + tail

    def test_末尾只有一两个引用时补发(self):
        sent, tail = self._run(["结论。", "[1]"])
        assert sent + tail == "结论。[1]"

    def test_扣住的引用后面来了正文要补发(self):
        """关键回归：扣尾不能把正常的行内引用吃掉。"""
        sent, tail = self._run(["因为 X", "[1]", "，所以 Y。"])
        assert sent + tail == "因为 X[1]，所以 Y。"

    def test_逐字符喂入与整块喂入结果一致(self):
        text = "结论是这样。[1][2][3]"
        whole, _ = self._run([text])
        char_by_char, _ = self._run(list(text))
        assert whole == char_by_char == "结论是这样。"

    def test_扣住期间不提前吐出编号(self):
        """模拟流式：喂到第三个引用时，它们都不该已经发出去。"""
        f = TrailingCitationFilter()
        assert f.feed("结论。") == "结论。"
        assert f.feed("[1]") == ""
        assert f.feed("[2]") == ""
        mid = f.feed("[3]")
        assert "[1]" not in mid and "[2]" not in mid and "[3]" not in mid
        assert f.flush() == ""  # 达到阈值，整串丢弃

    def test_编号被拆成三块到达时也能拦住(self):
        """回归：流式下 `[1]` 可能是 `[`、`1`、`]` 三个块，
        只匹配完整编号的正则会让过滤器完全失效。"""
        sent, tail = self._run(["结论。", "[", "1", "]", "[", "2", "]", "[", "3", "]"])
        assert sent + tail == "结论。"
        assert "[" not in sent + tail

    def test_长回答里的行内引用全保留(self):
        text = (
            "第一点依据甲[1]。"
            "第二点依据乙[2]。"
            "第三点依据丙[3]。"
            "综上，结论成立。[1][2][3]"
        )
        sent, tail = self._run([text])
        final = sent + tail
        # 每个编号只该剩下行内那一次，末尾那串被删掉
        for marker in ("[1]", "[2]", "[3]"):
            assert final.count(marker) == 1, f"{marker} 应当只剩行内那一次，实际 {final!r}"
        assert final.endswith("综上，结论成立。")

    def test_flush_可以重复调用(self):
        f = TrailingCitationFilter()
        f.feed("结论。[1][2][3]")
        assert f.flush() == ""
        assert f.flush() == ""

    def test_空输入不报错(self):
        f = TrailingCitationFilter()
        assert f.feed("") == ""
        assert f.flush() == ""

    def test_阈值常量是3(self):
        assert MIN_DUMP_MARKERS == 3
