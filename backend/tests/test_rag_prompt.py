"""RAG 提示词的「契约」测试。

提示词是纯字符串，改起来毫无阻力 —— 但它的两个方向都会出问题：

  * **写太松**：模型开始编造，RAG 的可信度就没了（这是整个系统的立身之本）；
  * **写太死**：模型变成一台拒答机器。实测原始版本的严格模式对
    「这个资料主要讲了什么？」只回 98 个字：「根据现有资料无法详细回答该问题…」
    然后堆上 [1][2][3][4][5][6] 六个引用 —— 用户反馈「回答过于死板」。

所以这里同时钉住两个方向的性质，避免后续调提示词时顾此失彼。
"""

from __future__ import annotations

from app.core.llm import ChatMessage
from app.core.rag.prompt import (
    SYSTEM_PROMPT_LOOSE,
    SYSTEM_PROMPT_STRICT,
    build_context_block,
    build_messages,
    format_citation_label,
)
from app.core.retrieval.types import RetrievedChunk


def _chunk(content: str, *, filename: str = "论文.pdf", page: int | None = 3, cid: str = "c1"):
    return RetrievedChunk(
        chunk_id=cid,
        doc_id="d1",
        kb_id="kb1",
        ordinal=0,
        content=content,
        filename=filename,
        page=page,
        dense_rank=1,
        fused_score=1.0,
    )


# ======================================================================
# 底线：不允许编造（这是 RAG 的立身之本，任何改写都不能丢）
# ======================================================================
class TestGroundingIsPreserved:
    def test_严格模式禁止使用先验知识(self):
        assert "先验知识" in SYSTEM_PROMPT_STRICT or "你自己的知识" in SYSTEM_PROMPT_STRICT

    def test_严格模式要求只依据参考资料(self):
        assert "只用资料里的内容" in SYSTEM_PROMPT_STRICT
        assert "参考资料" in SYSTEM_PROMPT_STRICT

    def test_严格模式保留了无法回答的兜底话术(self):
        assert "根据现有资料无法回答该问题" in SYSTEM_PROMPT_STRICT

    def test_宽松模式要求区分资料与补充(self):
        assert "补充" in SYSTEM_PROMPT_LOOSE
        assert "编号" in SYSTEM_PROMPT_LOOSE

    def test_引用编号格式两套都还在(self):
        for p in (SYSTEM_PROMPT_STRICT, SYSTEM_PROMPT_LOOSE):
            assert "[1]" in p


# ======================================================================
# 反面：不许死板。这几条是「回答过于死板」那次修复的核心
# ======================================================================
class TestPromptIsNotRigid:
    def test_严格模式明确允许概括归纳(self):
        """回归：原始版本只写「只能依据资料作答」，模型于是不敢做任何总结，
        对宽泛问题一律拒答。"""
        assert "概括" in SYSTEM_PROMPT_STRICT

    def test_严格模式要求宽泛问题不要拒答(self):
        """回归：「这个资料主要讲了什么？」曾直接被拒答。"""
        assert "不要拒答" in SYSTEM_PROMPT_STRICT
        assert "宽泛" in SYSTEM_PROMPT_STRICT

    def test_两套都明确禁止引用堆砌(self):
        """回归：模型曾在一个没有实质内容的回答后面堆 [1][2][3][4][5][6]。"""
        for p in (SYSTEM_PROMPT_STRICT, SYSTEM_PROMPT_LOOSE):
            assert "堆在末尾" in p or "末尾" in p

    def test_严格模式要求分点陈述时每点都带编号(self):
        """回归：放宽措辞后模型一度整段回答一个编号都不给，
        列表里 11 个人物全无出处 —— 溯源价值直接归零。"""
        assert "每一点都要带" in SYSTEM_PROMPT_STRICT

    def test_严格模式点名禁止推测性措辞(self):
        """回归：放宽「资料不足时」的措辞后，模型开始拿常识编步骤
        （问红烧肉做法时给出了资料里根本没有的「焯水、小火慢炖」），
        而且用「可以推测」把自己包装得很谨慎。必须点名这些说法。"""
        assert "可以推测" in SYSTEM_PROMPT_STRICT
        assert "同样是编造" in SYSTEM_PROMPT_STRICT
        assert "资料中未提及" in SYSTEM_PROMPT_STRICT

    def test_两条底线放在最前面(self):
        """7B 模型对长规则列表的遵循度会衰减，所以最要紧的两条必须置顶。"""
        idx_bottom = SYSTEM_PROMPT_STRICT.index("两条必须守住的底线")
        idx_howto = SYSTEM_PROMPT_STRICT.index("怎么算答得好")
        assert idx_bottom < idx_howto
        # 底线部分要占全文靠前的位置
        assert idx_bottom < len(SYSTEM_PROMPT_STRICT) * 0.2

    def test_两套都要求自然表达(self):
        for p in (SYSTEM_PROMPT_STRICT, SYSTEM_PROMPT_LOOSE):
            assert "自然" in p
            assert "公文" in p

    def test_两套都禁止套话开头(self):
        assert "套话" in SYSTEM_PROMPT_STRICT


# ======================================================================
# 上下文拼接
# ======================================================================
class TestContextBlock:
    def test_空结果返回空串(self):
        assert build_context_block([]) == ""

    def test_编号从1开始且连续(self):
        chunks = [_chunk("甲", cid="a"), _chunk("乙", cid="b"), _chunk("丙", cid="c")]
        block = build_context_block(chunks)
        assert "[1]" in block and "[2]" in block and "[3]" in block

    def test_包含文件名与页码(self):
        label = format_citation_label(_chunk("x", filename="报告.pdf", page=7), 2)
        assert "报告.pdf" in label and "第 7 页" in label and label.startswith("[2]")

    def test_无页码时不写页码(self):
        label = format_citation_label(_chunk("x", page=None), 1)
        assert "页" not in label

    def test_正文被拼进块里(self):
        assert "量子计算" in build_context_block([_chunk("量子计算很有意思")])


class TestBuildMessages:
    def test_首条是系统提示且用严格版(self):
        msgs = build_messages("问题", [], strict=True)
        assert msgs[0].role == "system"
        assert msgs[0].content == SYSTEM_PROMPT_STRICT

    def test_宽松模式切换系统提示(self):
        msgs = build_messages("问题", [], strict=False)
        assert msgs[0].content == SYSTEM_PROMPT_LOOSE

    def test_最后一条是用户消息且含问题(self):
        msgs = build_messages("混合检索怎么做", [], strict=True)
        assert msgs[-1].role == "user"
        assert "混合检索怎么做" in msgs[-1].content

    def test_有资料时带上参考资料块(self):
        msgs = build_messages("问题", [_chunk("资料正文")], strict=True)
        assert "【参考资料】" in msgs[-1].content
        assert "资料正文" in msgs[-1].content

    def test_无资料时不出现空的参考资料块(self):
        msgs = build_messages("问题", [], strict=True)
        assert "【参考资料】" not in msgs[-1].content

    def test_历史插在系统提示与当前提问之间(self):
        history = [
            ChatMessage(role="user", content="上一轮问题"),
            ChatMessage(role="assistant", content="上一轮回答"),
        ]
        msgs = build_messages("本轮问题", [], history=history, strict=True)
        assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
        assert msgs[1].content == "上一轮问题"
        assert "本轮问题" in msgs[3].content
