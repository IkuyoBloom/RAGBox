"""RAG Prompt 构造。

设计取舍记录（答辩常被问到的点）：

1. **带编号的参考资料 + 强制标注 [n]**
   这是「引用溯源」能落地的前提。编号直接对应检索结果的顺序，
   前端拿同一个下标就能渲染可点击的来源卡片。

2. **严格模式 vs 宽松模式**
   * 严格（默认）：资料不足时必须说「无法回答」，禁止用模型自身的先验知识补全。
     RAG 系统的可信度建立在「答案有据可查」上，宁可拒答也不能编。
   * 宽松：允许模型结合常识补充，但要区分标注哪部分是资料内的。
   评估模块会对比两种模式的忠实度（faithfulness），是论文的一个实验点。

3. **历史对话只保留最近若干轮**
   7B 模型的上下文有限，把全部历史塞进去会挤掉检索到的资料 —— 那是本末倒置。
   追问所需的信息由「查询改写」环节压缩成一句话，不需要原始历史。
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.llm import ChatMessage
from app.core.retrieval.types import RetrievedChunk

SYSTEM_PROMPT_STRICT = """你是一个知识库问答助手，帮用户从本地资料里找到答案。回答要准确、自然、有用。

有两条必须守住的底线：

1. **只用资料里的内容。** 不要用常识、经验或你自己的知识补充任何东西 ——
   包括「一般来说……」「可以推测……」「可能包括……」这类听起来谨慎的说法，
   它们同样是编造。资料里没有的，就直接说「资料中未提及」。
2. **每个结论都要带来源编号**，格式 [1]，依据多条时写 [1][3]。
   分点陈述时**每一点都要带**。但不要把编号堆在末尾。

怎么算答得好：

- 问题宽泛时（比如「这个资料讲了什么」）**不要拒答** —— 把资料里能看出的主题、
  涉及的对象和大致内容概括出来，这就是合格的回答。
- 资料只覆盖了一部分时，把覆盖到的那部分讲清楚，再说明缺什么。
- 只有资料确实与问题完全无关时，才回答「根据现有资料无法回答该问题」。
- 用中文，像跟人说话一样自然，不要写成公文；内容多就分点，内容少就一两句说完。
- 不要复述题目，不要描述思考过程，也不要用「根据参考资料……」这类套话开头。"""

SYSTEM_PROMPT_LOOSE = """你是一个知识渊博的问答助手，当前正在使用一个本地知识库检索系统。
回答要准确、自然、有用。

【依据】
- 优先使用下面提供的【参考资料】作答，并标注来源编号（格式 [1]，多条写成 [1][3]）。
- 资料不足时可以结合你自己的知识补充，但要明确区分：
  来自资料的结论带编号；你自己的补充另起一段，以「补充：」开头。

【引用】
- 只标你实际依据的那几条，标在对应那句话后面，不要堆在末尾。
- 资料里没有的关键信息，直接说明「资料中未提及」，不要用推测填空。

【表达】
- 用中文，像跟人说话一样自然，不要写成公文。
- 先直接回答，再补充细节；内容多就分点，内容少就一两句说完。
- 不要复述题目，不要描述你的思考过程。"""

NO_CONTEXT_ANSWER = (
    "根据现有资料无法回答该问题。\n\n"
    "知识库中没有检索到与该问题相关的内容。你可以尝试：\n"
    "- 换一种更具体的问法，或补充关键术语；\n"
    "- 确认相关文档已经上传并完成索引；\n"
    "- 或切换到其他知识库再试。"
)

_EMPTY_KB_ANSWER = (
    "当前知识库里还没有任何已索引的文档，因此无法回答问题。\n\n"
    "请先在左侧上传文档，等索引状态变成「就绪」后再提问。"
)


def format_citation_label(chunk: RetrievedChunk, index: int) -> str:
    """资料块标题行，例如：`[1] 《论文.pdf》 第 3 页`。"""
    parts = [f"[{index}]", f"《{chunk.filename}》"]
    if chunk.page:
        parts.append(f"第 {chunk.page} 页")
    return " ".join(parts)


def build_context_block(chunks: Sequence[RetrievedChunk]) -> str:
    """把检索结果拼成带编号的参考资料块。"""
    if not chunks:
        return ""
    blocks: list[str] = ["【参考资料】"]
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"{format_citation_label(chunk, i)}\n{chunk.content.strip()}")
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    history: Sequence[ChatMessage] = (),
    strict: bool = True,
) -> list[ChatMessage]:
    """组装最终发给模型的完整消息列表。"""
    system = SYSTEM_PROMPT_STRICT if strict else SYSTEM_PROMPT_LOOSE

    user_parts: list[str] = []
    if chunks:
        user_parts.append(build_context_block(chunks))
        user_parts.append(f"【问题】\n{question}")
    else:
        user_parts.append(f"【问题】\n{question}")

    messages: list[ChatMessage] = [ChatMessage(role="system", content=system)]
    messages.extend(history)
    messages.append(ChatMessage(role="user", content="\n\n".join(user_parts)))
    return messages


__all__ = [
    "NO_CONTEXT_ANSWER",
    "SYSTEM_PROMPT_LOOSE",
    "SYSTEM_PROMPT_STRICT",
    "build_context_block",
    "build_messages",
    "format_citation_label",
    "_EMPTY_KB_ANSWER",
]
