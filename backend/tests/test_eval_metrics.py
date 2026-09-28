"""评估指标单元测试。

覆盖目标：
  1. **每个纯函数的公式正确性** —— 用能笔算的小例子钉死数值，
     而不是"跑起来不报错就算过"；
  2. **边界情况** —— 空输入、除零、k 大于结果数、k<=0、gold 为空、
     完全无重叠、重复 id。这些正是论文指标最容易被悄悄算错的地方；
  3. **优雅降级** —— LLM 裁判不可用时必须返回 None，不能抛异常、
     也不能伪造成 0 分。

**全部测试都不需要 Ollama、不需要网络、不需要 ChromaDB**：
裁判用假 provider，数据库用内存 SQLite。这样在答辩现场的笔记本上
（没网、没拉模型）也能一键跑出全绿，作为"代码质量"的现场证据。
"""

from __future__ import annotations

import math

import pytest

from app.core.eval.metrics import (
    _normalize_tokens,
    aggregate_metrics,
    answer_context_overlap,
    char_ngram_overlap,
    citation_coverage,
    citation_validity,
    extract_citation_numbers,
    hit_rate_at_k,
    judge_answer_relevance,
    judge_faithfulness,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    split_sentences,
    token_f1,
)

# ======================================================================
# 测试替身
# ======================================================================
class _FakeResult:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeProvider:
    """假 LLM 通道：按预设脚本回话，用完就循环最后一条。"""

    name = "fake"
    model = "fake-1"

    def __init__(self, replies: list[str] | None = None, *, raises: Exception | None = None) -> None:
        self.replies = list(replies or [])
        self.raises = raises
        self.calls: list[tuple] = []

    async def chat(self, messages, *, temperature=None, max_tokens=None):  # noqa: ANN001
        self.calls.append((tuple(m.role for m in messages), temperature, max_tokens))
        if self.raises is not None:
            raise self.raises
        if not self.replies:
            return _FakeResult("")
        return _FakeResult(self.replies.pop(0) if len(self.replies) > 1 else self.replies[0])


# ======================================================================
# hit_rate_at_k
# ======================================================================
def test_hit_rate_基础命中与未命中() -> None:
    assert hit_rate_at_k(["a", "b", "c"], ["b"], 3) == 1.0
    assert hit_rate_at_k(["a", "b", "c"], ["z"], 3) == 0.0


def test_hit_rate_k_截断() -> None:
    """gold 排在第 3 位，K=2 时看不到 —— 这正是 K 的意义所在。"""
    assert hit_rate_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert hit_rate_at_k(["a", "b", "c"], ["c"], 3) == 1.0


def test_hit_rate_gold_为空返回_none() -> None:
    """无标准答案 -> 指标无定义。返回 0 会凭空拉低整体均值。"""
    assert hit_rate_at_k(["a"], [], 3) is None
    assert hit_rate_at_k(["a"], set(), 3) is None


def test_hit_rate_k_非正返回_none() -> None:
    """K=0 表示"没有可评估的检索深度"，不是"全部未命中"。"""
    assert hit_rate_at_k(["a"], ["a"], 0) is None
    assert hit_rate_at_k(["a"], ["a"], -1) is None


def test_hit_rate_空检索结果() -> None:
    assert hit_rate_at_k([], ["a"], 5) == 0.0


def test_hit_rate_重复_id_不虚高() -> None:
    """去重保留最高名次：重复项不该把后面的相关结果挤出 Top-K。"""
    assert hit_rate_at_k(["a", "a", "a"], ["a"], 1) == 1.0
    # 去重后是 [x, a]，取 Top-2 能看到 a；只取 Top-1 则看不到
    assert hit_rate_at_k(["x", "x", "a"], ["a"], 2) == 1.0
    assert hit_rate_at_k(["x", "x", "a"], ["a"], 1) == 0.0


# ======================================================================
# recall_at_k
# ======================================================================
def test_recall_部分召回() -> None:
    # gold 两条，Top-3 只捞回一条 -> 0.5
    assert recall_at_k(["a", "b", "c"], ["b", "d"], 3) == pytest.approx(0.5)


def test_recall_全部召回() -> None:
    assert recall_at_k(["a", "b"], ["a", "b"], 6) == 1.0


def test_recall_k_大于结果数按实际长度算() -> None:
    """K=10 但只检索到 2 条，分母仍是 |gold|，不做补零。"""
    assert recall_at_k(["a", "b"], ["a", "b", "z"], 10) == pytest.approx(2 / 3)


def test_recall_除零与空输入() -> None:
    assert recall_at_k([], [], 5) is None
    assert recall_at_k(["a"], [], 5) is None
    assert recall_at_k(["a"], ["a"], 0) is None
    # gold 非空但检索为空 -> 真实含义是 0 召回
    assert recall_at_k([], ["a"], 5) == 0.0


def test_recall_完全无重叠() -> None:
    assert recall_at_k(["x", "y"], ["a", "b"], 2) == 0.0


# ======================================================================
# precision_at_k
# ======================================================================
def test_precision_分母用实际检索条数() -> None:
    """只检索到 2 条时，分母是 2 而不是 K=6。"""
    assert precision_at_k(["a", "x"], ["a"], 6) == pytest.approx(0.5)


def test_precision_全对与全错() -> None:
    assert precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert precision_at_k(["x", "y"], ["a"], 2) == 0.0


def test_precision_gold_为空返回_none_检索为空返回_0() -> None:
    assert precision_at_k(["a"], [], 3) is None
    assert precision_at_k([], ["a"], 3) == 0.0


def test_precision_k_非正返回_none() -> None:
    assert precision_at_k(["a"], ["a"], 0) is None


# ======================================================================
# mrr
# ======================================================================
def test_mrr_首位命中得满分() -> None:
    assert mrr(["a", "b"], ["a"]) == 1.0


def test_mrr_第二位命中得二分之一() -> None:
    assert mrr(["x", "a"], ["a"]) == pytest.approx(0.5)


def test_mrr_第三位命中得三分之一() -> None:
    assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_mrr_只看第一条相关结果() -> None:
    """多个 gold 时，MRR 取**最早出现**的那个，不是平均值。"""
    assert mrr(["a", "b"], ["a", "b"]) == 1.0
    assert mrr(["x", "b", "a"], ["a", "b"]) == pytest.approx(0.5)


def test_mrr_未命中与空输入() -> None:
    assert mrr(["x", "y"], ["a"]) == 0.0
    assert mrr([], ["a"]) == 0.0
    assert mrr([], []) is None
    assert mrr(["a"], []) is None


def test_mrr_不受_k_影响() -> None:
    """MRR 没有 K 参数：正确答案排在第 20 位也是有效信息。"""
    retrieved = [f"c{i}" for i in range(19)] + ["a"]
    assert mrr(retrieved, ["a"]) == pytest.approx(1 / 20)


# ======================================================================
# ndcg_at_k
# ======================================================================
def test_ndcg_理想排序得满分() -> None:
    assert ndcg_at_k(["a", "b", "c"], ["a"], 3) == pytest.approx(1.0)


def test_ndcg_单条_gold_排第三() -> None:
    """IDCG = 1/log2(2) = 1；DCG = 1/log2(4) = 0.5 -> NDCG = 0.5。"""
    assert ndcg_at_k(["x", "y", "a"], ["a"], 3) == pytest.approx(0.5)


def test_ndcg_多条_gold_的折损() -> None:
    """gold={a,b}：理想是 a,b 占据前两位，实际 b 在第二位、a 在第三位。"""
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["x", "b", "a"], ["a", "b"], 3) == pytest.approx(dcg / idcg)


def test_ndcg_理想排序等于一() -> None:
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 3) == pytest.approx(1.0)


def test_ndcg_gold_多于_k_时_idcg_按_k_截断() -> None:
    """gold 有 4 条但 K=2：IDCG 只累计 2 条，前两位命中即满分。

    否则"检索深度不够"会被错算成"排序质量差"。
    """
    assert ndcg_at_k(["a", "b"], ["a", "b", "c", "d"], 2) == pytest.approx(1.0)


def test_ndcg_k_大于结果数与完全无重叠() -> None:
    # K 大于结果数时按实际长度算，仍然有定义
    assert ndcg_at_k(["a"], ["a"], 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], ["a"], 2) == 0.0


def test_ndcg_空_gold_与非法_k_返回_none() -> None:
    assert ndcg_at_k(["a"], [], 3) is None
    assert ndcg_at_k(["a"], ["a"], 0) is None
    assert ndcg_at_k([], ["a"], 3) == 0.0


# ======================================================================
# 引用相关：句子切分与编号抽取
# ======================================================================
def test_split_sentences_中文标点与换行() -> None:
    parts = split_sentences("第一句。第二句！第三句？\n第四句；第五句")
    assert [p.strip() for p in parts] == ["第一句", "第二句", "第三句", "第四句", "第五句"]


def test_split_sentences_空输入() -> None:
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_extract_citation_numbers_多种写法() -> None:
    assert extract_citation_numbers("见 [1] 与 [3]") == [1, 3]
    assert extract_citation_numbers("参考[1][2]") == [1, 2]
    assert extract_citation_numbers("[1,3]") == [1, 3]
    assert extract_citation_numbers("[1-3]") == [1, 2, 3]
    assert extract_citation_numbers("【5】") == [5]
    assert extract_citation_numbers("【5】[2]") == [5, 2]


def test_extract_citation_numbers_畸形区间不炸() -> None:
    """[1-99999] 这类畸形输出必须被限制，否则会瞬间造出上万个数字。"""
    nums = extract_citation_numbers("[1-99999]")
    assert 0 < len(nums) <= 100
    assert nums[0] == 1


def test_extract_citation_numbers_空与无引用() -> None:
    assert extract_citation_numbers("") == []
    assert extract_citation_numbers("没有任何角标的句子。") == []


# ======================================================================
# citation_coverage
# ======================================================================
def test_citation_coverage_全部带角标() -> None:
    answer = "过拟合是过度拟合训练数据[1]。缓解方法是正则化[2]。"
    assert citation_coverage(answer) == 1.0


def test_citation_coverage_一半带角标() -> None:
    answer = "第一个结论有来源[1]。第二个结论没有来源。"
    assert citation_coverage(answer) == pytest.approx(0.5)


def test_citation_coverage_空回答返回_none() -> None:
    assert citation_coverage("") is None
    assert citation_coverage("   ") is None


def test_citation_coverage_纯标点无实义句子返回_none() -> None:
    """「。。。」没有实义内容，覆盖率无定义，不能算 0。"""
    assert citation_coverage("。。。") is None


def test_citation_coverage_citations_参数不影响结果() -> None:
    """覆盖率只看角标有无；"标得对不对"由 citation_validity 负责。"""
    answer = "结论[1]。"
    assert citation_coverage(answer, [{"index": 1}]) == citation_coverage(answer, None) == 1.0
    assert citation_coverage(answer, []) == 1.0


# ======================================================================
# citation_validity
# ======================================================================
def test_citation_validity_全部越界() -> None:
    """只喂了 2 条资料却引用 [7] -> 模型在编造来源。"""
    assert citation_validity("结论[7]", 2) == 0.0


def test_citation_validity_部分越界() -> None:
    assert citation_validity("结论一[1]，结论二[7]，结论三[3]", 3) == pytest.approx(2 / 3)


def test_citation_validity_全部合法() -> None:
    assert citation_validity("见 [1][2][3]", 3) == 1.0


def test_citation_validity_无角标返回_1() -> None:
    """没有声称任何来源 = 没有编造行为；该不该标由 coverage 管。"""
    assert citation_validity("这句没有任何角标。", 3) == 1.0


def test_citation_validity_零条资料却引用() -> None:
    assert citation_validity("结论[1]", 0) == 0.0
    assert citation_validity("结论[1]", -1) == 0.0


def test_citation_validity_编号从_0_开始算越界() -> None:
    """角标从 1 开始，[0] 是非法编号。"""
    assert citation_validity("[0]", 3) == 0.0


# ======================================================================
# token_f1
# ======================================================================
def test_token_f1_完全相同得满分() -> None:
    assert token_f1("梯度下降用于优化神经网络", "梯度下降用于优化神经网络") == 1.0


def test_token_f1_完全无重叠得零分() -> None:
    assert token_f1("今天天气很好", "梯度下降优化神经网络参数") == 0.0


def test_token_f1_部分重叠落在区间内() -> None:
    score = token_f1("过拟合是训练集表现好但泛化差", "过拟合指模型在训练集表现好、在新数据上泛化差")
    assert score is not None
    assert 0.5 < score < 1.0


def test_token_f1_重复关键词不能刷分() -> None:
    """用 Counter 的多重集交集：把关键词重复十遍不该提高相似度。"""
    gold = "梯度下降"
    spam = "梯度下降 " * 10
    score = token_f1(spam, gold)
    assert score is not None
    assert score < 0.5


def test_token_f1_空输入语义() -> None:
    assert token_f1("", "") is None  # 没有可比内容
    assert token_f1("   ", "") is None
    assert token_f1("", "梯度下降") == 0.0  # 一侧为空 = 完全说错
    assert token_f1("梯度下降", "") == 0.0


def test_token_f1_停用词被忽略() -> None:
    """只差虚词时应得满分，说明停用词确实被过滤掉了。

    注：这里刻意选「的 / 了」这类纯虚词。中文分词存在边界歧义
    （「集上的」jieba 会切成「集上 / 的」），拿长句测会把分词差异
    误当成停用词过滤失败，那样的用例是在测 jieba 而不是测本模块。
    """
    assert token_f1("模型的训练效果", "模型训练效果") == 1.0
    assert token_f1("神经网络的优化了", "神经网络优化") == 1.0


def test_token_f1_不丢弃否定词() -> None:
    """「不」「没有」必须保留：否则「有监督」与「没有监督」会判为高度相似。"""
    assert token_f1("有监督学习需要标签", "没有监督学习需要标签") < 1.0
    tokens = _normalize_tokens("模型不收敛")
    assert "不" in tokens


# ======================================================================
# char_ngram_overlap
# ======================================================================
def test_char_ngram_完全相同得满分() -> None:
    assert char_ngram_overlap("机器学习算法", "机器学习算法") == 1.0


def test_char_ngram_完全无重叠() -> None:
    assert char_ngram_overlap("今天天气不错", "梯度下降优化") == 0.0


def test_char_ngram_忽略空白() -> None:
    """中文里的空格只是排版产物，不该影响相似度。"""
    assert char_ngram_overlap("机器 学习\n算法", "机器学习算法") == 1.0

def test_char_ngram_部分重叠() -> None:
    score = char_ngram_overlap("机器学习算法", "机器学习")
    assert score is not None
    assert 0.0 < score < 1.0


def test_char_ngram_空输入与非法_n() -> None:
    assert char_ngram_overlap("", "") is None
    assert char_ngram_overlap("", "机器学习") == 0.0
    assert char_ngram_overlap("机器学习", "") == 0.0
    assert char_ngram_overlap("机器学习", "机器学习", n=0) == 0.0


def test_char_ngram_短于_n_的文本() -> None:
    """文本短于 n 时抽不出 gram，视为空。"""
    assert char_ngram_overlap("机", "机", n=2) is None
    assert char_ngram_overlap("机", "机器学习", n=2) == 0.0


def test_char_ngram_n_越大越严格() -> None:
    bigram = char_ngram_overlap("机器学习算法", "机器学习模型", n=2)
    trigram = char_ngram_overlap("机器学习算法", "机器学习模型", n=3)
    assert bigram is not None and trigram is not None
    assert bigram >= trigram


# ======================================================================
# answer_context_overlap
# ======================================================================
def test_answer_context_overlap_答案照抄资料得高分() -> None:
    context = (
        "过拟合是指模型在训练数据上表现很好，但在未见过的测试数据上表现较差的现象。"
        "缓解过拟合的常用方法包括正则化、Dropout、早停、数据增强。"
    )
    answer = "过拟合是模型在训练数据上表现很好，但在未见过的测试数据上表现较差。缓解方法包括正则化、Dropout、早停。"
    score = answer_context_overlap(answer, [context])
    assert score is not None
    assert score > 0.5


def test_answer_context_overlap_答案与资料无关得零分() -> None:
    score = answer_context_overlap("今天天气很好适合散步", ["过拟合是模型过度拟合训练数据的现象"])
    assert score == 0.0


def test_answer_context_overlap_拼接多条资料() -> None:
    contexts = ["过拟合是模型过度拟合训练数据", "正则化可以缓解过拟合"]
    answer = "过拟合是模型过度拟合训练数据，可以通过正则化缓解"
    score = answer_context_overlap(answer, contexts)
    assert score is not None
    assert score > 0.5


def test_answer_context_overlap_空输入() -> None:
    assert answer_context_overlap("", ["资料"]) is None
    assert answer_context_overlap("   ", ["资料"]) is None
    assert answer_context_overlap("回答", []) == 0.0
    assert answer_context_overlap("回答", [""]) == 0.0
    assert answer_context_overlap("", []) is None


# ======================================================================
# aggregate_metrics
# ======================================================================
def test_aggregate_跳过_none_并报告_n() -> None:
    """None 不参与均值，但 n 必须如实反映有效样本数。"""
    out = aggregate_metrics({"recall": [1.0, 0.0, None, 0.5]})
    assert out["recall"]["mean"] == pytest.approx(0.5)
    assert out["recall"]["n"] == 3
    assert out["recall"]["min"] == 0.0
    assert out["recall"]["max"] == 1.0


def test_aggregate_全为_none_返回空块() -> None:
    out = aggregate_metrics({"mrr": [None, None]})
    assert out["mrr"] == {"mean": None, "std": None, "min": None, "max": None, "n": 0}


def test_aggregate_空输入序列() -> None:
    out = aggregate_metrics({"hit_rate": []})
    assert out["hit_rate"]["n"] == 0
    assert out["hit_rate"]["mean"] is None


def test_aggregate_单样本标准差为_0() -> None:
    """只有一个有效样本时波动无法估计，用 0 表示并靠 n=1 提醒读者。"""
    out = aggregate_metrics({"recall": [0.7]})
    assert out["recall"]["std"] == 0.0
    assert out["recall"]["n"] == 1


def test_aggregate_标准差为样本标准差() -> None:
    out = aggregate_metrics({"x": [1.0, 2.0, 3.0]})
    assert out["x"]["std"] == pytest.approx(1.0)  # n-1 口径


# ======================================================================
# LLM 裁判：解析与降级
# ======================================================================
@pytest.mark.asyncio
async def test_judge_relevance_正常解析() -> None:
    provider = _FakeProvider(["评分：4\n理由：回答基本切题，略有遗漏"])
    result = await judge_answer_relevance(provider, "什么是过拟合？", "过拟合是模型过度拟合训练数据。")
    assert result is not None
    score, reason = result
    assert score == 4.0
    assert "略有遗漏" in reason


@pytest.mark.asyncio
async def test_judge_relevance_全角冒号与小数() -> None:
    provider = _FakeProvider(["评分: 3.5\n理由：部分正确"])
    result = await judge_answer_relevance(provider, "问题", "回答")
    assert result is not None
    assert result[0] == 3.5


@pytest.mark.asyncio
async def test_judge_relevance_无标签短输出兜底() -> None:
    provider = _FakeProvider(["5"])
    result = await judge_answer_relevance(provider, "问题", "回答")
    assert result is not None
    assert result[0] == 5.0


@pytest.mark.asyncio
async def test_judge_relevance_分数越界被夹紧() -> None:
    high = await judge_answer_relevance(_FakeProvider(["评分：9"]), "q", "a")
    low = await judge_answer_relevance(_FakeProvider(["评分：0"]), "q", "a")
    assert high is not None and high[0] == 5.0
    assert low is not None and low[0] == 1.0


@pytest.mark.asyncio
async def test_judge_relevance_无法解析返回_none() -> None:
    """关键降级：解析不出来必须返回 None，不能猜一个分数。"""
    provider = _FakeProvider(["我不知道该怎么评分，这个问题太模糊了，需要更多信息才能判断。"])
    assert await judge_answer_relevance(provider, "问题", "回答") is None


@pytest.mark.asyncio
async def test_judge_relevance_空输出与异常都不抛() -> None:
    assert await judge_answer_relevance(_FakeProvider([""]), "问题", "回答") is None
    assert await judge_answer_relevance(_FakeProvider(raises=RuntimeError("ollama 未启动")), "q", "a") is None


@pytest.mark.asyncio
async def test_judge_relevance_空问题或空回答返回_none() -> None:
    provider = _FakeProvider(["评分：5"])
    assert await judge_answer_relevance(provider, "", "回答") is None
    assert await judge_answer_relevance(provider, "问题", "   ") is None
    # 没有真正调用模型：该走的降级路径不该浪费一次推理
    assert provider.calls == []


@pytest.mark.asyncio
async def test_judge_faithfulness_正常解析() -> None:
    provider = _FakeProvider(["评分：0.8\n理由：主要论断有依据，细节为补充"])
    result = await judge_faithfulness(provider, "过拟合需要正则化缓解", ["过拟合可以通过正则化缓解"])
    assert result is not None
    score, reason = result
    assert score == pytest.approx(0.8)
    assert "依据" in reason


@pytest.mark.asyncio
async def test_judge_faithfulness_越界夹紧与无法解析() -> None:
    clamped = await judge_faithfulness(_FakeProvider(["评分：7"]), "a", ["c"])
    assert clamped is not None and clamped[0] == 1.0
    assert await judge_faithfulness(_FakeProvider(["说不清楚"]), "a", ["c"]) is None
    assert await judge_faithfulness(_FakeProvider([""]), "a", ["c"]) is None


@pytest.mark.asyncio
async def test_judge_faithfulness_空输入与异常降级() -> None:
    provider = _FakeProvider(["评分：1.0"])
    assert await judge_faithfulness(provider, "", ["资料"]) is None
    assert await judge_faithfulness(provider, "回答", []) is None
    assert provider.calls == []
    assert await judge_faithfulness(_FakeProvider(raises=TimeoutError("超时")), "a", ["c"]) is None
