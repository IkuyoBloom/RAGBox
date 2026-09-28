"""核心检索层回归测试。

这里钉住的每一条都对应一个**真实出现过的 bug**，不是凑覆盖率：

1. `tokenize` 曾把 jieba 切好的「梯度下降法」逐字拆成「梯/度/下/降/法」，
   等于分词白做还引入大量单字噪声。根因是过滤正则写成了
   `[\\u4e00-\\u9fff]`（只匹配单个汉字）配 `finditer`。
2. `split_text` 曾在段落层面直接成片，导致由大量短段落组成的文档
   切出几十个十几字的碎块，Top-K 全被碎片占满。
3. `normalize_text` 必须把 PDF 的视觉硬换行合并回句子，
   否则切片到处断句。

BM25 / RRF / 切分都不依赖 Ollama，本文件可在无模型环境下运行。
"""

from __future__ import annotations

import pytest

from app.core.ingest.loaders import normalize_text
from app.core.ingest.splitter import split_text
from app.core.retrieval.bm25 import BM25Index, BM25Registry, tokenize
from app.core.retrieval.hybrid import rrf_fuse

# ======================================================================
# 中文分词
# ======================================================================
class TestTokenize:
    def test_保留多字词而非拆成单字(self):
        """回归：曾把「梯度下降法」拆成 梯/度/下/降/法。"""
        tokens = tokenize("梯度下降法是训练神经网络最常用的优化算法之一。")
        multi = [t for t in tokens if len(t) > 1 and not t.isascii()]
        assert multi, f"分词结果全是单字，说明又被逐字拆开了：{tokens}"
        # jieba 搜索引擎模式会同时给出「神经」「网络」「神经网络」
        assert "梯度" in tokens or "梯度下降" in tokens
        assert "神经网络" in tokens or "神经" in tokens

    def test_不产生孤立的单字噪声(self):
        """纯中文文本不应产出大量单字 token（少量专有单字允许，但不应占多数）。"""
        tokens = tokenize("检索增强生成通过在生成前引入外部知识来缓解幻觉问题")
        singles = [t for t in tokens if len(t) == 1 and not t.isascii()]
        assert len(singles) <= len(tokens) * 0.5, f"单字占比过高：{tokens}"

    def test_停用词被过滤(self):
        tokens = tokenize("这个 是 一个 的 测试")
        for stop in ("的", "是", "一个"):
            assert stop not in tokens

    def test_英文数字与带连接符的写法保留(self):
        tokens = tokenize("使用 qwen2.5 与 text-embedding 模型")
        joined = " ".join(tokens)
        assert "qwen2.5" in joined or "qwen" in joined
        assert "text" in joined

    def test_标点不进入结果(self):
        tokens = tokenize("什么是 BM25？？？、、、！！！")
        assert not any(t in "？、！，。；：" for t in tokens)

    def test_空输入(self):
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_同一文本两次调用结果一致(self):
        text = "混合检索融合了稀疏与稠密两路结果"
        assert tokenize(text) == tokenize(text)


# ======================================================================
# BM25
# ======================================================================
class TestBM25Index:
    CORPUS = [
        "梯度下降法用于优化神经网络的损失函数",
        "今天天气很好适合出门散步",
        "神经网络的反向传播算法推导过程",
        "向量数据库支持近似最近邻检索",
    ]

    def _index(self) -> BM25Index:
        return BM25Index(
            [f"c{i}" for i in range(len(self.CORPUS))],
            [tokenize(t) for t in self.CORPUS],
        )

    def test_相关文档排在无关文档之前(self):
        hits = self._index().search("神经网络优化", top_k=4)
        assert hits, "应当至少命中一条"
        assert hits[0].chunk_id in ("c0", "c2")

    def test_完全无关的查询返回空(self):
        hits = self._index().search("量子纠缠与拓扑绝缘体", top_k=4)
        assert hits == [], f"无关查询不应有命中，却返回了 {hits}"

    def test_rank_从1开始且连续(self):
        hits = self._index().search("神经网络", top_k=4)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))

    def test_top_k_截断(self):
        hits = self._index().search("神经网络", top_k=1)
        assert len(hits) <= 1

    def test_top_k_非正数返回空(self):
        index = self._index()
        assert index.search("神经网络", top_k=0) == []
        assert index.search("神经网络", top_k=-3) == []

    def test_分数降序(self):
        hits = self._index().search("神经网络算法", top_k=4)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_空语料不崩(self):
        """回归：rank_bm25 在空语料上 avgdl 除零，构造索引即抛 ZeroDivisionError。"""
        index = BM25Index([], [])
        assert index.search("任意查询", top_k=5) == []
        assert len(index) == 0

    def test_chunk_ids_为空但语料非空也不崩(self):
        index = BM25Index([], [[]])
        assert index.search("测试", top_k=2) == []

    def test_全停用词文档不导致除零(self):
        """全是停用词/空白的文档会让 avgdl=0，构造时必须不抛异常。"""
        index = BM25Index(["a", "b"], [[], []])
        assert index.search("测试", top_k=2) == []

    def test_查询全是停用词返回空(self):
        assert self._index().search("的 了 和", top_k=4) == []


class TestBM25Registry:
    def test_同签名命中缓存(self):
        reg = BM25Registry()
        idx = BM25Index(["a"], [["x"]])
        reg.put("kb1", (1, 1), idx)
        assert reg.get("kb1", (1, 1)) is idx

    def test_签名变化视为失效(self):
        reg = BM25Registry()
        reg.put("kb1", (1, 1), BM25Index(["a"], [["x"]]))
        assert reg.get("kb1", (2, 1)) is None

    def test_invalidate_清除缓存(self):
        reg = BM25Registry()
        reg.put("kb1", (1, 1), BM25Index(["a"], [["x"]]))
        reg.invalidate("kb1")
        assert reg.get("kb1", (1, 1)) is None

    def test_不同知识库互不干扰(self):
        reg = BM25Registry()
        a = BM25Index(["a"], [["x"]])
        b = BM25Index(["b"], [["y"]])
        reg.put("kb1", (1, 1), a)
        reg.put("kb2", (1, 1), b)
        assert reg.get("kb1", (1, 1)) is a
        assert reg.get("kb2", (1, 1)) is b


# ======================================================================
# RRF 融合
# ======================================================================
class _Hit:
    """模拟 BM25Hit，避免为了构造测试数据去碰真实索引。"""

    __slots__ = ("chunk_id", "score", "rank")

    def __init__(self, chunk_id: str, rank: int, score: float = 1.0):
        self.chunk_id, self.rank, self.score = chunk_id, rank, score


class TestRRFFuse:
    def test_两路一致时该文档得分最高(self):
        dense = [("a", 0.9), ("b", 0.8)]
        sparse = [_Hit("a", 1), _Hit("b", 2)]
        fused = rrf_fuse(dense, sparse, k=60)
        assert fused["a"] > fused["b"]

    def test_双路命中优于单路命中(self):
        """RRF 的核心价值：被两路都认可的文档应该排在只被一路认可的之前。"""
        dense = [("only_dense", 0.99), ("both", 0.5)]
        sparse = [_Hit("both", 1)]
        fused = rrf_fuse(dense, sparse, k=60)
        assert fused["both"] > fused["only_dense"]

    def test_分数只依赖排名不依赖原始分数(self):
        dense_a = [("a", 0.99), ("b", 0.01)]
        dense_b = [("a", 0.51), ("b", 0.50)]
        assert rrf_fuse(dense_a, [], k=60) == rrf_fuse(dense_b, [], k=60)

    def test_k_越小头部优势越大(self):
        dense = [("a", 1.0), ("b", 0.9)]
        small_k = rrf_fuse(dense, [], k=1)
        large_k = rrf_fuse(dense, [], k=1000)
        assert (small_k["a"] - small_k["b"]) > (large_k["a"] - large_k["b"])

    def test_权重生效(self):
        dense = [("a", 1.0)]
        sparse = [_Hit("b", 1)]
        fused = rrf_fuse(dense, sparse, k=60, dense_weight=3.0, sparse_weight=1.0)
        assert fused["a"] > fused["b"]

    def test_两路都空返回空(self):
        assert rrf_fuse([], [], k=60) == {}

    def test_仅一路有结果也正常(self):
        assert "a" in rrf_fuse([("a", 1.0)], [], k=60)
        assert "b" in rrf_fuse([], [_Hit("b", 1)], k=60)

    def test_k_非法值被夹紧而不除零(self):
        fused = rrf_fuse([("a", 1.0)], [], k=0)
        assert fused["a"] > 0

    def test_结果只含出现过的文档(self):
        fused = rrf_fuse([("a", 1.0)], [_Hit("b", 1)], k=60)
        assert set(fused) == {"a", "b"}


# ======================================================================
# 文本归一化（PDF 硬换行）
# ======================================================================
class TestNormalizeText:
    def test_合并中文硬换行(self):
        """PDF 每个视觉行一个换行，必须合并回完整句子。"""
        raw = "梯度下降法是一种\n常用的优化算法"
        assert "梯度下降法是一种常用的优化算法" in normalize_text(raw)

    def test_句末标点处保留换行(self):
        raw = "第一句话结束了。\n第二句话开始了。"
        out = normalize_text(raw)
        assert "\n" in out
        assert "第一句话结束了。" in out
        assert "第二句话开始了。" in out

    def test_英文之间补空格(self):
        out = normalize_text("hello\nworld")
        assert "hello world" in out

    def test_中文之间不补空格(self):
        assert "中文之间" in normalize_text("中文\n之间")

    def test_连字符断词被修复(self):
        assert "optimization" in normalize_text("opti-\nmization")

    def test_压缩多余空行(self):
        out = normalize_text("段落一\n\n\n\n\n段落二")
        assert "\n\n\n" not in out

    def test_不间断空格被替换(self):
        assert "\u00a0" not in normalize_text("a\u00a0b")

    def test_空输入(self):
        assert normalize_text("") == ""
        assert normalize_text("   \n  ") == ""


# ======================================================================
# 切分
# ======================================================================
class TestSplitText:
    def test_短文本不切(self):
        text = "这是一段很短的文本。"
        assert split_text(text, chunk_size=500) == [text]

    def test_空输入返回空列表(self):
        assert split_text("", chunk_size=500) == []
        assert split_text("    ", chunk_size=500) == []

    def test_超长文本被切成多片(self):
        text = "。".join(f"这是第{i}个用于测试的句子内容" for i in range(60))
        chunks = split_text(text, chunk_size=200, chunk_overlap=30)
        assert len(chunks) > 1

    def test_每片不超过目标长度的合理上界(self):
        text = "。".join(f"第{i}句测试文本内容" for i in range(80))
        chunks = split_text(text, chunk_size=200, chunk_overlap=30)
        over = [c for c in chunks if len(c) > 200 * 2]
        assert not over, f"存在严重超长切片：{[len(c) for c in over]}"

    def test_短段落被打包而不是各占一片(self):
        """回归：曾因在段落层面直接成片，切出大量十几字的碎片。"""
        text = "\n\n".join(
            f"第{i}条：这是一个用于测试的短段落，长度不足以单独成片。" for i in range(1, 21)
        )
        chunks = split_text(text, chunk_size=200, chunk_overlap=30)
        avg = sum(len(c) for c in chunks) / len(chunks)
        assert avg > 120, f"平均片长仅 {avg:.0f}，短段落没有被有效打包"

    def test_重叠生效(self):
        text = "。".join(f"第{i}句足够长的测试句子用于触发切分逻辑" for i in range(40))
        no_overlap = split_text(text, chunk_size=150, chunk_overlap=0)
        with_overlap = split_text(text, chunk_size=150, chunk_overlap=50)
        assert len(with_overlap) >= len(no_overlap)

    def test_无标点长文本也能硬切(self):
        text = "甲" * 1000
        chunks = split_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(chunks)

    def test_overlap_大于等于_size_被夹紧不崩(self):
        text = "。".join(f"第{i}句测试内容" for i in range(30))
        chunks = split_text(text, chunk_size=100, chunk_overlap=999)
        assert chunks

    def test_切分结果无空白片(self):
        text = "。".join(f"第{i}句测试句子内容" for i in range(30))
        chunks = split_text(text, chunk_size=120, chunk_overlap=20)
        assert all(c.strip() for c in chunks)

    def test_内容不丢失大意(self):
        """切分是「切」不是「丢」：关键句子应出现在某一片里。"""
        key = "这一段包含必须被保留下来的关键结论"
        text = "填充内容。" * 40 + key + "。" + "更多填充内容。" * 40
        chunks = split_text(text, chunk_size=200, chunk_overlap=40)
        assert any(key in c for c in chunks), "关键句子在切分中丢失了"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
