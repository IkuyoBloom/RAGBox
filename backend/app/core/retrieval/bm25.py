"""BM25 稀疏检索。

中文场景的两个坑，这里都处理了：
  1. **分词**：BM25 建立在词袋上，中文没有空格，必须先分词。
     用 jieba 的搜索引擎模式（lcut_for_search），把长词切出子词，
     提升「梯度下降」这类专业术语的子串召回。
  2. **语料级统计**：BM25 的 IDF 依赖整个知识库的文档频率，
     所以索引是「按知识库建、按知识库缓存」的，不能跨库共用。
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

from app.core.retrieval.types import RetrievedChunk  # noqa: F401  (类型参考)

logger = logging.getLogger(__name__)

# 纯中文片段：整段保留为**一个词**。
# 注意不要写成 [\u4e00-\u9fff] 再用 finditer —— 那会把 jieba 切好的
# 「梯度下降法」逐字拆成 梯/度/下/降/法，等于白分词还引入大量单字噪声。
_CJK_RUN = re.compile(r"^[\u4e00-\u9fff]+$")
# 字母数字串（保留 2.5、qwen-2.5 这类带连接符的写法）
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]+(?:[.\-_][A-Za-z0-9]+)*")
# 检索里没有信息量的中文停用词（只列高频虚词，不做过度裁剪）
_STOPWORDS = frozenset(
    "的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 那 之 与 及 或 而 但 被 把 让 从 对 为 以 于 中 后 前 里 时 个 吗 呢 吧 啊".split()
)

_jieba_lock = threading.Lock()
_jieba_ready = False


def _ensure_jieba() -> Any:
    """惰性初始化 jieba 词典。

    jieba 首次 import 会加载词典（约 1~2 秒），放在模块顶层会拖慢进程启动，
    也会让 `--reload` 变得很烦。这里按需加载。
    """
    global _jieba_ready
    import jieba

    with _jieba_lock:
        if not _jieba_ready:
            jieba.initialize()
            _jieba_ready = True
    return jieba


def tokenize(text: str) -> list[str]:
    """把文本切成 BM25 用的词元序列。

    用 jieba 的搜索引擎模式（lcut_for_search）：它会在长词之外额外吐出子词
    （「神经网络」-> 神经 / 网络 / 神经网络），让「网络的优化」这类查询
    也能命中「神经网络」所在的文档。索引与查询两侧用同一套切法，
    保证词表一致。
    """
    if not text:
        return []
    jieba = _ensure_jieba()
    tokens: list[str] = []
    for piece in jieba.lcut_for_search(text):
        piece = piece.strip()
        if not piece:
            continue
        if _CJK_RUN.match(piece):
            if piece not in _STOPWORDS:
                tokens.append(piece)
            continue
        # 中英混排或纯英文数字片段：只抽字母数字串，丢掉标点
        for match in _ALNUM_RUN.finditer(piece):
            tok = match.group(0).lower()
            if tok not in _STOPWORDS:
                tokens.append(tok)
    return tokens


@dataclass(slots=True)
class BM25Hit:
    chunk_id: str
    score: float
    rank: int  # 1-based


class BM25Index:
    """某个知识库的 BM25 索引（纯内存）。"""

    def __init__(self, chunk_ids: list[str], tokenized_corpus: list[list[str]]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunk_ids = chunk_ids
        # 空语料必须单独处理：BM25Okapi 内部算 avgdl = num_doc / corpus_size，
        # corpus_size 为 0 时直接 ZeroDivisionError。
        # 这不是理论边界 —— 「新建知识库还没上传文档」和「文档刚被全部删除」
        # 都是界面上真实可达的状态，构造索引时崩溃会让整个检索接口 500。
        if not chunk_ids:
            self._bm25 = None
            return
        # 全是停用词/空文档时 avgdl 也会是 0，塞个零宽占位符把长度顶起来
        safe_corpus = [toks if toks else ["\u200b"] for toks in tokenized_corpus]
        self._bm25 = BM25Okapi(safe_corpus)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        if top_k <= 0 or not self.chunk_ids or self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)

        # 只取正分：BM25Okapi 会给未命中的文档返回 0 或负值，
        # 保留它们等于把噪声塞进 RRF 融合，反而稀释了真正的命中。
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[BM25Hit] = []
        for i in order:
            if len(hits) >= top_k:
                break
            if scores[i] <= 0:
                break
            hits.append(BM25Hit(chunk_id=self.chunk_ids[i], score=float(scores[i]), rank=len(hits) + 1))
        return hits


class BM25Registry:
    """索引缓存：按知识库维度缓存，写操作后显式失效。

    缓存签名用 (chunk 数, 文档数) 双指标 —— 只数 chunk 会漏掉
    「删一份文档再加一份等量文档」这种情况。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[tuple[int, int], BM25Index]] = {}

    def get(self, kb_id: str, signature: tuple[int, int]) -> BM25Index | None:
        with self._lock:
            entry = self._cache.get(kb_id)
            if entry and entry[0] == signature:
                return entry[1]
        return None

    def put(self, kb_id: str, signature: tuple[int, int], index: BM25Index) -> None:
        with self._lock:
            self._cache[kb_id] = (signature, index)

    def invalidate(self, kb_id: str) -> None:
        with self._lock:
            self._cache.pop(kb_id, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


registry = BM25Registry()


__all__ = ["BM25Hit", "BM25Index", "BM25Registry", "registry", "tokenize"]
