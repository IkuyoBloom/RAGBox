"""检索与生成质量指标（全部为纯函数，可脱离数据库 / 模型单元测试）。

## 为什么指标要「自己实现」而不是调 ragas / deepeval

1. 「你这个召回率是怎么算的」一定会被问到。第三方库把公式封在黑盒里，
   答不上来；自己实现则每一行都能说清楚。
2. 那些库会拖进 langchain、openai、datasets 等一大堆依赖，与「完全本地部署」
   的题目设定冲突。
3. 指标本身不是难点，**边界情况的处理**才是。这里的每个函数都显式定义了
   空输入 / 除零 / k 越界时的返回语义，并用 pytest 钉死。

## 一条贯穿全模块的约定：`None` 不等于 0

没有标准答案（gold 为空）时，指标**无法定义**，返回 `None`；
有标准答案但一个都没命中，返回 `0.0`。

如果两种情况都返回 0，聚合均值就会被「不该被计入的样本」拉低 ——
例如某个样本没有 gold_chunk_ids，按 0 分计入会凭空把整体召回率砍掉一截。
因此聚合函数（`aggregate_metrics`）会跳过 `None`，并如实报告参与计算的样本数 `n`。

## 为什么中文相似度用「分词 F1 + 字符 2-gram」两把尺子

* `token_f1`：词级，贴合人类判读，但对分词错误敏感（「机器学习」被切成
  「机器/学习」时，与「机器学习算法」的重合度会虚低）。
* `char_ngram_overlap`：字符级，不依赖分词器，对中文改写、语序变化更稳，
  代价是会被常见字组合灌水。

两者一起看：词级高、字符级也高 -> 确实在复述资料；词级低而字符级高
-> 多半只是用了同样的字，语义并不重合。单一指标容易被「看起来很像」骗过去。
"""

from __future__ import annotations

import logging
import math
import re
import statistics
import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from app.core.eval._stopwords import ALL_STOPWORDS

logger = logging.getLogger(__name__)

# 论文表格里检索指标的列名与顺序。目前只在 `runner.summarize` 的渲染顺序里
# 隐含体现（那张表是手写的、列更全），这里保留一份机器可读的常量，
# 供将来做自动化出表脚本时引用，避免列名两处各写一份。
RETRIEVAL_METRIC_NAMES: tuple[str, ...] = (
    "hit_rate",
    "recall",
    "precision",
    "mrr",
    "ndcg",
)


# ======================================================================
# 检索指标
#
# 统一约定：
#   * retrieved_ids 按**相关性降序**排列（融合检索的输出天然满足）；
#   * 指标在 Top-K 上计算，K 不足时按实际长度算（不补零、不外推）；
#   * 重复 id 只算一次（保留最高名次），避免上游 bug 让分数虚高。
# ======================================================================
def _dedup_preserve_order(ids: Iterable[str]) -> list[str]:
    """去重但保留首次出现的名次。

    检索链路正常时不会出现重复 chunk，但「向量库残留 + 回表去重」这类
    上游问题确实会造出重复。评估指标不该因为上游的小毛病给出假的漂亮分数，
    所以在这里兜一道。
    """
    seen: set[str] = set()
    out: list[str] = []
    for cid in ids:
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def _topk(retrieved_ids: Sequence[str], k: int) -> list[str]:
    """取 Top-K 并归一化。

    k <= 0 表示「没有可评估的检索深度」，返回空列表，让各指标自然退化成
    「未命中」，而不是抛异常 —— 评估脚本不该因为一个手滑的参数整轮崩掉。
    """
    if k <= 0:
        return []
    return _dedup_preserve_order(retrieved_ids)[:k]


def hit_rate_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float | None:
    """Top-K 命中率（Hit Rate @ K）。

    公式：``HR@K = 1[ |TopK(retrieved) ∩ gold| > 0 ]``

    取值：``{0.0, 1.0}``；gold 为空或 k <= 0 时返回 ``None``。

    定义成 0/1 而不是「命中数 / K」是刻意的：HR@K 回答的是
    **「用户在前 K 条里能不能看到至少一条对的」**，是个**可用性**问题，
    与命中了几条无关。因此在多样本聚合后，HR@K 的均值 =
    「有百分之多少的问题，前 K 条里至少能找到一个相关片段」，
    是一个可以直接写进论文的百分比。
    """
    gold = set(gold_ids or ())
    if not gold or k <= 0:
        return None
    return 1.0 if set(_topk(retrieved_ids, k)) & gold else 0.0


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float | None:
    """召回率 @ K（Recall @ K）。

    公式：``Recall@K = |TopK(retrieved) ∩ gold| / |gold|``

    取值：``[0, 1]``；gold 为空或 k <= 0 时返回 ``None``（0/0 无定义）。

    分母用 ``|gold|`` 而不是 ``min(|gold|, K)``：召回率衡量的是
    「标准答案涉及的知识有没有被找齐」。如果一个问题的答案分散在 5 个片段里，
    而系统只能喂给模型 6 条，召回率天然接近上限 —— 这正是我们想要的信号：
    它把「检索深度不足」和「排序不好」区分开了。
    """
    gold = set(gold_ids or ())
    if not gold or k <= 0:
        return None
    return len(set(_topk(retrieved_ids, k)) & gold) / len(gold)


def precision_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float | None:
    """精确率 @ K（Precision @ K）。

    公式：``Precision@K = |TopK(retrieved) ∩ gold| / min(K, |retrieved|)``

    取值：``[0, 1]``；gold 为空或 k <= 0 返回 ``None``；
    有 gold 但一条都没检索到，返回 ``0.0``。

    **分母为什么是 ``min(K, |retrieved|)`` 而不是 K**：
    当系统只召回 3 条时，用 K=6 作分母等于假设「那 3 个不存在的槽位都是错的」，
    会系统性低估精确率。分母取「实际考察了多少条」，指标才是在评价
    「检索出来的东西有多少是对的」。

    **重要限制（偏悲观的下界）**：未被标注为 gold 的片段未必真的不相关，
    所以 Precision@K 是下界；本项目 gold_chunk_ids 由合成评估集给出，
    覆盖面同样有限，论文里必须说明这一点。
    """
    gold = set(gold_ids or ())
    if not gold or k <= 0:
        return None
    top = _topk(retrieved_ids, k)
    if not top:
        return 0.0
    return len(set(top) & gold) / len(top)


def mrr(retrieved_ids: Sequence[str], gold_ids: Iterable[str]) -> float | None:
    """平均倒数排名（Mean Reciprocal Rank，单样本上即 RR）。

    公式：``RR = 1 / rank(第一个相关片段)``；一个都没命中则 ``0.0``。

    取值：``(0, 1] ∪ {0.0}``；gold 为空时返回 ``None``。

    刻意**不带 K**：MRR 描述的是「**第一条**正确结果排得多靠前」，
    天然只关心首位，人为截断反而会丢掉「正确答案排在第 20 位」这种坏消息。
    聚合时对所有样本取算术平均，即 MRR。
    """
    gold = set(gold_ids or ())
    if not gold:
        return None
    for rank, cid in enumerate(_dedup_preserve_order(retrieved_ids), start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    """折损累计增益（二值相关 + 指数增益）：

        ``DCG@K = Σ_{i=1..K} rel_i / log2(i + 1)``

    ``rel_i ∈ {0,1}`` 表示第 i 位的片段是否相关。指数增益 ``2**rel - 1``
    在二值情形下恰好等于 ``rel``，所以不必额外乘幂。
    """
    total = 0.0
    for i, cid in enumerate(_topk(retrieved_ids, k), start=1):
        if cid in gold_ids:
            total += 1.0 / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float | None:
    """归一化折损累计增益（NDCG @ K）。

    公式：

        ``DCG@K  = Σ_{i=1..K} rel_i / log2(i + 1)``
        ``IDCG@K = Σ_{i=1..min(|gold|, K)} 1 / log2(i + 1)``   （理想排序）
        ``NDCG@K = DCG@K / IDCG@K``

    取值：``[0, 1]``；gold 为空或 k <= 0 时返回 ``None``。

    **除以 IDCG 的意义**：不同问题的相关片段数量不同（有的 1 条、有的 5 条），
    原始 DCG 不可比。归一化后「把最相关的排在最前面」得 1.0，
    「正确答案排在很后面」趋近 0，于是跨问题求均值才有意义。

    **IDCG 的分子为什么取 ``min(|gold|, K)``**：理想排序最多只能在 Top-K 里
    放下 K 条；若 gold 有 20 条而 K=6，用 20 条算 IDCG 会让 NDCG 永远上不去 ——
    那是「检索深度不足」的问题，不该记在「排序质量」的账上。

    **与 MRR 的分工**：MRR 只看第一条，NDCG 看整条排序。当一次检索命中多条
    相关片段时，NDCG 能反映它们的相对位置是否合理 —— 这正是混合检索与纯向量
    检索最常出现差异的地方（RRF 会把两路都命中的片段顶到前面）。
    """
    gold = set(gold_ids or ())
    if not gold or k <= 0:
        return None
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg <= 0:  # pragma: no cover - ideal_hits >= 1 时数学上不可能
        return None
    return _dcg_at_k(retrieved_ids, gold, k) / idcg


# ======================================================================
# 生成指标：引用质量
# ======================================================================
# 句末标点 + 换行。逗号不切句：中文长句里逗号太密，切碎后
# 「句子有没有带角标」会退化成噪声指标。
_SENTENCE_SPLIT = re.compile(r"[。！？!?；;\n]+")
# 「有实义内容」的判据：出现中日韩文字、数字或拉丁字母即算。
# 不判断语义，评估代码越简单越不容易出错。
_HAS_CONTENT = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf0-9A-Za-z]")
# 引用角标：半角 [1] / [1,2] / [1-3] / [1][2]，以及中文全角【1】
_CITATION = re.compile(r"\[\s*(\d+(?:\s*[,、，\-~]\s*\d+)*)\s*\]|【\s*(\d+)\s*】")
# 畸形输出 `[1-99999]` 的保护上限：真展开会瞬间造出上万个数字
_CITATION_LIMIT = 50


def split_sentences(text: str) -> list[str]:
    """把回答切成句子（按句末标点与换行切，保留原标点）。

    保留标点是因为调用方可能想原样展示句子；这里不清洗空白，
    由下游按需 ``strip()``。空串返回空列表。
    """
    if not text:
        return []
    return [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _counts_as_sentence(text: str) -> bool:
    """太短的残句（「好的」「。」）不该计入覆盖率，否则指标会被噪声主导。"""
    return bool(_HAS_CONTENT.search(text))


def extract_citation_numbers(text: str) -> list[int]:
    """抽出回答里所有引用编号（按出现顺序，允许重复）。

    支持 ``[1]``、``[1][3]``、``[1,3]``、``[1-3]``、``【1】`` 几种写法：
    小模型很少严格听话，写在 prompt 里的格式只是「推荐格式」，
    评估代码必须容忍常见变体，否则会把**格式问题**误判成**引用问题**。
    """
    out: list[int] = []
    for match in _CITATION.finditer(text or ""):
        body = match.group(1) or match.group(2) or ""
        for piece in re.split(r"[,、，]", body):
            piece = piece.strip()
            if not piece:
                continue
            if "-" in piece or "~" in piece:
                parts = re.split(r"[-~]", piece, maxsplit=1)
                try:
                    lo, hi = int(parts[0]), int(parts[1])
                except (IndexError, TypeError, ValueError):
                    continue
                if hi < lo:
                    lo, hi = hi, lo
                out.extend(range(lo, min(hi, lo + _CITATION_LIMIT) + 1))
                continue
            try:
                out.append(int(piece))
            except (TypeError, ValueError):
                continue
    return out


def citation_coverage(answer: str, citations: Sequence[Any] | None = None) -> float | None:
    """引用覆盖率：带 ``[n]`` 角标的句子占全部实义句子的比例。

    公式：``coverage = 带角标句子数 / 实义句子数``

    取值：``[0, 1]``；回答为空或没有实义句子时返回 ``None``。

    为什么需要它：RAG 的核心卖点是「答案可溯源」。覆盖率低意味着模型大段
    回答没有出处，用户无法核对 —— 哪怕答案碰巧是对的，系统的可信度也已经
    丧失。这是**可用性指标**，不是正确性指标。

    ``citations`` 参数保留但**不参与计算**：覆盖率只关心「有没有标角标」，
    「标得对不对」由 `citation_validity` 负责。两个指标分开，
    才能区分「模型忘了标」和「模型乱标」两种不同故障。

    已知局限：列举式回答（「一、… \\n 二、…」）若整段只有句末一个角标，
    覆盖率会偏低。但这恰好反映了「多个结论共用一个来源」的溯源粒度问题，
    偏严看待是合理的。
    """
    sentences = [s for s in split_sentences(answer) if _counts_as_sentence(s)]
    if not sentences:
        return None
    cited = sum(1 for s in sentences if _CITATION.search(s))
    return cited / len(sentences)


def citation_validity(answer: str, num_contexts: int) -> float | None:
    """引用有效率：角标编号落在 ``1..num_contexts`` 范围内的比例。

    公式：``validity = 合法角标数 / 角标总数``

    取值：``[0, 1]``；回答里没有任何角标时返回 ``1.0``；
    一条资料都没喂却出现角标时返回 ``0.0``。

    越界角标（只喂了 3 条资料却引用 ``[7]``）是**幻觉的硬证据**：
    模型在发明一个不存在的来源。这个指标不依赖任何模型或人工标注，
    是零成本的幻觉探测器，也是「严格模式 vs 宽松模式」对比里最直观的数据。

    **无角标返回 1.0 是否太宽松**：不算。此时「回答没有声称任何来源」，
    不存在编造行为；「回答该不该带角标」由 `citation_coverage` 负责。
    两个指标正交，组合起来才能区分「完全没引用」「引用了但越界」
    「引用完整」三种状态。
    """
    nums = extract_citation_numbers(answer)
    if not nums:
        return 1.0
    if num_contexts <= 0:
        return 0.0
    return sum(1 for n in nums if 1 <= n <= num_contexts) / len(nums)


# ======================================================================
# 中文友好的文本相似度（自实现，不引入新依赖）
# ======================================================================
# 停用词表与 dataset.extract_keywords 共用一份（见 _stopwords 模块的说明）：
# 两处判断标准一旦漂移，同一个词会在指标侧被当噪声、在关键词侧被当术语，
# 评估结论就变得无法解释。
_STOPWORDS = ALL_STOPWORDS

# 是否丢弃单字 CJK。jieba 常切出单字残片（「梯度」->「梯」），
# 这些字在别的句子里撞车的概率很高，会同时灌高 precision 和 recall，
# 让「忠实度」失去区分度。默认丢弃；需要逐字比对时可改这个模块级开关。
_DROP_SINGLE_CJK = True

# **任何情况下都不许丢**的词。
#
# 写这段是因为测试抓到一个真 bug：`_DROP_SINGLE_CJK` 会把单字的否定词
# 「不」一起删掉，于是「模型收敛」与「模型不收敛」的 token 集合完全相同，
# F1 = 1.0 —— 忠实度指标把**语义相反**的两句话判成一致。这是灾难性的。
#
# 所以否定 / 反义 / 程度类的高危单字必须豁免，且排在所有过滤规则之前。
# 这些词数量有限且闭合，穷举比让模型去猜可靠。
_NEVER_DROP = frozenset(
    "不 没 无 非 未 别 莫 勿 免 反 逆 否 空 零 全 半 超 极 最 更 越 太 过 欠 准 "
    "有 高 低 快 慢 多 少 真 假 对 错 同 异 增 减 正 负".split()
)

# 只保留这些字符，其余（标点、空白、emoji）当分隔符。
# 用白名单而不是黑名单：中文标点与全角符号的变体太多，列举不全。
_CJK_RUN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]+(?:[.\-_][A-Za-z0-9]+)*")

_jieba_lock = threading.Lock()
_jieba_ready = False


def _ensure_jieba() -> Any:
    """惰性初始化 jieba。

    与 `retrieval.bm25` 同样的取舍：jieba 首次 import 要加载词典（1~2 秒），
    放模块顶层会拖慢进程启动。评估不是热路径，按需加载更合适。
    """
    global _jieba_ready
    import jieba

    with _jieba_lock:
        if not _jieba_ready:
            jieba.initialize()
            _jieba_ready = True
    return jieba


def _normalize_tokens(text: str) -> list[str]:
    """分词 + 去停用词，输出用于集合运算的 token 列表。

    刻意**不复用** `retrieval.bm25.tokenize`：那个函数用 ``lcut_for_search``
    （搜索引擎模式），会把「神经网络」额外吐成「神经 / 网络 / 神经网络」。
    用它算 F1 时，同一个概念按不同粒度被重复计入，相似度会被系统性抬高，
    作为「忠实度代理」就失真了。评估需要的是**稳定、可复现**的切分，
    因此这里用 jieba 的精确模式 ``lcut``。
    """
    if not text or not text.strip():
        return []
    jieba = _ensure_jieba()
    tokens: list[str] = []
    for raw in jieba.lcut(text):
        raw = raw.strip()
        if not raw:
            continue
        # CJK 串：jieba 已经切好，不再二次拆分
        if _CJK_RUN.fullmatch(raw):
            # 高危单字豁免必须排在停用词和单字过滤之前，否则「不」会被删掉，
            # 让「收敛」与「不收敛」得到完全相同的 token 集合
            if raw in _NEVER_DROP:
                tokens.append(raw)
                continue
            if raw in _STOPWORDS:
                continue
            if _DROP_SINGLE_CJK and len(raw) == 1:
                continue
            tokens.append(raw)
            continue
        # 其余片段（中英混排 / 纯标点）：抽字母数字串，丢掉标点
        for match in _ALNUM_RUN.finditer(raw):
            tok = match.group(0).lower()
            if tok in _STOPWORDS:
                continue
            tokens.append(tok)
    return tokens


def token_f1(pred: str, gold: str) -> float | None:
    """词级 F1（jieba 分词 + 去停用词 + 多重集）。

    公式（用计数器统计重复词，即多重集而非集合）：

        ``P  = |pred_tok ∩ gold_tok| / |pred_tok|``
        ``R  = |pred_tok ∩ gold_tok| / |gold_tok|``
        ``F1 = 2PR / (P + R)``

    取值：``[0, 1]``；两侧都为空返回 ``None``；只有一侧为空返回 ``0.0``。

    交集按 ``min(计数)`` 算（``Counter`` 的 ``&`` 语义），
    因此答非所问的长篇大论无法靠重复关键词刷分。

    **为什么用 F1 而不是准确率 / BLEU**：
    * F1 同时惩罚「漏说」（低召回）和「多说」（低精确），
      而 RAG 回答最常见的两种毛病恰好就是这两类；
    * BLEU 建立在 n-gram 精确率上，对中文语序变化过于敏感，
      还需要短句惩罚之类的补丁，解释成本高于收益。

    返回 ``None`` 的语义是「没有可比内容」，聚合时应跳过而不是记 0 分。
    """
    pred_tokens = _normalize_tokens(pred)
    gold_tokens = _normalize_tokens(gold)
    if not pred_tokens and not gold_tokens:
        return None
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap <= 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _char_ngrams(text: str, n: int) -> set[str]:
    """抽字符 n-gram 集合。

    先归一化：小写化 + 去掉**所有空白**。中文里空格/换行只是排版产物，
    保留它们会让「机器 学习」与「机器学习」判为不重叠。
    标点**保留**：它是句子结构的一部分，去掉后「A，但 B」与「A 但 B」无差别，
    反而掩盖了改写幅度。

    用集合而非多重集：这里要的是「覆盖了哪些字面组合」，
    重复出现不该让分数翻倍（那是文本长度，不是相似度）。
    """
    if n < 1:
        return set()
    normalized = re.sub(r"\s+", "", (text or "").lower())
    if len(normalized) < n:
        return set()
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def char_ngram_overlap(pred: str, gold: str, n: int = 2) -> float | None:
    """字符 n-gram 重叠度（默认 2-gram），即字符级 F1。

    公式：``F1 = 2|G(pred) ∩ G(gold)| / (|G(pred)| + |G(gold)|)``，
    其中 ``G(x)`` 是 x 去空白、小写化后的字符 n-gram 集合。

    取值：``[0, 1]``；两侧都为空返回 ``None``；只有一侧为空返回 ``0.0``。

    中文里比词级更稳的原因：中文没有词边界，分词器对未登录词
    （新术语、专有名词、中英混排）的切分不稳定；字符 2-gram 不需要词典，
    同义改写（「训练」/「训练过程」）也能拿到部分重合分。

    ``n=2`` 是经验值：``n=1`` 会把「的」「是」这类高频字算进去，
    区分度接近随机；``n>=3`` 又过于严格，稍有改写就归零。
    """
    if n < 1:
        return 0.0
    pred_grams = _char_ngrams(pred, n)
    gold_grams = _char_ngrams(gold, n)
    if not pred_grams and not gold_grams:
        return None
    if not pred_grams or not gold_grams:
        return 0.0
    overlap = len(pred_grams & gold_grams)
    if overlap <= 0:
        return 0.0
    return 2 * overlap / (len(pred_grams) + len(gold_grams))


def answer_context_overlap(answer: str, contexts: Sequence[str]) -> float | None:
    """回答与参考资料的字面重合度 —— 「忠实度」的廉价代理指标。

    做法：把全部参考资料拼接成长文，与回答算一次 `token_f1`。

    取值：``[0, 1]``；回答为空返回 ``None``；回答非空但参考资料为空返回 ``0.0``。

    ## 为什么是「廉价代理」而不是真忠实度

    真正的忠实度要求判断「回答里的每个论断是否被资料**蕴含**」，
    这需要自然语言推理模型。字面重合度只能证明**用词**来自资料，
    不能证明**逻辑**来自资料 —— 模型完全可以抄资料里的词、编造错误的关系。

    所以它的定位是**筛子**，不是结论：
      * 重合度极低（< 0.15）几乎必然在自由发挥，值得人工重点抽查；
      * 重合度高也不代表正确，还需要 `judge_faithfulness` 或人工复核。

    ## 为什么拼接所有资料，而不是逐条比完取最大值

    模型的答案常常综合多条资料，逐条比会低估综合型回答。代价是拼接长文里
    有大量与回答无关的内容，但这被 F1 的分母正确惩罚了（参考资料越长、
    越难拿到高 F1），方向上是对的。如果后续发现「参考资料篇幅大导致分数
    系统性偏低」，换成逐条最大 + 长度加权即可 —— 接口不用变。
    """
    if not (answer or "").strip():
        return None
    gold = "\n".join(c for c in (contexts or []) if c and c.strip())
    if not gold.strip():
        return 0.0
    return token_f1(answer, gold)


# ======================================================================
# 聚合
# ======================================================================
def aggregate_metrics(values: Mapping[str, Iterable[float | None]]) -> dict[str, dict[str, Any]]:
    """把逐样本指标聚合成「均值 ± 标准差 + 极值 + 有效样本数」。

    输出结构（可直接 `json.dumps`）：

        {"recall": {"mean": 0.72, "std": 0.11, "min": 0.5,
                    "max": 1.0, "n": 12}, ...}

    **为什么必须带上 n**：降级模式下部分样本没有 gold，指标为 `None`。
    如果只报均值不报 n，读者无法判断「0.72 是基于 12 条还是 1 条算出来的」。
    论文表格里 n 是必填列，这里就把它固化进数据结构。

    **标准差为什么用样本标准差（n-1）**：这些样本是从总体里抽样出来的
    一组问题，我们要估计的是「系统在总体上的表现波动」，属于推断统计场景，
    用 n-1 更合适。只有一个有效样本时 `statistics.stdev` 会抛异常，
    这里显式返回 0.0 —— 「只有一个样本，波动无法估计」和「完全没有波动」
    在报告里不作区分，但要靠 n=1 提醒读者别当真。
    """
    out: dict[str, dict[str, Any]] = {}
    for name, raw_values in values.items():
        nums = [float(v) for v in raw_values if v is not None]
        if not nums:
            out[name] = {"mean": None, "std": None, "min": None, "max": None, "n": 0}
            continue
        out[name] = {
            "mean": statistics.fmean(nums),
            "std": statistics.stdev(nums) if len(nums) > 1 else 0.0,
            "min": min(nums),
            "max": max(nums),
            "n": len(nums),
        }
    return out


# ======================================================================
# LLM 裁判（可选，用得上才调）
# ======================================================================
class _ChatProvider(Protocol):
    """裁判只需要 LLMProvider 的一个方法。

    用 Protocol 而不是直接标注 `LLMProvider`：裁判逻辑与具体通道无关，
    测试里塞个假对象就能验证解析与降级，不必构造真的 Ollama 客户端。
    """

    async def chat(
        self,
        messages: Sequence[Any],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any: ...


_RELEVANCE_SYSTEM = """你是一个严格的问答质量评估员。

任务：判断【回答】是否真正回答了【问题】。

评分标准（1~5 分整数）：
5 = 完全正确且切题，信息完整
4 = 基本正确，略有遗漏或冗余
3 = 部分正确，存在明显遗漏或含少量无关内容
2 = 大部分不正确，只沾了边
1 = 完全答非所问，或答成另一个问题

只输出下面两行，不要任何其他内容：
评分：<1~5 的整数>
理由：<不超过 50 字的一句话>"""

_FAITHFULNESS_SYSTEM = """你是一个严格的事实验证员。

任务：判断【回答】是否**完全**由【参考资料】支撑。

评分标准（0~1 之间的小数）：
1.0 = 所有论断都能在参考资料中找到依据
0.7 = 主要论断有依据，个别细节是模型自行补充的
0.4 = 只有一半左右有依据，存在明显的外部知识
0.0 = 完全找不到依据，或与参考资料矛盾

注意：用词不同但意思相同，算「有依据」；不要因为表述差异扣分。
只输出下面两行，不要任何其他内容：
评分：<0~1 的小数>
理由：<不超过 50 字的一句话>"""

# 解析模型输出：「评分：4」「评分: 4/5」「评分 4 分」「得分 4」
_SCORE_LINE = re.compile(
    r"(?:评分|得分|分数|score)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_REASON_LINE = re.compile(r"(?:理由|原因|reason)\s*[:：]?\s*(.+)", re.IGNORECASE)
# 没有任何标签时的兜底：抓第一个独立出现的数字
_BARE_SCORE = re.compile(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)(?![0-9])")
_MAX_REASON_CHARS = 200


def _parse_scored_reply(raw: str) -> tuple[float, str] | None:
    """从模型回复里抠出 (分数, 理由)。抠不出来返回 None。

    容错点（都是小模型真实会犯的）：
      * 带不带「分」「/5」后缀；
      * 全角冒号、半角冒号、直接空格；
      * 评分和理由写在同一行；
      * 完全不按格式、正文里只飘着一个数字。

    只要连数字都找不到，就返回 None —— **宁可丢一条裁判数据，
    也不能瞎猜一个分数**，否则评估报告里混进模型噪声，比没有更糟。
    """
    text = (raw or "").strip()
    if not text:
        return None

    score: float | None = None
    match = _SCORE_LINE.search(text)
    if match:
        score = float(match.group(1))
    else:
        # 兜底只在「整段回复很短」时启用：长回复里随便一个数字
        # （比如「参考资料 3 条」）会被误当成分数
        if len(text) <= 40:
            bare = _BARE_SCORE.search(text)
            if bare:
                score = float(bare.group(1))
    if score is None:
        return None

    reason = ""
    match = _REASON_LINE.search(text)
    if match:
        reason = match.group(1).strip()
    if not reason:
        # 没有「理由：」标签时，把评分所在行之外的内容当理由
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        reason = " ".join(ln for ln in lines if not _SCORE_LINE.search(ln))
    return score, reason[:_MAX_REASON_CHARS]


async def _ask_judge(provider: _ChatProvider, system: str, user: str) -> str | None:
    """调一次裁判模型，返回原始文本；任何异常都吞掉并返回 None。

    为什么吞异常：LLM 裁判是**可选项**。Ollama 没启动、模型被卸载、
    输出超时 —— 这些都不该让整轮评估失败。评估脚本的价值在于能跑完，
    少几个裁判分数只是报告里多一列空值。
    """
    from app.core.llm import ChatMessage

    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
    try:
        result = await provider.chat(messages, temperature=0.0, max_tokens=256)
    except Exception as exc:  # LLMError / 超时 / 连接失败，一律降级
        logger.warning("LLM 裁判调用失败，本条记为缺失：%s", exc)
        return None
    content = getattr(result, "content", None)
    if content is None and isinstance(result, str):
        content = result
    return str(content or "")


async def judge_answer_relevance(
    provider: _ChatProvider,
    question: str,
    answer: str,
) -> tuple[float, str] | None:
    """用 LLM 给「回答是否切题」打 1~5 分，返回 ``(分数, 理由)``。

    **必须优雅降级**：模型输出解析不出分数时返回 ``None``
    （而不是 0 分，也不是抛异常）。裁判失败与「回答很差」是两件事，
    混在一起会让评估报告失真。调用方（runner）据此把该条记为缺失。

    为什么要有它：`token_f1` / `char_ngram_overlap` 只能测字面，
    测不出「答非所问但用词很像」和「表述完全不同但意思正确」。
    裁判模型补的正是这一块。

    已知局限：裁判也是 LLM，会有偏好（偏爱长回答、偏爱与自己风格接近的
    回答），且与生成模型同源时存在自我偏好偏差。论文里应把它当作
    **辅助参考**，并与字面指标、人工抽查三方交叉验证。
    """
    if not (question or "").strip() or not (answer or "").strip():
        return None

    user = f"【问题】\n{question.strip()}\n\n【回答】\n{answer.strip()}\n\n请评分："
    raw = await _ask_judge(provider, _RELEVANCE_SYSTEM, user)
    if raw is None:
        return None

    parsed = _parse_scored_reply(raw)
    if parsed is None:
        logger.debug("相关性裁判输出无法解析，忽略：%r", raw[:200])
        return None
    score, reason = parsed
    # 模型偶尔会写 6 分或 0 分，夹到定义域内而不是丢弃 ——
    # 「满分偏上」和「零分」都是真实存在的判断倾向
    return float(min(5.0, max(1.0, score))), reason


async def judge_faithfulness(
    provider: _ChatProvider,
    answer: str,
    contexts: Sequence[str],
) -> tuple[float, str] | None:
    """用 LLM 判断回答是否完全由参考资料支撑，返回 ``(0~1 分数, 理由)``。

    与 `answer_context_overlap` 的分工：那个是零成本的字面代理，
    这个是真正去读语义的裁判，但更慢、更贵、且依赖模型可用。
    默认关闭（`run_evaluation(use_llm_judge=False)`），需要论文里
    「忠实度」那一列数据时再打开。

    降级行为与 `judge_answer_relevance` 一致：解析失败返回 ``None``；
    回答为空或参考资料为空时也返回 ``None``（无从判断）。
    """
    if not (answer or "").strip():
        return None
    blocks = [c.strip() for c in (contexts or []) if c and c.strip()]
    if not blocks:
        return None

    # 每段资料截断：裁判只需要看内容，不需要看整篇论文；
    # 同时防止把本地小模型的上下文撑爆导致超时
    joined = "\n\n".join(f"[{i}] {c[:1200]}" for i, c in enumerate(blocks, start=1))
    user = f"【参考资料】\n{joined}\n\n【回答】\n{answer.strip()}\n\n请评分："
    raw = await _ask_judge(provider, _FAITHFULNESS_SYSTEM, user)
    if raw is None:
        return None

    parsed = _parse_scored_reply(raw)
    if parsed is None:
        logger.debug("忠实度裁判输出无法解析，忽略：%r", raw[:200])
        return None
    score, reason = parsed
    # 0~1 之外的值（模型可能写 4 分制或百分制）夹到边界，不静默丢弃
    return float(min(1.0, max(0.0, score))), reason


__all__ = [
    "RETRIEVAL_METRIC_NAMES",
    "aggregate_metrics",
    "answer_context_overlap",
    "char_ngram_overlap",
    "citation_coverage",
    "citation_validity",
    "extract_citation_numbers",
    "hit_rate_at_k",
    "judge_answer_relevance",
    "judge_faithfulness",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "split_sentences",
    "token_f1",
]
