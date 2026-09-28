"""评估执行器：对每个样本 × 每个检索模式跑完整链路，产出结构化报告。

## 为什么直接调 `rag.engine.answer_once`，而不是自己拼一遍检索+生成

`answer_once` 就是线上问答走的同一条链路（含查询改写、检索、Prompt 组装、
流式生成、落库）。评估自己实现一套"实验版链路"，迟早会和线上行为漂移，
届时论文里的数据就不再描述这个系统了。

代价是评估必须连数据库和一个可用的生成通道。为此本项目保留了两条**降级路径**：

  * 没有 LLM：`answer_once` 会在 `error` 字段里带回失败原因，报告照常产出，
    只是生成类指标为空 —— 「检索做完了、生成没做」这种情况本身也是有效结论
    （例如想单独比较三种检索模式时，完全可以不接模型）；
  * 没有知识库：传 `offline_contexts=True`，直接用样本自带的 `context`
    字段当参考资料，用来验证报告管线本身没坏。

## 标准答案的三档降级（`gold_mode`）

评估结果必须自带可信度标注。本项目没有人工标注，所以按下列顺序降级，
并把**实际用的是哪一档**逐样本记录进报告：

1. `chunk_ids` —— 有 `gold_chunk_ids` 时最准。此时 recall / precision /
   nDCG 都是严格的「标准答案 vs 检索结果」。
2. `answer_overlap` —— 没有 chunk_id 但有 `gold_answer` 时，
   用「片段正文与标准答案的 token F1 ≥ ANSWER_OVERLAP_THRESHOLD」判定相关。
   这比关键词更能反映语义，阈值以下的片段算作不相关。
3. `keywords` —— 只有 `gold_keywords` 时，用关键词是否出现在片段正文里判定。
   召回率仍是有意义的（"该片段找回来了没有"），
   但**精确率与 nDCG 会偏乐观**：关键词没命中的片段被当成不相关，
   而实际上它可能也相关内容。

报告顶层因此带 `notes` 字段，把这三档的差异和适用边界直接写进 JSON。
不做这种标注的评估报告，跨模式对比出来的「混合检索 +8%」是不能信的。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.eval.dataset import EvalSample
from app.core.eval.metrics import (
    aggregate_metrics,
    answer_context_overlap,
    citation_coverage,
    citation_validity,
    hit_rate_at_k,
    judge_answer_relevance,
    judge_faithfulness,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    token_f1,
)
from app.core.llm import ChatMessage, build_provider
from app.core.retrieval.hybrid import get_bm25_index
from app.db.models import Chunk, KnowledgeBase

logger = logging.getLogger(__name__)

# 支持的检索模式。与 `retrieval.hybrid.retrieve` 的 SearchMode 一致，
# 这里再列一份是为了让报告能自己声明"跑过哪几种"，不依赖上游常量是否导出。
DEFAULT_MODES: tuple[str, ...] = ("hybrid", "dense_only", "sparse_only")

# 用「片段正文与标准答案的 token F1」判定相关时的阈值。
# 0.3 的依据：标准答案若基本出自某片段，F1 通常在 0.5 以上；
# 只是提到相关话题的片段一般在 0.1 以下。0.3 落在两簇之间的空档，
# 偏严 —— 宁可少判几条第相关，也不要让 precision 被"沾边"的片段灌水。
ANSWER_OVERLAP_THRESHOLD = 0.3

# 进度回调：runner 不关心它是打印、写日志还是推进度条
ProgressCallback = Callable[[str, int, int], Awaitable[None] | None]


# ======================================================================
# 报告数据结构
# ======================================================================
@dataclass(slots=True)
class JudgeReport:
    """LLM 裁判结果。分数为 None 表示**没判成功**（模型不可用或输出无法解析）。

    注意：`None` 与 `0.0` 含义完全不同 —— 前者是"没有数据"，
    后者是"判了，判得很低"。聚合时前者被跳过，后者参与均值。
    """

    relevance_score: float | None = None
    relevance_reason: str = ""
    faithfulness_score: float | None = None
    faithfulness_reason: str = ""


@dataclass(slots=True)
class EvalSampleResult:
    """单条样本在某个检索模式下的全部观测值。"""

    question: str
    mode: str
    gold_mode: str  # chunk_ids | answer_overlap | keywords | none
    answer: str = ""
    error: str | None = None
    rewritten_query: str | None = None

    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    citation_count: int = 0

    metrics: dict[str, float | None] = field(default_factory=dict)
    judge: JudgeReport | None = None

    retrieval_ms: int = 0
    generation_ms: int = 0
    elapsed_ms: int = 0


@dataclass(slots=True)
class EvalReport:
    """一次评估的完整结果，`asdict()` 后可直接 `json.dumps`。

    结构分层是刻意的：
      * `aggregate` —— 每个模式一组均值，论文里的表格直接用它；
      * `samples`   —— 逐样本明细，画散点图/箱线图、做误差分析用它；
      * `timings`   —— 效率数据（检索 vs 生成的耗时拆分）；
      * `notes`     —— 本次评估的可信度声明。
    """

    kb_id: str
    kb_name: str
    dataset_name: str
    modes: list[str]
    provider: str = ""
    model: str = ""
    generated_at: str = ""
    sample_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    top_k: int = 6
    use_llm_judge: bool = False
    offline_contexts: bool = False
    aggregate: dict[str, dict[str, Any]] = field(default_factory=dict)
    samples: list[EvalSampleResult] = field(default_factory=list)
    timings: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转成纯 dict（JSON 可序列化）。"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """序列化成 JSON 字符串，供 CLI 落盘与接口返回。"""
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)


# ======================================================================
# 标准答案判定
# ======================================================================
def keyword_hits(text: str, keywords: Sequence[str]) -> list[str]:
    """返回片段正文里命中的关键词（不区分大小写）。

    中英文用同一套逻辑：英文关键词统一小写后做子串匹配。
    不做分词再做集合交，是因为关键词本身就是从语料里抽出来的短语，
    子串匹配更稳（「BM25」不该被分词器切开，「RAG」也不该被匹配到「RAGE」以外的
    词里 —— 但在这个尺度的语料上不值得为此上正则边界）。
    """
    lowered = (text or "").lower()
    return [kw for kw in keywords if kw and kw.lower() in lowered]


def _score_by_answer_overlap(
    sample: EvalSample,
    retrieved: Sequence[dict[str, Any]],
) -> list[str]:
    """按「片段正文与标准答案的 token F1」判定相关片段 id。"""
    scored: list[tuple[str, float]] = []
    for item in retrieved:
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            continue
        score = token_f1(sample.gold_answer, str(item.get("content") or ""))
        scored.append((chunk_id, score or 0.0))
    hit_ids = {cid for cid, score in scored if score >= ANSWER_OVERLAP_THRESHOLD}
    # 保持检索顺序，保证召回/精确率计算里"名次"的语义不被破坏
    return [str(item["chunk_id"]) for item in retrieved if str(item.get("chunk_id")) in hit_ids]


def _score_by_keywords(
    sample: EvalSample,
    retrieved: Sequence[dict[str, Any]],
) -> list[str]:
    """按关键词字面命中判定相关片段 id。"""
    out: list[str] = []
    for item in retrieved:
        if keyword_hits(str(item.get("content") or ""), sample.gold_keywords):
            chunk_id = str(item.get("chunk_id") or "")
            if chunk_id:
                out.append(chunk_id)
    return out


def resolve_gold(
    sample: EvalSample,
    retrieved: Sequence[dict[str, Any]],
) -> tuple[list[str], str]:
    """把样本的标准答案解析成「一组相关 chunk_id」+ 判定方式。

    返回的第二个值是 `gold_mode`，务必记进报告：它是读者判断
    「这份指标能不能信」的唯一依据。
    """
    if sample.gold_chunk_ids:
        return list(sample.gold_chunk_ids), "chunk_ids"
    if sample.gold_answer and retrieved:
        hits = _score_by_answer_overlap(sample, retrieved)
        if hits:
            return hits, "answer_overlap"
    if sample.gold_keywords and retrieved:
        hits = _score_by_keywords(sample, retrieved)
        if hits:
            return hits, "keywords"
    # 三个都没戏：检索指标全部无定义（返回 None），但生成类指标照算
    return [], "none"


# ======================================================================
# 单样本评估
# ======================================================================
def _retrieval_metrics(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int,
) -> dict[str, float | None]:
    """算一组检索指标。gold 为空时各函数自己会返回 None，这里不做特判。"""
    return {
        "hit_rate": hit_rate_at_k(retrieved_ids, gold_ids, k),
        "recall": recall_at_k(retrieved_ids, gold_ids, k),
        "precision": precision_at_k(retrieved_ids, gold_ids, k),
        "mrr": mrr(retrieved_ids, gold_ids),
        "ndcg": ndcg_at_k(retrieved_ids, gold_ids, k),
    }


async def _evaluate_one(
    *,
    db: Session,
    sample: EvalSample,
    mode: str,
    kb_id: str,
    provider_name: str | None,
    top_k: int,
    use_llm_judge: bool,
    judge_provider: Any,
    offline_contexts: bool,
) -> EvalSampleResult:
    """跑一条样本的一个模式。

    生成阶段用本地样本的 `context`（离线模式）或真实 RAG 链路。
    离线模式存在的意义：没有 Ollama、没有知识库时也能验证
    「指标计算 + 报告序列化」这条最容易被写错的路。
    """
    started = time.perf_counter()
    result = EvalSampleResult(question=sample.question, mode=mode, gold_mode="none")

    if offline_contexts:
        # 离线模式：把样本自带片段当作"检索结果"，生成阶段留空
        contexts = [sample.context] if sample.context else []
        result.retrieved_chunks = [
            {
                "chunk_id": f"offline:{i}",
                "content": text,
                "filename": "",
                "page": None,
                "fused_score": 0.0,
            }
            for i, text in enumerate(contexts, start=1)
        ]
        result.retrieved_ids = [str(c["chunk_id"]) for c in result.retrieved_chunks]
        answer = sample.gold_answer
        result.answer = answer
        result.generation_ms = 0
        result.retrieval_ms = 0
    else:
        from app.core.rag.engine import answer_once
        from app.db.models import Conversation, new_id

        # 每次评估都用独立会话：共用会话会把上一题的问答喂进历史，
        # 触发查询改写，从而污染"单轮检索"这个实验条件
        conversation_id = new_id()
        try:
            convo = Conversation(id=conversation_id, kb_id=kb_id, title="[eval]")
            db.add(convo)
            db.commit()
        except Exception as exc:  # pragma: no cover - 数据库异常
            logger.warning("创建评估会话失败：%s", exc)
            db.rollback()

        try:
            outcome = await answer_once(
                conversation_id=conversation_id,
                kb_id=kb_id,
                question=sample.question,
                provider_name=provider_name,
                mode=mode,  # type: ignore[arg-type]
            )
        except Exception as exc:
            logger.exception("样本评估失败（mode=%s）：%s", mode, sample.question)
            result.error = f"{type(exc).__name__}: {exc}"
            result.elapsed_ms = int((time.perf_counter() - started) * 1000)
            return result

        result.answer = str(outcome.get("answer") or "")
        result.error = outcome.get("error")
        result.rewritten_query = outcome.get("rewritten_query")
        result.citations = list(outcome.get("citations") or [])
        result.citation_count = len(result.citations)
        result.retrieval_ms = int(outcome.get("retrieval_ms") or 0)
        result.generation_ms = int(outcome.get("latency_ms") or 0)
        result.retrieved_ids = [str(c.get("chunk_id")) for c in result.citations if c.get("chunk_id")]

        # 引用卡片里带 preview（前 400 字），做关键词/重合判定时够用；
        # 命中不到时再回表取全文，避免为每条样本多查 N 次数据库
        contexts = [
            (str(c.get("preview") or ""), str(c.get("chunk_id") or "")) for c in result.citations
        ]
        if any(not preview for _, preview in contexts) and contexts:
            ids = [cid for _, cid in contexts if cid]
            rows = db.execute(
                select(Chunk.id, Chunk.content).where(Chunk.id.in_(ids))
            ).all()
            full = {str(r[0]): str(r[1]) for r in rows}
            contexts = [(preview or full.get(cid, ""), cid) for preview, cid in contexts]

        result.retrieved_chunks = [
            {
                "chunk_id": cid,
                "content": preview,
                "filename": str(result.citations[i].get("filename") or ""),
                "page": result.citations[i].get("page"),
                "fused_score": result.citations[i].get("score"),
            }
            for i, (preview, cid) in enumerate(contexts)
        ]

    # ---------------- 检索指标 ----------------
    gold_ids, gold_mode = resolve_gold(sample, result.retrieved_chunks)
    result.gold_mode = gold_mode
    result.metrics.update(_retrieval_metrics(result.retrieved_ids, gold_ids, top_k))

    # ---------------- 生成指标（不需要外部模型） ----------------
    context_texts = [str(c.get("content") or "") for c in result.retrieved_chunks]
    result.metrics["citation_coverage"] = citation_coverage(result.answer, result.citations)
    # num_contexts 用"喂给模型的资料条数"而不是 citations 长度：
    # 引用越界指的是"引用了不存在于本轮上下文里的编号"
    result.metrics["citation_validity"] = citation_validity(
        result.answer, len(context_texts) or len(result.citations)
    )
    result.metrics["answer_overlap"] = answer_context_overlap(result.answer, context_texts)

    # ---------------- LLM 裁判（可选） ----------------
    if use_llm_judge and judge_provider is not None and result.answer.strip():
        judge = JudgeReport()
        try:
            rel = await judge_answer_relevance(judge_provider, sample.question, result.answer)
        except Exception as exc:  # pragma: no cover - 裁判内部已吞异常，这里再兜一层
            logger.warning("相关性裁判异常：%s", exc)
            rel = None
        if rel is not None:
            judge.relevance_score, judge.relevance_reason = rel

        try:
            faith = await judge_faithfulness(judge_provider, result.answer, context_texts)
        except Exception as exc:  # pragma: no cover
            logger.warning("忠实度裁判异常：%s", exc)
            faith = None
        if faith is not None:
            judge.faithfulness_score, judge.faithfulness_reason = faith

        result.judge = judge
        result.metrics["judge_relevance"] = judge.relevance_score
        result.metrics["judge_faithfulness"] = judge.faithfulness_score

    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result


# ======================================================================
# 报告聚合
# ======================================================================
def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stat_block(values: Sequence[float]) -> dict[str, float | None]:
    """一组数字的 mean/min/max。空输入全部返回 None。"""
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": _mean(values),
        "min": min(values),
        "max": max(values),
    }


def _aggregate_mode(rows: Sequence[EvalSampleResult]) -> dict[str, dict[str, Any]]:
    """把同一模式下的逐样本指标聚合成 `{指标: {mean, std, min, max, n}}`。

    **每条样本的指标都参与，而不是取平均后再平均**：先算样本级指标再聚合，
    是信息检索评估的标准做法（每查询指标 → 宏观平均，macro-average）。
    宏观平均对样本量少的问题更公平，也让标准差有明确含义：
    它衡量「系统在不同问题上的稳定性」。
    """
    values: dict[str, list[float | None]] = {}
    for row in rows:
        for name, value in row.metrics.items():
            values.setdefault(name, []).append(value)
    return aggregate_metrics(values)


def _aggregate_timings(rows: Sequence[EvalSampleResult]) -> dict[str, Any]:
    """耗时统计：检索 / 生成 / 端到端。

    检索和生成分开统计是必须的：混合检索比纯向量多一次 BM25，
    多出来的开销如果混进端到端均值里，会看不清是"检索变慢了"
    还是"生成变慢了"。论文里"效率"那一节需要这个拆分。
    """
    retrieval = [float(r.retrieval_ms) for r in rows]
    generation = [float(r.generation_ms) for r in rows if r.generation_ms > 0]
    elapsed = [float(r.elapsed_ms) for r in rows]
    return {
        "retrieval_ms": _stat_block(retrieval),
        "generation_ms": _stat_block(generation),
        "elapsed_ms": _stat_block(elapsed),
    }


def _build_notes(
    *,
    gold_modes: dict[str, int],
    failed: int,
    use_llm_judge: bool,
    offline_contexts: bool,
) -> list[str]:
    """把本次评估的可信度边界写进报告。

    这些说明**必须随报告一起走**：报告被单独发给别人、或者半年后自己回看时，
    只看到「recall=0.9」而没有这句限定，就会得出错误的结论。
    """
    notes: list[str] = []
    if gold_modes.get("chunk_ids"):
        notes.append(
            f"{gold_modes['chunk_ids']} 条样本使用 gold_chunk_ids 判定，"
            "这是最严格的一档，各检索指标可直接跨模式对比。"
        )
    if gold_modes.get("answer_overlap"):
        notes.append(
            f"{gold_modes['answer_overlap']} 条样本没有 gold_chunk_ids，"
            f"改用「片段正文与标准答案的 token F1 >= {ANSWER_OVERLAP_THRESHOLD}」判定相关片段。"
            "召回率可信；精确率与 NDCG 偏乐观（未被判为相关的片段可能也相关内容）。"
        )
    if gold_modes.get("keywords"):
        notes.append(
            f"{gold_modes['keywords']} 条样本只用 gold_keywords 判定。"
            "关键词未命中的片段被当作不相关，"
            "因此精确率与 NDCG 只能作为**上界**参考，跨模式对比仍有意义（条件相同）。"
        )
    if gold_modes.get("none"):
        notes.append(
            f"{gold_modes['none']} 条样本缺乏任何可用的标准答案，"
            "检索指标记为缺失（None）而非 0 分，聚合时已跳过。"
        )
    if failed:
        notes.append(f"{failed} 条样本的执行本身出错（见明细里的 error 字段），其指标缺失。")
    if use_llm_judge:
        notes.append(
            "已启用 LLM 裁判（judge_relevance 1~5 分 / judge_faithfulness 0~1）。"
            "裁判也是模型，存在长度偏好与自我偏好，只能作为辅助证据。"
        )
    else:
        notes.append(
            "未启用 LLM 裁判；忠实度由 answer_overlap（字面重合）代理，"
            "它只能证明用词来自资料，不能证明论断被资料支撑。"
        )
    if offline_contexts:
        notes.append(
            "本次为 **离线模式**：参考资料取自数据集自带的 context 字段，"
            "回答直接复用 gold_answer，未调用检索链路与生成模型。"
            "结论仅用于验证指标与报告管线，不代表系统真实性能。"
        )
    return notes


# ======================================================================
# 主入口
# ======================================================================
async def run_evaluation(
    db: Session,
    kb_id: str,
    samples: Sequence[EvalSample],
    modes: Sequence[str] = DEFAULT_MODES,
    *,
    provider_name: str | None = None,
    use_llm_judge: bool = False,
    progress_cb: ProgressCallback | None = None,
    offline_contexts: bool = False,
    top_k: int | None = None,
) -> EvalReport:
    """跑一轮完整评估，返回 `EvalReport`。

    ## 参数

    * `db`：调用方提供的会话。runner 只用它读知识库元信息和少量片段正文；
      真正的问答链路内部用自己的短会话（见 `rag.engine` 的设计说明），
      所以评估不会长时间占用数据库连接。
    * `samples`：样本列表，通常来自 `dataset.load_dataset` / 内置示例集。
    * `modes`：要对比的检索模式，默认三种全跑（这就是消融实验）。
    * `use_llm_judge`：是否额外调 LLM 打分。默认关闭 —— 它会让评估耗时
      成倍增加，且需要可用的生成通道。
    * `progress_cb`：`(mode, 已完成数, 总数)` 回调，可为同步或异步函数。
      回调抛出的异常会被吞掉：进度条坏了不该让实验失败。
    * `offline_contexts`：离线自检模式，见模块 docstring。

    ## 返回

    `EvalReport`，`to_dict()` / `to_json()` 后可直接落盘或返回给前端。

    ## 失败处理

    单条样本失败（检索异常、生成超时）不会中断整轮：记录 `error` 后继续，
    最后在 `failed_count` 和 `notes` 里如实报告。评估最忌讳"跑到一半崩了
    什么都没留下" —— 部分失败的结果本身就是有用的实验数据。
    """
    if not samples:
        raise ValueError("样本列表为空，无法评估")

    mode_list = [m for m in modes if m] or list(DEFAULT_MODES)

    from app.core.settings_service import SettingsService

    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise ValueError(f"知识库不存在：{kb_id}")
    kb_name = kb.name

    # 空库不该静默跑出一份"全部召回率为 0"的报告：那种数据看着像系统缺陷，
    # 实际是数据没准备好。这里给出明确警告，让使用者先去看知识库状态。
    has_chunk = db.execute(select(Chunk.id).where(Chunk.kb_id == kb_id).limit(1)).first()
    if has_chunk is None and not offline_contexts:
        logger.warning("知识库 %s 没有任何片段，检索将全部落空", kb_name)

    cfg = SettingsService(db).effective()
    if top_k is None:
        top_k = int(cfg.get("final_top_k") or 6)

    # 检索只返回 final_top_k 条，报告也按同一个 K 计指标 ——
    # 否则会出现「拿 K=6 的检索结果去算 Recall@10」这种自欺欺人的数字
    provider = None
    provider_label = ""
    model_label = ""
    if not offline_contexts:
        try:
            provider = build_provider(cfg, provider=provider_name)
            provider_label, model_label = provider.name, provider.model
        except Exception as exc:
            # 生成通道不可用不致命：检索类指标照样能跑出来
            logger.warning("生成通道不可用（%s），本次只统计检索类指标", exc)

    judge_provider = None
    if use_llm_judge:
        if provider is not None:
            judge_provider = provider
        else:
            try:
                judge_provider = build_provider(cfg, provider=provider_name)
            except Exception as exc:
                logger.warning("裁判通道不可用，本次跳过 LLM 裁判：%s", exc)

    total = len(samples)
    results: list[EvalSampleResult] = []

    # ---------------- 预热（保证各模式耗时可比）----------------
    # 冷启动时 Ollama 要把向量模型加载进显存，首次调用要好几秒。不预热的话，
    # 这笔一次性开销会全部记在**第一个被评估的模式**头上，跨模式的耗时对比就失真了。
    #
    # 这不是臆测：实测中 hybrid 因为排在最前面，平均检索耗时被抬到 2136 ms，
    # 而 dense_only 只有 386 ms —— 可 hybrid 仅仅多跑了一路耗时 0 ms 的 BM25。
    # 用 scripts/bench_retrieval.py 隔离生成后复测，稳定态下两者都是约 380 ms，
    # 差异完全来自「谁先跑」。
    #
    # 预热顺序有讲究：先 BM25（jieba 词典初始化 + 建索引），再生成模型，
    # 最后向量模型 —— 让它成为显存里最新的那个，紧接着的计时检索才能拿到稳定值。
    if not offline_contexts:
        warm_kb = db.get(KnowledgeBase, kb_id)
        if warm_kb is not None:
            try:
                get_bm25_index(db, kb_id)
            except Exception as exc:
                logger.debug("BM25 预热失败（不影响评估）：%s", exc)

            if provider is not None:
                try:
                    await provider.chat(
                        [ChatMessage(role="user", content="你好")], max_tokens=1
                    )
                except Exception as exc:
                    logger.debug("生成通道预热失败（不影响评估）：%s", exc)

            try:
                from app.core.embedding import build_embedder

                warm_embedder = build_embedder(cfg, model=warm_kb.embed_model)
                await warm_embedder.embed_query("预热")
                logger.info("预热完成（BM25 / 生成 / 向量模型均已载入）")
            except Exception as exc:
                logger.debug("向量模型预热失败（不影响评估）：%s", exc)

    for mode in mode_list:
        for index, sample in enumerate(samples, start=1):
            row = await _evaluate_one(
                db=db,
                sample=sample,
                mode=mode,
                kb_id=kb_id,
                provider_name=provider_name,
                top_k=top_k,
                use_llm_judge=use_llm_judge,
                judge_provider=judge_provider,
                offline_contexts=offline_contexts,
            )
            results.append(row)
            if progress_cb is not None:
                try:
                    maybe = progress_cb(mode, index, total)
                    if hasattr(maybe, "__await__"):
                        await maybe  # type: ignore[misc]
                except Exception as exc:
                    logger.debug("进度回调异常（已忽略）：%s", exc)

    # ---------------- 组装报告 ----------------
    aggregate: dict[str, dict[str, Any]] = {}
    timings: dict[str, Any] = {}
    for mode in mode_list:
        rows = [r for r in results if r.mode == mode]
        aggregate[mode] = _aggregate_mode(rows)
        timings[mode] = _aggregate_timings(rows)

    gold_modes: dict[str, int] = {}
    for row in results:
        gold_modes[row.gold_mode] = gold_modes.get(row.gold_mode, 0) + 1

    failed = sum(1 for r in results if r.error)
    completed = sum(1 for r in results if not r.error)

    report = EvalReport(
        kb_id=kb_id,
        kb_name=kb_name,
        dataset_name="",
        modes=mode_list,
        provider=provider_label,
        model=model_label,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        sample_count=total,
        completed_count=completed,
        failed_count=failed,
        top_k=top_k,
        use_llm_judge=bool(use_llm_judge and judge_provider is not None),
        offline_contexts=offline_contexts,
        aggregate=aggregate,
        samples=results,
        timings=timings,
        notes=_build_notes(
            gold_modes=gold_modes,
            failed=failed,
            use_llm_judge=bool(use_llm_judge and judge_provider is not None),
            offline_contexts=offline_contexts,
        ),
    )
    return report


def summarize(report: EvalReport) -> str:
    """把报告渲染成一张纯文本对照表，供 CLI / 日志直接打印。

    表格是给**人**看的，列顺序固定为
    「命中率 / 召回 / 精确 / MRR / NDCG / 引用覆盖 / 引用有效 / 字面重合」，
    正好对应论文里检索质量与生成质量两张表。缺失（None）显示为 `-`，
    而不是 0，避免把"没测"读成"得了 0 分"。
    """
    columns: list[tuple[str, str]] = [
        ("hit_rate", "命中率"),
        ("recall", "召回"),
        ("precision", "精确"),
        ("mrr", "MRR"),
        ("ndcg", "NDCG"),
        ("citation_coverage", "引用覆盖"),
        ("citation_validity", "引用有效"),
        ("answer_overlap", "字面重合"),
        ("judge_relevance", "裁判相关"),
        ("judge_faithfulness", "裁判忠实"),
    ]

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)

    lines: list[str] = []
    header = f"{'模式':<14}" + "".join(f"{label:>10}" for _, label in columns)
    lines.append(header)
    lines.append("-" * len(header))
    for mode in report.modes:
        agg = report.aggregate.get(mode, {})
        cells = []
        for key, _ in columns:
            block = agg.get(key) or {}
            cells.append(fmt(block.get("mean")))
        lines.append(f"{mode:<14}" + "".join(f"{c:>10}" for c in cells))

    # n 行单独列出：没有 sample 数，上面那串均值无法判断可信度
    lines.append("")
    lines.append("各项指标的有效样本数 n：")
    for mode in report.modes:
        agg = report.aggregate.get(mode, {})
        counts = " ".join(f"{label}={ (agg.get(key) or {}).get('n', 0) }" for key, label in columns)
        lines.append(f"  {mode}: {counts}")

    lines.append("")
    lines.append("耗时（毫秒，均值 / 最小 / 最大）：")
    for mode in report.modes:
        t = report.timings.get(mode, {})
        ret = t.get("retrieval_ms") or {}
        gen = t.get("generation_ms") or {}
        e2e = t.get("elapsed_ms") or {}
        lines.append(
            f"  {mode:<14} 检索 {fmt(ret.get('mean'))} / {fmt(ret.get('min'))} / {fmt(ret.get('max'))}"
            f"    生成 {fmt(gen.get('mean'))} / {fmt(gen.get('min'))} / {fmt(gen.get('max'))}"
            f"    端到端 {fmt(e2e.get('mean'))} / {fmt(e2e.get('min'))} / {fmt(e2e.get('max'))}"
        )
    return "\n".join(lines)


__all__ = [
    "ANSWER_OVERLAP_THRESHOLD",
    "DEFAULT_MODES",
    "EvalReport",
    "EvalSampleResult",
    "JudgeReport",
    "ProgressCallback",
    "keyword_hits",
    "resolve_gold",
    "run_evaluation",
    "summarize",
]
