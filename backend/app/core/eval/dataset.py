"""评估数据集：格式定义、读写、LLM 自动合成、内置示例集。

## 为什么要有「合成评估集」这条路

人工标注一个检索评估集（每个问题配上 gold chunk）在本项目的时间尺度下不现实：
标 50 条就得把整库文档重读一遍，而且标注者的判断无法复现。

业界常用的替代做法是**反向合成**：先从语料里抽一个片段，
再让 LLM 基于这个片段写一个「该片段能回答的问题」和标准答案。
这样：

  * gold 是**天然正确**的 —— 问题就是从这个片段生成的，
    所以这个片段的 chunk_id 必然是标准答案来源；
  * 规模可以随便扩，成本只是几次本地推理；
  * 可复现：固定随机种子 + 记录 seed 和生成模型，
    换台机器重跑能得到同一批问题。

**必须写进论文的局限**：合成问题与源片段存在**词汇高度重叠**，
会让检索指标系统性偏乐观（召回率天然高），且无法覆盖「用户会怎么问」
的真实分布。因此合成集适合做**消融对比**（哪个模式更好），
不适合宣称「系统召回率 92%」。论文里应把合成集与人工抽检结合报告。

## 文件格式约定

数据集就是一个 JSON 文件，顶层是对象（不用裸数组，方便加元信息）：

    {
      "name": "sample_ml",
      "description": "机器学习基础概念（内置，无需 LLM）",
      "version": 1,
      "created_at": "2026-01-01T00:00:00Z",
      "generator": "builtin",
      "samples": [
        {
          "question": "什么是过拟合？",
          "gold_answer": "……",
          "gold_chunk_ids": ["abc123"],       // 可选，有则最准
          "gold_keywords": ["过拟合", "训练集"], // 可选，无 chunk_id 时的回退
          "context": "……",                     // 可选，离线自检用
          "notes": "……"
        }
      ]
    }

`gold_chunk_ids` 与 `gold_keywords` 至少给一个，否则这条样本的检索指标
无定义（加载时会跳过并计数）。二者都给时以 chunk_id 为准。
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.eval._stopwords import ALL_STOPWORDS
from app.db.models import Chunk, KnowledgeBase

logger = logging.getLogger(__name__)

# 数据集版本号：格式将来若有不兼容改动，靠它区分老文件
DATASET_VERSION = 1

# 内置数据集名 -> 描述。CLI / 接口列数据集时直接用
_BUILTIN_NAMES = ("sample_ml", "sample_retrieval")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ======================================================================
# 数据结构
# ======================================================================
@dataclass(slots=True)
class EvalSample:
    """一条评估样本。

    `gold_chunk_ids` 与 `gold_keywords` 是**两级标准答案**，
    runner 会按「有 chunk_id 用 chunk_id，没有才用关键词」的顺序降级，
    并把实际用到的那一档写进报告（`EvalSampleResult.gold_mode`）。
    评估结果必须自带可信度标注，否则降级数据会被误当成人工标注数据。

    `context` 是这条样本的原始参考片段，仅供**不连知识库**的自检使用
    （例如没启动 Ollama / 没建库时验证整条评估链路）。

    `source_chunk_id` 记录合成来源，便于人工复核抽查「问题是不是真的
    由这个片段回答得了」。
    """

    question: str
    gold_answer: str = ""
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_keywords: list[str] = field(default_factory=list)
    context: str = ""
    source_chunk_id: str | None = None
    notes: str = ""

    @property
    def has_gold(self) -> bool:
        """是否有可用于检索评估的标准答案。"""
        return bool(self.gold_chunk_ids or self.gold_keywords)

    def to_dict(self) -> dict[str, Any]:
        """转 JSON。空字段省略，让数据集文件保持人类可读。"""
        data: dict[str, Any] = {"question": self.question}
        if self.gold_answer:
            data["gold_answer"] = self.gold_answer
        if self.gold_chunk_ids:
            data["gold_chunk_ids"] = list(self.gold_chunk_ids)
        if self.gold_keywords:
            data["gold_keywords"] = list(self.gold_keywords)
        if self.context:
            data["context"] = self.context
        if self.source_chunk_id:
            data["source_chunk_id"] = self.source_chunk_id
        if self.notes:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalSample | None:
        """从 dict 构造。缺少 question 时返回 None（调用方负责跳过并计数）。"""
        question = str(raw.get("question") or "").strip()
        if not question:
            return None
        return cls(
            question=question,
            gold_answer=str(raw.get("gold_answer") or "").strip(),
            gold_chunk_ids=[str(x) for x in (raw.get("gold_chunk_ids") or []) if str(x).strip()],
            gold_keywords=[str(x).strip() for x in (raw.get("gold_keywords") or []) if str(x).strip()],
            context=str(raw.get("context") or "").strip(),
            source_chunk_id=(str(raw["source_chunk_id"]) if raw.get("source_chunk_id") else None),
            notes=str(raw.get("notes") or "").strip(),
        )


# ======================================================================
# 内置示例数据集（不依赖 LLM，无 Ollama 也能跑通全流程）
# ======================================================================
# 每条都带 context：这是「离线自检」的关键 —— 即使没建知识库，
# `run_evaluation(kb_id=..., offline_contexts=True)` 也能算出
# citation_coverage / citation_validity / answer_context_overlap 三组生成指标，
# 用来验证报告管线本身没坏。检索指标仍然需要真的知识库。
_SAMPLE_ML: tuple[EvalSample, ...] = (
    EvalSample(
        question="什么是过拟合？怎么缓解？",
        gold_answer=(
            "过拟合指模型在训练集上表现很好、在未见过的新数据上表现明显变差，"
            "本质是把训练数据里的噪声和偶然规律也学了进去。"
            "常用的缓解手段有正则化、Dropout、早停、数据增强和交叉验证，"
            "或者在保证性能的前提下降低模型复杂度。"
        ),
        gold_keywords=["过拟合", "训练集", "泛化", "正则化", "Dropout"],
        context=(
            "过拟合（Overfitting）是指模型在训练数据上表现很好，但在未见过的测试数据上表现较差的现象。"
            "其本质是模型把训练数据中的噪声和偶然规律也学到了，没有抓住真正的普遍规律。"
            "判断过拟合的典型信号是：训练集损失持续下降，而验证集损失在某个点之后开始上升。"
            "缓解过拟合的常用方法包括：L1/L2 正则化、Dropout、早停（Early Stopping）、"
            "数据增强、交叉验证，以及适当降低模型复杂度。"
        ),
        notes="内置基础题。关键词回退模式下近似唯一命中，检索指标参考价值有限。",
    ),
    EvalSample(
        question="监督学习和无监督学习的区别是什么？",
        gold_answer=(
            "区别在于训练数据是否带标签。监督学习使用带标签的数据，"
            "目标是学习输入到输出的映射，典型任务有分类和回归；"
            "无监督学习使用无标签数据，目标是发现数据内部的结构，"
            "典型任务有聚类、降维和异常检测。"
        ),
        gold_keywords=["监督学习", "无监督学习", "标签", "聚类", "分类"],
        context=(
            "监督学习使用带标签的数据进行训练，每个样本都有对应的正确答案，"
            "模型学习从输入到输出的映射关系，典型任务包括分类和回归。"
            "无监督学习使用不带标签的数据，模型需要自己发现数据中的结构和模式，"
            "典型任务包括聚类、降维和异常检测。"
            "半监督学习介于两者之间，同时利用少量带标签数据和大量无标签数据。"
        ),
    ),
    EvalSample(
        question="梯度下降法是怎么工作的？学习率有什么影响？",
        gold_answer=(
            "梯度下降通过计算损失函数对参数的梯度，沿梯度反方向更新参数，逐步降低损失。"
            "学习率决定每一步的更新幅度：太大会在最优解附近震荡甚至发散，"
            "太小则收敛过慢、容易停在局部最优。"
        ),
        gold_keywords=["梯度下降", "学习率", "损失函数", "收敛"],
        context=(
            "梯度下降法是一种一阶迭代优化算法：计算损失函数对每个参数的偏导数（梯度），"
            "然后沿梯度的反方向按学习率缩放后更新参数，使损失逐步下降。"
            "学习率是最关键的超参数：学习率过大会导致参数更新幅度过大，"
            "在最优解附近震荡甚至发散；学习率过小则收敛速度很慢，"
            "训练时间长且容易停留在局部最优。实践中常用学习率衰减或自适应优化器"
            "（如 Adam）来平衡收敛速度与稳定性。"
        ),
    ),
    EvalSample(
        question="为什么需要 RAG（检索增强生成）？",
        gold_answer=(
            "RAG 在生成之前先从外部知识库检索相关资料，再让模型基于资料作答。"
            "它缓解了大模型的知识过时问题、无法访问私有数据的问题，"
            "并因为答案有出处而降低了幻觉，让回答可溯源。"
        ),
        gold_keywords=["RAG", "检索增强", "幻觉", "知识库", "大语言模型"],
        context=(
            "检索增强生成（Retrieval-Augmented Generation，RAG）在生成回答之前，"
            "先从外部知识库中检索与问题相关的文档片段，再把这些片段作为上下文"
            "交给大语言模型生成答案。这样可以缓解三类问题：一是大模型训练数据"
            "存在知识截止时间，无法回答最新信息；二是企业私有数据不在训练集中；"
            "三是大模型容易产生幻觉，编造看似合理但错误的内容。"
            "由于答案基于检索到的原文，RAG 还能提供引用溯源，便于用户核对。"
        ),
    ),
    EvalSample(
        question="BM25 和稠密向量检索各有什么优缺点？",
        gold_answer=(
            "BM25 属于稀疏检索，依赖词频与逆文档频率，擅长精确术语匹配、速度快、可解释，"
            "但无法处理同义改写；稠密向量检索把文本映射到语义空间，能召回语义相近的表达，"
            "但需要嵌入模型、对专有名词和数字不敏感。两者互补，因此常用混合检索加融合排序。"
        ),
        gold_keywords=["BM25", "稀疏检索", "稠密检索", "向量", "混合检索"],
        context=(
            "BM25 是稀疏检索的代表，基于词频与逆文档频率给文档打分，"
            "优点是精确匹配术语时效果好、计算快、结果可解释，"
            "缺点是无法理解同义词和改写表达，词表不匹配就会漏召。"
            "稠密向量检索先用嵌入模型把文本编码成向量，再按向量相似度检索，"
            "优点是可以召回语义相近但用词不同的内容，"
            "缺点是需要额外的嵌入模型、对罕见专有名词和数字不敏感，"
            "且相似度分数不如词频那样直观可解释。"
            "两者互补，因此工程上常采用 BM25 与向量检索并行的混合检索，"
            "再用 RRF 之类的融合算法合并排序结果。"
        ),
    ),
    EvalSample(
        question="精确率、召回率和 F1 值分别衡量什么？",
        gold_answer=(
            "精确率衡量预测为正的样本里有多少真的是正类，关注「报得准不准」；"
            "召回率衡量真正的正类里有多少被找出来，关注「有没有漏」；"
            "F1 是精确率与召回率的调和平均，用来在两者之间取平衡。"
        ),
        gold_keywords=["精确率", "召回率", "F1", "调和平均"],
        context=(
            "精确率（Precision）= 真正例 /（真正例 + 假正例），衡量模型预测为正的样本中"
            "有多少确实是正类，关注的是「报出来的准不准」。"
            "召回率（Recall）= 真正例 /（真正例 + 假负例），衡量所有真正的正类中"
            "有多少被模型找了出来，关注的是「有没有漏掉」。"
            "两者往往此消彼长，F1 值是精确率与召回率的调和平均，"
            "对较小的一方更敏感，适合在需要综合权衡时作为单一指标使用。"
        ),
    ),
)

_SAMPLE_RETRIEVAL: tuple[EvalSample, ...] = (
    EvalSample(
        question="RRF 融合公式里的 k 起什么作用？",
        gold_answer=(
            "k 是倒数排名融合的平滑常数，出现在分母 (k + rank) 里。"
            "k 越大，头部名次之间的差距被压得越平，融合结果越平滑、越依赖两路的一致性；"
            "k 越小越突出头部结果。论文里常用 k=60。"
        ),
        gold_keywords=["RRF", "融合", "平滑", "名次"],
        context=(
            "倒数排名融合（Reciprocal Rank Fusion）的公式是："
            "score(d) = Σ_r weight_r / (k + rank_r(d))。"
            "k 是平滑常数：k 越大，不同名次之间的分数差距越小，"
            "排序越平滑，越需要多个检索器同时给出较高名次才能排到前面；"
            "k 越小，第 1 名与第 2 名的差距越大，头部结果的优势越明显。"
            "工业界与论文中普遍取 k = 60。"
        ),
    ),
    EvalSample(
        question="切片重叠（chunk overlap）为什么必要？",
        gold_answer=(
            "固定长度切片会把本该连在一起的语义截断在边界上，"
            "导致关键信息被切成两半、任何一片都不完整。"
            "让相邻切片保留一段重叠，可以让被切断的语义至少在某一片里保持完整，"
            "代价是存储和索引量增加，所以重叠比例一般取切片长度的 10%~20%。"
        ),
        gold_keywords=["切片", "重叠", "边界", "语义"],
        context=(
            "文档切片时，如果严格按固定长度切分，一个完整的语义单元"
            "（比如一个定义、一段推导）很可能正好跨在两片之间被截断，"
            "检索到其中任意一片都拿不到完整信息。"
            "相邻切片之间保留一定重叠可以让被截断的内容至少在某一篇里保持完整。"
            "但重叠过大会显著增加索引体积和检索时的重复内容，"
            "实践中重叠长度通常取切片长度的 10%~20%。"
        ),
    ),
)

_BUILTIN: dict[str, tuple[str, tuple[EvalSample, ...]]] = {
    "sample_ml": ("机器学习基础概念（内置示例，无需 LLM 与知识库）", _SAMPLE_ML),
    "sample_retrieval": ("检索与 RAG 原理（内置示例，偏系统设计）", _SAMPLE_RETRIEVAL),
}


def _clone(sample: EvalSample) -> EvalSample:
    """深拷贝一条样本。

    不能只 `replace(sample)`：那会让新旧样本**共享** `gold_keywords` 等可变列表，
    调用方 append 一下就把内置常量改了，第二次跑评估结果会不一样。
    内置数据集必须是一次性常量，所有可变字段都得复制。
    """
    return EvalSample(
        question=sample.question,
        gold_answer=sample.gold_answer,
        gold_chunk_ids=list(sample.gold_chunk_ids),
        gold_keywords=list(sample.gold_keywords),
        context=sample.context,
        source_chunk_id=sample.source_chunk_id,
        notes=sample.notes,
    )


def builtin_datasets() -> dict[str, list[EvalSample]]:
    """内置示例数据集。返回副本，避免调用方改到模块级常量。"""
    return {name: [_clone(s) for s in samples] for name, (_, samples) in _BUILTIN.items()}


def get_builtin_dataset(name: str = "sample_ml") -> list[EvalSample]:
    """按名字取内置数据集；名字不存在则抛 KeyError（错误信息带可选值）。

    用抛异常而不是返回空列表：空数据集跑出来的报告「所有指标都是 None」，
    很容易被误读成「系统表现差」，不如直接报错。
    """
    entry = _BUILTIN.get(name)
    if entry is None:
        available = "、".join(_BUILTIN)
        raise KeyError(f"没有名为 `{name}` 的内置数据集，可选：{available}")
    return [_clone(s) for s in entry[1]]


def list_datasets(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """列出可用数据集：内置的 + `data/eval/datasets/` 下的 JSON 文件。

    `data_dir` 不传则用配置里的 data_dir。返回结构可直接序列化给前端。
    """
    out: list[dict[str, Any]] = [
        {
            "name": name,
            "source": "builtin",
            "description": desc,
            "sample_count": len(samples),
            "path": None,
        }
        for name, (desc, samples) in _BUILTIN.items()
    ]

    directory = datasets_dir(data_dir)
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            info: dict[str, Any] = {
                "name": path.stem,
                "source": "file",
                "description": "",
                "sample_count": 0,
                "path": str(path),
            }
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                info["description"] = str(payload.get("description") or "")
                info["sample_count"] = len(payload.get("samples") or [])
                info["generator"] = payload.get("generator")
            except Exception as exc:
                # 坏文件不该让整个列表接口 500，标出来让用户自己看
                info["description"] = f"（读取失败：{type(exc).__name__}: {exc}）"
                info["sample_count"] = 0
            out.append(info)
    return out


def datasets_dir(data_dir: Path | None = None) -> Path:
    """数据集存放目录：`<data_dir>/eval/datasets`。"""
    if data_dir is None:
        from app.config import get_settings

        data_dir = get_settings().data_dir
    return Path(data_dir) / "eval" / "datasets"


# ======================================================================
# 读写
# ======================================================================
def save_dataset(
    path: str | Path,
    samples: Sequence[EvalSample],
    *,
    name: str | None = None,
    description: str = "",
    generator: str = "manual",
    extra: dict[str, Any] | None = None,
) -> Path:
    """把样本写成 JSON 数据集，返回落盘路径（自动建父目录）。

    顶层显式写入 `name / description / version / generator / created_at`：
    论文里引用一份评估数据时必须能说清「它是谁、什么时候、怎么生成的」，
    这些元信息如果只存在于文件名和作者记忆里，半年后自己都复现不了。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "name": name or target.stem,
        "description": description,
        "version": DATASET_VERSION,
        "generator": generator,
        "created_at": _utcnow_iso(),
        "sample_count": len(samples),
        "samples": [s.to_dict() for s in samples],
    }
    if extra:
        payload.update(extra)
    # ensure_ascii=False：中文问题必须是可读的，否则人工核对时要先解码
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def load_dataset(path: str | Path) -> list[EvalSample]:
    """从 JSON 文件读数据集。

    只支持 `{"samples": [...]}` 对象格式；裸数组文件（[{"question": ...}]）
    也容忍 —— 手写数据集时很多人会直接写数组，没必要为此报错。

    单条样本格式错误（缺 question / 整个是字符串）时**跳过并打日志**，
    不让一条脏数据毁掉整份数据集。真正的问题是「一条都没读出来」，
    那时抛 ValueError，避免拿着空集跑出一份全是 None 的报告。
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"数据集文件不存在：{source}")

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"数据集不是合法 JSON：{source}（{exc}）") from exc

    if isinstance(payload, dict):
        raw_samples = payload.get("samples")
    elif isinstance(payload, list):
        raw_samples = payload
    else:
        raise ValueError(f"数据集顶层必须是对象或数组：{source}")

    if not isinstance(raw_samples, list):
        raise ValueError(f"数据集缺少 `samples` 数组：{source}")

    samples: list[EvalSample] = []
    skipped = 0
    for raw in raw_samples:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        sample = EvalSample.from_dict(raw)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    if skipped:
        logger.warning("数据集 %s 中有 %d 条样本格式不合法，已跳过", source, skipped)
    if not samples:
        raise ValueError(f"数据集里没有一条可用的样本：{source}")
    return samples


def resolve_dataset(
    name_or_path: str,
    *,
    data_dir: Path | None = None,
) -> tuple[list[EvalSample], str]:
    """按名字或路径取数据集，返回 `(样本, 实际用到的名字)`。

    解析顺序（先具体后抽象，符合「用户给了路径就用路径」的直觉）：
      1. 存在的文件路径；
      2. `<data_dir>/eval/datasets/<name>.json`；
      3. 内置数据集名。

    CLI 的 `--dataset sample_ml` 和 `--dataset data/eval/datasets/my.json`
    因此都能用同一个参数表达。
    """
    candidate = Path(name_or_path)
    if candidate.is_file():
        return load_dataset(candidate), candidate.stem

    stored = datasets_dir(data_dir) / f"{name_or_path}.json"
    if stored.is_file():
        return load_dataset(stored), name_or_path

    if name_or_path in _BUILTIN:
        return get_builtin_dataset(name_or_path), name_or_path

    available = "、".join(sorted(set(_BUILTIN) | {p.stem for p in datasets_dir(data_dir).glob("*.json")}))
    raise KeyError(f"找不到数据集 `{name_or_path}`。可用：{available}")


# ======================================================================
# 自动构建（合成评估集）
# ======================================================================
_QUESTION_SYSTEM = """你是一个检索评估数据集的构建助手。

给定一段文档片段，请生成一个**只能靠这段片段回答**的问题，以及标准答案。

要求：
1. 问题要像一个真实用户会问的问题，不要提到「这段文字」「上述片段」。
2. 问题必须具体到能唯一指向这段片段的内容，避免「这篇文章讲了什么」这类泛问。
3. 标准答案必须完全来自片段，不要补充片段之外的知识，不要臆测。
4. 用中文。

只输出下面两行，不要任何其他内容：
问题：<一句话问题>
答案：<2~4 句话的标准答案>"""

# 生成结果的解析：允许模型把标签写成「问题:」「Q:」「标准答案：」等变体
_QUESTION_LINE = re.compile(r"^\s*(?:问题|问句|question|q)\s*[:：]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_ANSWER_LINE = re.compile(
    r"^\s*(?:答案|标准答案|回答|answer|a)\s*[:：]\s*(.+)$", re.IGNORECASE | re.MULTILINE
)
# 关键词的最小长度。1 字词（「化」「梯」「网」）在别的片段里撞车的概率太高，
# 会把大量无关片段判成相关，所以只保留 2 字及以上的词。
# 没有上限约束：jieba 在精确模式下不会吐超长词，而拼接方案已被否决（见 docstring）。
_KEYWORD_MIN_LEN = 2
# 只由标点/空白组成的 token（jieba 会把它们原样吐出来）
_PUNCT_ONLY = re.compile(r"^[^\w\u4e00-\u9fff]+$")
# 关键词过滤用的停用词：与指标侧的 token 过滤共用同一份表，
# 避免两边标准漂移（详见 eval/_stopwords 模块）
_KEYWORD_STOP = ALL_STOPWORDS


def extract_keywords(text: str, *, limit: int = 8) -> list[str]:
    """从片段里抽关键词，作为 `gold_chunk_ids` 缺失时的命中判定依据。

    做法：jieba 精确分词 -> 丢掉停用词与纯标点 -> 只要 2 字以上的词
    -> 去重后按长度降序、再按出现顺序取前 `limit` 个。

    ## 为什么用 jieba 而不是正则捞连续汉字

    第一版用的是 ``[\\u4e00-\\u9fff]{2,8}`` 这类正则，结果很糟：
    中文没有空格，``{2,8}`` 会贪婪吞下「过拟合是指模型在」这种跨词碎块，
    拿它去片段里做子串匹配永远命中不了，等于关键词回退完全失效。
    jieba 让词边界由词典决定，抽出来的是能用的词。

    ## 试过但被否决的两个方案（记录下来避免重复踩坑）

    1. **相邻词拼接**：想把 jieba 切碎的「过 / 拟合」拼回「过拟合」。
       实测会因为超长候选占优而产出「模型训练数据」「拟合指模型」这类
       跨短语碎块 —— 比不拼还糟。而且**没必要**：匹配用的是子串包含，
       「拟合」本来就能在「过拟合」里命中。
    2. **用 jieba 词典校验候选词**：即只保留 ``jieba.dt.FREQ`` 里有的词。
       实测「过拟合」「正则化」「机器学习」「梯度下降」在 jieba 主词典里
       **全都不存在**（FREQ=0），这个校验会把最该留下的领域术语全删掉。

    结论：**不要试图"猜术语"**。抽词只是为了拿到能被子串匹配的锚点，
    半截词（「拟合」「正则」）一样有效。判据越简单越不容易出错。

    ## 排序依据

    按**词长降序**、再按**首次出现位置升序**。词长优先是因为中文里长词
    更具体（「神经网络」比「网络」信息量大）；位置优先是为了可复现
    （同样输入永远得到同样列表）。

    ## 局限

    这是**粗粒度**回退判据：抽出的词里必然混有「模型」「方法」这类通用词，
    导致关键词命中偏宽松。评估报告会把这类样本标为
    ``gold_mode="keywords"`` 并附说明，使用者据此打折看待指标。
    """
    if not text or not text.strip():
        return []

    # 延迟导入：dataset 是接口 / CLI 都会加载的模块，而 jieba 初始化要 1 秒左右，
    # 只在真正需要抽关键词时才付这个代价
    from app.core.eval.metrics import _ensure_jieba

    jieba = _ensure_jieba()
    candidates: dict[str, int] = {}  # 词 -> 首次出现位置
    for position, raw in enumerate(jieba.lcut(text)):
        token = raw.strip()
        if len(token) < _KEYWORD_MIN_LEN:
            continue
        if token in _KEYWORD_STOP or _PUNCT_ONLY.match(token):
            continue
        candidates.setdefault(token, position)

    ordered = sorted(candidates.items(), key=lambda kv: (-len(kv[0]), kv[1]))
    return [token for token, _ in ordered[:limit]]


def _parse_generated(raw: str) -> tuple[str, str] | None:
    """从模型回复里抠出 (问题, 答案)。解析不出来返回 None。

    宁可丢掉一条，也不要往数据集里塞半成品 —— 合成集的质量问题
    在后续指标里会被误读成「系统缺陷」。
    """
    text = (raw or "").strip()
    if not text:
        return None

    q_match = _QUESTION_LINE.search(text)
    a_match = _ANSWER_LINE.search(text)
    question = q_match.group(1).strip() if q_match else ""
    answer = a_match.group(1).strip() if a_match else ""

    # 模型偷懒只输出一行问题、或者把答案换行续写时，做一次兜底
    if question and not answer:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        rest = [ln for ln in lines if not _QUESTION_LINE.match(ln)]
        answer = " ".join(rest).strip()
    if not question:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        question = lines[0] if lines else ""
        if len(lines) > 1 and not answer:
            answer = " ".join(lines[1:]).strip()

    question = question.strip().strip("\"'“”")
    answer = answer.strip().strip("\"'“”")
    # 太长的问题多半是模型开始自问自答，宁可不收
    if not question or len(question) > 200:
        return None
    return question, answer


async def build_dataset_from_kb(
    db: Session,
    kb_id: str,
    n_samples: int = 10,
    *,
    provider_name: str | None = None,
    seed: int | None = 42,
    min_chunk_chars: int = 80,
) -> list[EvalSample]:
    """从知识库随机抽 chunk，让 LLM 反向生成「问题 + 标准答案」的合成评估集。

    ## 参数

    * `seed`：固定随机种子。**默认给 42 而不是 None**，因为评估集必须可复现 ——
      同一份语料重跑得到同一批问题，指标才可比。要换一批就显式改 seed。
    * `min_chunk_chars`：过滤过短的片段。几十个字的残片生成不出像样的问题，
      而且这种片段本身就是切分器的产物（表格线、页眉），拿它当 gold 会污染指标。

    ## 容错

    某一条生成失败（LLM 超时、输出格式不对、片段无信息量）就**跳过继续**，
    绝不中断整轮。合成评估集是"有多少算多少"的活：
    要 10 条只成功 7 条，跑 7 条也比一条都没有强。

    ## 返回

    `EvalSample` 列表。`gold_chunk_ids` 就是被抽中的那个 chunk 的 id
    （所以 Gold 是天然正确的），同时用 `extract_keywords` 附一份
    `gold_keywords`，便于在换了知识库之后仍能复用一个数据集做粗评估。
    """
    from app.core.llm import ChatMessage, build_provider
    from app.core.settings_service import SettingsService

    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise ValueError(f"知识库不存在：{kb_id}")

    rows = db.execute(
        select(Chunk.id, Chunk.content)
        .where(Chunk.kb_id == kb_id)
        .order_by(Chunk.doc_id, Chunk.ordinal)
    ).all()
    pool = [(str(r[0]), str(r[1])) for r in rows if len(str(r[1]).strip()) >= min_chunk_chars]
    if not pool:
        raise ValueError(
            f"知识库 {kb.name} 里没有长度 >= {min_chunk_chars} 字符的片段，无法合成评估集"
        )

    rng = random.Random(seed)
    # 抽 min(n, 池子大小) 条。不重复抽样：同一个片段生成两个问题会
    # 让指标里出现两条高度相关的样本，等于给某个片段加权
    picked = rng.sample(pool, k=min(max(1, n_samples), len(pool)))

    cfg = SettingsService(db).effective()
    provider = build_provider(cfg, provider=provider_name)
    logger.info("开始合成评估集：知识库=%s，候选片段=%d，目标 %d 条", kb.name, len(pool), len(picked))

    samples: list[EvalSample] = []
    for index, (chunk_id, content) in enumerate(picked, start=1):
        # 单段截断：题目只要片段本身的信息量，塞整片反而让模型去概括表格
        snippet = content.strip()[:1500]
        messages = [
            ChatMessage(role="system", content=_QUESTION_SYSTEM),
            ChatMessage(role="user", content=f"【文档片段】\n{snippet}\n\n请生成问题和答案："),
        ]
        try:
            result = await provider.chat(messages, temperature=0.3, max_tokens=400)
        except Exception as exc:
            logger.warning("第 %d/%d 条合成失败（LLM 调用）：%s", index, len(picked), exc)
            continue

        parsed = _parse_generated(str(getattr(result, "content", "") or ""))
        if parsed is None:
            logger.warning("第 %d/%d 条合成失败（输出无法解析）", index, len(picked))
            continue

        question, answer = parsed
        samples.append(
            EvalSample(
                question=question,
                gold_answer=answer,
                gold_chunk_ids=[chunk_id],
                gold_keywords=extract_keywords(content),
                source_chunk_id=chunk_id,
                notes=f"由 {provider.name}/{provider.model} 从该片段合成（seed={seed}）",
            )
        )

    logger.info("合成完成：成功 %d / 目标 %d 条", len(samples), len(picked))
    if not samples:
        raise ValueError(
            "一条样本都没合成成功。请检查 LLM 通道是否可用（Ollama 是否启动、模型是否已拉取）。"
        )
    return samples


__all__ = [
    "DATASET_VERSION",
    "EvalSample",
    "build_dataset_from_kb",
    "builtin_datasets",
    "datasets_dir",
    "extract_keywords",
    "get_builtin_dataset",
    "list_datasets",
    "load_dataset",
    "resolve_dataset",
    "save_dataset",
]
