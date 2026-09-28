"""评估模块：把「检索好不好、回答可不可信」变成可复现的数字。

## 为什么单独成一个包，而不是塞进 retrieval / rag

评估要同时用到检索层（chunk_id 序列）和生成层（答案、引用），
放在任何一侧都会造成反向依赖。独立成包之后：

  * `metrics` 只依赖标准库 + jieba，是**纯函数**，可以脱离数据库、脱离
    模型、脱离网络做单元测试 —— 论文里的指标公式能被逐条验证；
  * `dataset` 管数据（合成评估集 / 内置示例集）；
  * `runner` 管流程（调 `rag.engine.answer_once` 跑完整链路）。

这样「指标定义」和「跑实验」解耦：换了实验流程，指标不用重测；
改了指标实现，历史报告里的公式仍能从 docstring 里查到出处。

## 降级策略（贯穿整个包的设计原则）

实际使用中没有人工标注：`gold_chunk_ids` 往往拿不到。所以每一层都定义了
「没有标准答案时怎么退而求其次」，并且把「这次用的是哪一档」记进报告里
（`EvalSampleResult.gold_mode`）。**评估结果必须自带可信度标注**，
否则把降级数据当成 gold 数据写进论文就是学术事故。
"""

from __future__ import annotations

from app.core.eval.dataset import (
    EvalSample,
    build_dataset_from_kb,
    builtin_datasets,
    get_builtin_dataset,
    list_datasets,
    load_dataset,
    save_dataset,
)
from app.core.eval.metrics import (
    aggregate_metrics,
    answer_context_overlap,
    char_ngram_overlap,
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
from app.core.eval.runner import (
    EvalReport,
    EvalSampleResult,
    run_evaluation,
)

__all__ = [
    "EvalReport",
    "EvalSample",
    "EvalSampleResult",
    "aggregate_metrics",
    "answer_context_overlap",
    "build_dataset_from_kb",
    "builtin_datasets",
    "char_ngram_overlap",
    "citation_coverage",
    "citation_validity",
    "get_builtin_dataset",
    "hit_rate_at_k",
    "judge_answer_relevance",
    "judge_faithfulness",
    "list_datasets",
    "load_dataset",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "run_evaluation",
    "save_dataset",
    "token_f1",
]
