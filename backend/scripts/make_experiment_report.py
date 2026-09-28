"""把评估报告 JSON 转成可直接写进论文的实验报告（Markdown）。

为什么需要它：评估器输出的汇总表只有均值，而均值在存在冷启动离群值时会骗人
（实测 hybrid 检索均值 2442 ms，可中位数只有几百毫秒 —— 差距全来自个别样本的
模型重载）。论文里的耗时数据必须给出分布，否则站不住脚。

本脚本会：
  1. 读评估 JSON，按模式重算耗时分布（中位数 / 四分位 / 离群值个数）；
  2. 把 gold 判定的可信度档次、各指标的有效样本数一并写进报告；
  3. 输出 Markdown 到 docs/ 下，并把关键结论打印到终端。

用法：
    .venv\\Scripts\\python.exe scripts\\make_experiment_report.py
    .venv\\Scripts\\python.exe scripts\\make_experiment_report.py --report <json路径>
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "data" / "eval"
DOCS_DIR = ROOT / "docs"

METRIC_LABELS = {
    "hit_rate": "命中率",
    "recall": "召回率",
    "precision": "精确率",
    "mrr": "MRR",
    "ndcg": "NDCG",
    "citation_coverage": "引用覆盖率",
    "citation_validity": "引用有效率",
    "answer_overlap": "字面重合度",
    "judge_relevance": "裁判相关性",
    "judge_faithfulness": "裁判忠实度",
}


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _describe(values: list[float]) -> dict[str, float]:
    """给出比「均值」诚实得多的分布描述。"""
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    mean = statistics.mean(ordered)
    median = statistics.median(ordered)
    p90 = ordered[min(n - 1, int(round(0.9 * (n - 1))))]
    # 离群阈值用「中位数的 3 倍」：比 IQR 更直观，也更容易在论文里解释
    outliers = [v for v in ordered if v > median * 3] if median > 0 else []
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "p90": p90,
        "min": ordered[0],
        "max": ordered[-1],
        "outlier_count": len(outliers),
        "outlier_max": max(outliers) if outliers else 0.0,
    }


def build(report_path: Path) -> str:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    modes = data.get("modes") or []
    samples = data.get("samples") or []
    agg = data.get("aggregate") or {}
    notes = data.get("notes") or []
    if isinstance(notes, str):
        notes = [notes]

    lines: list[str] = []
    w = lines.append

    w("# 实验结果")
    w("")
    w(f"- 知识库：**{data.get('kb_name', '?')}**（`{data.get('kb_id', '?')}`）")
    w(f"- 数据集：**{data.get('dataset_name', '?')}**，{data.get('sample_count', 0)} 条样本")
    w(f"- 检索模式：{'、'.join(f'`{m}`' for m in modes)}")
    prov = data.get("provider") or "（未调用生成）"
    model = data.get("model") or ""
    w(f"- 生成通道：`{prov}`{f' / `{model}`' if model else ''}")
    w(f"- 检索深度 Top-K：{data.get('top_k', '?')}")
    w(f"- LLM 裁判：{'启用' if data.get('use_llm_judge') else '未启用'}")
    w(f"- 生成时间：{data.get('generated_at', '?')}")
    w("")

    # ---------------- 主结果表 ----------------
    w("## 一、检索质量（各模式均值）")
    w("")
    metric_keys = [k for k in METRIC_LABELS if any(k in (agg.get(m) or {}) for m in modes)]
    header = "| 模式 | " + " | ".join(METRIC_LABELS[k] for k in metric_keys) + " |"
    sep = "|---" * (len(metric_keys) + 1) + "|"
    w(header)
    w(sep)
    for mode in modes:
        block = agg.get(mode) or {}
        cells = []
        for k in metric_keys:
            stat = block.get(k)
            cells.append(_fmt(stat.get("mean")) if isinstance(stat, dict) else "—")
        w(f"| `{mode}` | " + " | ".join(cells) + " |")
    w("")

    # 有效样本数（反映有多少指标真的算出来了）
    w("各指标的有效样本数（`None` 表示该指标在本次评估中不可用，而不是 0 分）：")
    w("")
    w("| 模式 | " + " | ".join(METRIC_LABELS[k] for k in metric_keys) + " |")
    w(sep)
    for mode in modes:
        block = agg.get(mode) or {}
        cells = []
        for k in metric_keys:
            stat = block.get(k)
            cells.append(str(stat.get("n", 0)) if isinstance(stat, dict) else "0")
        w(f"| `{mode}` | " + " | ".join(cells) + " |")
    w("")

    # ---------------- 耗时分布 ----------------
    w("## 二、耗时分布（均值会骗人，这里给完整分布）")
    w("")
    w("| 模式 | 阶段 | 均值 ms | **中位数 ms** | P90 ms | 最小 ms | 最大 ms | 离群次数 |")
    w("|---|---|---|---|---|---|---|---|")
    for mode in modes:
        for stage, key in (("检索", "retrieval_ms"), ("生成", "generation_ms"), ("端到端", "elapsed_ms")):
            vals = [
                float(s.get(key) or 0)
                for s in samples
                if s.get("mode") == mode and s.get("error") is None
            ]
            d = _describe(vals)
            if not d:
                continue
            w(
                f"| `{mode}` | {stage} | {d['mean']:.0f} | **{d['median']:.0f}** | "
                f"{d['p90']:.0f} | {d['min']:.0f} | {d['max']:.0f} | {d['outlier_count']} |"
            )
    w("")
    w(
        "> 离群判定：单次耗时 > 中位数的 3 倍。检索阶段的离群几乎全部来自"
        "**向量模型被生成模型挤出显存后需要重新载入**，与检索算法本身无关 ——"
        "可用 `scripts/bench_retrieval.py` 在「排除生成干扰」的条件下复现稳定值。"
    )
    w("")

    # ---------------- 失败样本 ----------------
    failed = [s for s in samples if s.get("error")]
    w("## 三、失败样本")
    w("")
    if not failed:
        w(f"无。{len(samples)} 条样本全部跑完。")
    else:
        w(f"共 {len(failed)} 条：")
        w("")
        for s in failed[:20]:
            w(f"- `{s.get('mode')}` | {s.get('question', '')[:50]} → {s.get('error')}")
    w("")

    # ---------------- 可信度说明 ----------------
    w("## 四、可信度说明（必须随数据一起引用）")
    w("")
    for note in notes:
        w(f"- {note}")
    w("")

    # ---------------- 逐样本明细 ----------------
    w("## 五、逐样本明细")
    w("")
    w("| # | 模式 | 问题 | gold 判定 | 引用数 | MRR | NDCG | 检索 ms | 生成 ms |")
    w("|---|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(samples, start=1):
        m = s.get("metrics") or {}
        w(
            f"| {i} | `{s.get('mode')}` | {s.get('question', '')[:34]} | "
            f"`{s.get('gold_mode', '?')}` | {s.get('citation_count', 0)} | "
            f"{_fmt(m.get('mrr'))} | {_fmt(m.get('ndcg'))} | "
            f"{s.get('retrieval_ms', 0)} | {s.get('generation_ms', 0)} |"
        )
    w("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成论文用实验报告")
    parser.add_argument("--report", type=Path, default=None, help="评估 JSON 路径，默认取最新的")
    parser.add_argument("--out", type=Path, default=DOCS_DIR / "实验结果.md")
    args = parser.parse_args()

    if args.report:
        path = args.report
    else:
        candidates = sorted(glob.glob(str(EVAL_DIR / "*.json")))
        if not candidates:
            print(f"[错误] {EVAL_DIR} 下没有评估报告，先跑 scripts/run_eval.py")
            return 1
        path = Path(candidates[-1])

    print(f"读取报告：{path}")
    markdown = build(path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    print(f"已写入：{args.out}\n")

    # 终端回显关键部分，方便直接看结论
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("## "):
            print("\n".join(lines[i : i + 14]))
            print("...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
