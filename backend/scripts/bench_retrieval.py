"""隔离实验：三种检索模式在**排除生成干扰**后的纯检索耗时。

起因：评估报告里 hybrid 的平均检索耗时（2136ms）远高于 dense_only（354ms），
但 hybrid 只是比 dense_only 多跑了一路「0ms 的 BM25」——数字对不上，
怀疑是 Ollama 在 embedding 模型与生成模型之间反复换入换出显存造成的。

这个脚本不调用生成，只反复跑 retrieve()，并打印 trace 里的分阶段耗时，
用来判断时间究竟花在「向量化查询」还是「BM25 建索引」上。

用法：
    .venv\\Scripts\\python.exe scripts\\bench_retrieval.py --kb-id <id> --rounds 6
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.retrieval import RetrievalTrace, retrieve  # noqa: E402
from app.core.settings_service import SettingsService  # noqa: E402
from app.db.base import session_scope  # noqa: E402
from app.db.models import KnowledgeBase  # noqa: E402

QUERIES = [
    "RRF 融合公式里的 k 起什么作用？",
    "切片重叠为什么必要？",
    "BM25 有什么缺点？",
]


async def bench(kb_id: str, rounds: int, modes: list[str]) -> int:
    with session_scope() as db:
        cfg = SettingsService(db).effective()
        kb = db.get(KnowledgeBase, kb_id)
        if kb is None:
            print(f"[错误] 知识库不存在：{kb_id}")
            return 1
        print(f"知识库：{kb.name}  embed_model={kb.embed_model}")

        results: dict[str, list[int]] = {m: [] for m in modes}
        stage_totals: dict[str, dict[str, list[int]]] = {m: {} for m in modes}

        for mode in modes:
            print(f"\n=== 模式 {mode} ===")
            for i in range(rounds):
                query = QUERIES[i % len(QUERIES)]
                trace = RetrievalTrace(query=query, mode=mode)
                hits = await retrieve(db, cfg, kb, query, mode=mode, trace=trace)

                stages = {s["stage"]: s.get("ms") for s in trace.stages if "ms" in s}
                for name, ms in stages.items():
                    stage_totals[mode].setdefault(name, []).append(int(ms))

                results[mode].append(trace.elapsed_ms)
                stage_str = "  ".join(f"{k}={v}ms" for k, v in stages.items())
                print(f"    #{i + 1}  总 {trace.elapsed_ms:>5}ms   {stage_str}   命中 {len(hits)} 条")

        print("\n" + "=" * 74)
        print("  纯检索耗时汇总（不含任何生成调用）")
        print("=" * 74)
        print(f"{'模式':<14}{'均值':>9}{'中位数':>9}{'最小':>9}{'最大':>9}   分阶段均值")
        print("-" * 74)
        for mode in modes:
            vals = results[mode]
            if not vals:
                continue
            stages = stage_totals[mode]
            stage_str = "  ".join(
                f"{k}={statistics.mean(v):.0f}ms" for k, v in stages.items() if v
            )
            print(
                f"{mode:<14}{statistics.mean(vals):>9.0f}{statistics.median(vals):>9.0f}"
                f"{min(vals):>9}{max(vals):>9}   {stage_str}"
            )

        # 关键判断：如果首次明显慢于后续，说明是「模型加载/换页」而非算法开销
        print("\n  逐次趋势（用于判断是否为一次性加载开销）：")
        for mode in modes:
            vals = results[mode]
            if vals:
                first, rest = vals[0], vals[1:]
                avg_rest = statistics.mean(rest) if rest else 0
                verdict = "首次含模型加载" if first > avg_rest * 2 else "各次相近，无一次性开销"
                print(f"    {mode:<14} 第1次 {first:>5}ms   后续均值 {avg_rest:>7.0f}ms   -> {verdict}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="纯检索耗时隔离实验")
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--rounds", type=int, default=6, help="每种模式跑几轮")
    parser.add_argument("--modes", default="hybrid,dense_only,sparse_only")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    return asyncio.run(bench(args.kb_id, args.rounds, modes))


if __name__ == "__main__":
    raise SystemExit(main())
