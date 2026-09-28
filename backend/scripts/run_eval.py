"""命令行评估入口：把实验结果打印到终端，并落盘成 JSON。

用法（在 backend 目录下）：

    # 看有哪些数据集
    .venv\\Scripts\\python.exe scripts\\run_eval.py --list-datasets

    # 跑内置示例集，对比混合检索与纯向量检索
    .venv\\Scripts\\python.exe scripts\\run_eval.py --kb-id <知识库id> --dataset sample_ml --modes hybrid,dense_only

    # 先用 LLM 从知识库合成一份评估集再跑（需要 Ollama 可用）
    .venv\\Scripts\\python.exe scripts\\run_eval.py --kb-id <id> --build-dataset 20 --dataset-name synth_20

    # 不连知识库、不调模型，只验证"指标计算 + 报告序列化"这条管线
    .venv\\Scripts\\python.exe scripts\\run_eval.py --kb-id <id> --offline

## 为什么脚本和接口共用 `core.eval.runner`

评估代码一旦分叉（脚本一份、接口一份），论文里的数据就会逐渐和系统脱钩。
所以本脚本只做三件事：解析参数、调 runner、渲染输出 —— 不含任何指标逻辑。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 允许直接以脚本方式运行（与 scripts/check_env.py 保持一致的写法）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.eval.dataset import (  # noqa: E402
    build_dataset_from_kb,
    list_datasets,
    resolve_dataset,
    save_dataset,
)
from app.core.eval.runner import DEFAULT_MODES, EvalReport, run_evaluation, summarize  # noqa: E402

TITLE_WIDTH = 68


def _rule(char: str = "=") -> None:
    print(char * TITLE_WIDTH)


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。

    单独抽成函数是为了能在测试 / `--help` 里直接拿到它，
    不必执行 `main()`（那会连着连数据库、跑模型）。
    """
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="检索消融与生成质量评估：对每个样本 × 每个检索模式跑完整 RAG 链路，输出结构化报告。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/run_eval.py --list-datasets\n"
            "  python scripts/run_eval.py --kb-id abc123 --dataset sample_ml --modes hybrid,dense_only\n"
            "  python scripts/run_eval.py --kb-id abc123 --build-dataset 20 --dataset-name synth_20\n"
            "  python scripts/run_eval.py --kb-id abc123 --offline\n"
        ),
    )

    # ---- 必填项（用 required=False 手动校验：--list-datasets 不需要 kb ----）
    parser.add_argument("--kb-id", default=None, help="要评估的知识库 id")
    parser.add_argument(
        "--dataset",
        default="sample_ml",
        help="数据集：内置名（sample_ml / sample_retrieval）、data/eval/datasets 下的文件名，或 JSON 路径",
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help=f"逗号分隔的检索模式，可选 {', '.join(DEFAULT_MODES)}（默认全部）",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["ollama", "openai"],
        help="覆盖生成通道；留空用设置里的当前通道",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑前 N 条样本（快速冒烟用）",
    )

    # ---- 开关 ----
    parser.add_argument("--list-datasets", action="store_true", help="列出可用数据集后退出")
    parser.add_argument("--llm-judge", action="store_true", help="额外调用 LLM 打分（更慢）")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线自检：用样本自带 context 当参考资料，不调检索与生成，仅验证指标与报告管线",
    )
    parser.add_argument("--quiet", action="store_true", help="只打印汇总表，不打印逐样本明细")

    # ---- 合成数据集 ----
    parser.add_argument(
        "--build-dataset",
        type=int,
        default=0,
        metavar="N",
        help="先用 LLM 从知识库合成 N 条评估集再跑（需要可用的生成通道）",
    )
    parser.add_argument(
        "--dataset-name",
        default="synthetic",
        help="合成数据集的名字（决定落盘文件名，默认 synthetic）",
    )
    parser.add_argument("--seed", type=int, default=42, help="合成时的随机种子，固定种子保证可复现")

    # ---- 输出 ----
    parser.add_argument(
        "--out",
        default=None,
        help="报告输出路径；留空则写到 data/eval/<数据集>_<时间戳>.json",
    )
    return parser


def parse_modes(raw: str) -> list[str]:
    """解析 `--modes`，过滤掉空项与非法值。

    非法值只警告不报错：评估脚本的价值在于能跑完，
    而拼错一个模式名（`dense` 而不是 `dense_only`）太常见了，
    直接崩掉会让人以为整个实验环境坏了。
    """
    modes: list[str] = []
    for piece in (raw or "").split(","):
        name = piece.strip()
        if not name:
            continue
        if name not in DEFAULT_MODES:
            print(f"  [警告] 忽略未知检索模式 `{name}`，可选：{', '.join(DEFAULT_MODES)}")
            continue
        if name not in modes:
            modes.append(name)
    return modes or list(DEFAULT_MODES)


async def _run(args: argparse.Namespace) -> int:
    from app.config import get_settings
    from app.db import init_db
    from app.db.base import session_scope

    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    # 预加载 jieba 词典。理由有两条：
    #   1. 它首次加载要 1 秒左右，而离线模式下第一个样本会把这段时间
    #      算进自己的耗时里，报告里就会出现一条 600ms 的假尖峰；
    #   2. jieba 初始化时会往 stderr 打「Building prefix dict...」，
    #      提前到进度输出之前打完，避免混进结果区域。
    if not args.list_datasets:
        from app.core.eval.metrics import _ensure_jieba

        _ensure_jieba()

    # ---------------- 只列数据集 ----------------
    if args.list_datasets:
        _rule()
        print("  可用评估数据集")
        _rule()
        for item in list_datasets(settings.data_dir):
            tag = "内置" if item["source"] == "builtin" else "文件"
            print(f"  [{tag}] {item['name']:<20} {item['sample_count']:>3} 条  {item['description']}")
            if item.get("path"):
                print(f"          {item['path']}")
        print()
        print("  内置数据集不依赖 LLM，可直接开始评估：")
        print("      python scripts/run_eval.py --kb-id <知识库id> --dataset sample_ml")
        return 0

    # ---------------- 载入 / 合成数据集 ----------------
    if not args.kb_id:
        print("  [错误] 缺少 --kb-id。用 --list-datasets 查看可用数据集。", file=sys.stderr)
        return 2

    modes = parse_modes(args.modes)

    dataset_name = args.dataset
    if args.build_dataset > 0:
        _rule()
        print(f"  ① 从知识库合成评估集（目标 {args.build_dataset} 条，seed={args.seed}）")
        _rule()
        with session_scope() as db:
            try:
                samples = await build_dataset_from_kb(
                    db,
                    args.kb_id,
                    args.build_dataset,
                    provider_name=args.provider,
                    seed=args.seed,
                )
            except Exception as exc:
                print(f"  [失败] 合成评估集失败：{type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
        target = save_dataset(
            settings.data_dir / "eval" / "datasets" / f"{args.dataset_name}.json",
            samples,
            name=args.dataset_name,
            description=f"由 LLM 从知识库 {args.kb_id} 合成（seed={args.seed}）",
            generator=f"llm:{args.provider or 'default'}",
        )
        print(f"  已生成 {len(samples)} 条样本 -> {target}")
        dataset_name = args.dataset_name
    else:
        try:
            samples, dataset_name = resolve_dataset(args.dataset, data_dir=settings.data_dir)
        except Exception as exc:
            print(f"  [失败] 无法载入数据集：{type(exc).__name__}: {exc}", file=sys.stderr)
            print("         用 --list-datasets 查看可用数据集。", file=sys.stderr)
            return 1

    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print("  [失败] 数据集里没有样本", file=sys.stderr)
        return 1

    # ---------------- 跑评估 ----------------
    _rule()
    print("  ② 开始评估")
    _rule()
    print(f"  知识库    {args.kb_id}")
    print(f"  数据集    {dataset_name}（{len(samples)} 条）")
    print(f"  检索模式  {', '.join(modes)}")
    print(f"  生成通道  {args.provider or '（按当前设置）'}")
    print(f"  离线模式  {'是（不调检索与生成）' if args.offline else '否'}")
    print(f"  LLM 裁判  {'开启' if args.llm_judge else '关闭'}")

    def on_progress(mode: str, done: int, total: int) -> None:
        # --quiet 时不打印进度：那种模式下面向的是"把输出重定向到文件"的用法，
        # 混进一堆 \r 会让日志没法看
        if args.quiet:
            return
        # 用 \r 原地刷新，避免几十条样本刷满终端
        print(f"\r  进行中：{mode:<12} {done}/{total}", end="", flush=True)

    def on_progress_done() -> None:
        if args.quiet:
            return
        print("\r" + " " * 40 + "\r", end="")

    with session_scope() as db:
        try:
            report = await run_evaluation(
                db,
                kb_id=args.kb_id,
                samples=samples,
                modes=modes,
                provider_name=args.provider,
                use_llm_judge=args.llm_judge,
                progress_cb=on_progress,
                offline_contexts=args.offline,
            )
        except Exception as exc:
            on_progress_done()
            print(f"  [失败] 评估执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        finally:
            on_progress_done()

    report.dataset_name = dataset_name
    _print_report(report, quiet=args.quiet)

    # ---------------- 落盘 ----------------
    out_path = (
        Path(args.out)
        if args.out
        else settings.data_dir / "eval" / f"{dataset_name}_{_stamp()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.to_json(indent=2) + "\n", encoding="utf-8")
    _rule()
    print(f"  报告已写入：{out_path}")
    print(f"  逐样本明细与指标定义见 JSON 里的 samples / notes 字段。")
    return 0


def _stamp() -> str:
    import time

    return time.strftime("%Y%m%d_%H%M%S")


def _print_report(report: EvalReport, *, quiet: bool) -> None:
    """把报告渲染到终端。

    汇总表是重点（论文表格直接照抄），明细只打印「问题 / 检索结果数 / 召回 /
    端到端耗时 / 是否出错」这几列 —— 终端放不下全部指标，全放反而看不清。
    """
    print()
    _rule()
    print("  ③ 汇总（各模式均值）")
    _rule()
    print(summarize(report))

    if report.notes:
        print()
        print("  可信度说明：")
        for note in report.notes:
            print(f"    - {note}")

    if quiet:
        return

    print()
    _rule()
    print("  ④ 逐样本明细（每个模式一条）")
    _rule()
    for mode in report.modes:
        rows = [r for r in report.samples if r.mode == mode]
        if not rows:
            continue
        print(f"\n  [{mode}]")
        for i, row in enumerate(rows, start=1):
            recall = row.metrics.get("recall")
            recall_text = "-" if recall is None else f"{recall:.2f}"
            flag = "!" if row.error else " "
            print(
                f"   {flag}{i:>3}. 召回 {recall_text}  检索 {len(row.retrieved_ids)} 条"
                f"  引用 {row.citation_count} 条  {row.elapsed_ms:>5} ms  gold={row.gold_mode}"
            )
            print(f"        Q: {row.question[:70]}")
            if row.error:
                print(f"        错误：{row.error[:150]}")
            elif row.answer:
                one_line = " ".join(row.answer.split())
                print(f"        A: {one_line[:100]}")

    failed = [r for r in report.samples if r.error]
    if failed:
        print()
        print(f"  [注意] 有 {len(failed)} 条样本执行失败，报告里已如实标注。")
        print(f"         首个错误：{failed[0].error}")


def main(argv: list[str] | None = None) -> int:
    """同步入口。`--help` 会由 argparse 处理后以 0 退出。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n  已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
