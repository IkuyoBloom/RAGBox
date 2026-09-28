"""验证真实 SSE 流式输出：token 是「边生成边推送」还是「攒完一次性返回」。

为什么必须打真服务器：FastAPI 的 TestClient 会把整个响应读完再返回，
流式与否在它眼里是一样的。只有走真实 HTTP 连接、边收边记时间戳，
才能证明首字节远早于响应结束。

用法（需后端与 Ollama 已启动）：

    python scripts/check_stream.py <知识库ID>
    python scripts/check_stream.py <知识库ID> --question "BM25 有什么缺点？" --mode hybrid
    python scripts/check_stream.py --list-kb          # 先列出可选知识库

退出码：0 表示确实是流式；1 表示疑似被缓冲或出错。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE = "http://127.0.0.1:8000"


def load_httpx():
    try:
        import httpx  # noqa: PLC0415

        return httpx
    except ImportError:
        print("[错误] 需要 httpx。请用后端虚拟环境运行本脚本：")
        print("       backend\\.venv\\Scripts\\python.exe scripts\\check_stream.py <kb_id>")
        raise SystemExit(1)


def list_kbs(httpx, base: str) -> int:
    try:
        data = httpx.get(f"{base}/api/kb", timeout=10).json()
    except Exception as exc:
        print(f"[错误] 读取知识库列表失败：{exc}")
        return 1
    if not data:
        print("  还没有知识库。先跑：python backend/scripts/prepare_demo_kb.py --reset")
        return 1
    print("  可用知识库：")
    for kb in data:
        print(f"    {kb['id']}  {kb['name']}  ({kb['chunk_count']} 切片)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 SSE 真流式输出")
    parser.add_argument("kb_id", nargs="?", help="知识库 id")
    parser.add_argument("--base", default=DEFAULT_BASE, help="后端地址")
    parser.add_argument("--question", default="BM25 检索有什么缺点？")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense_only", "sparse_only"])
    parser.add_argument("--list-kb", action="store_true", help="列出可选知识库后退出")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    httpx = load_httpx()

    if args.list_kb:
        return list_kbs(httpx, args.base)

    if not args.kb_id:
        print("[错误] 需要知识库 id。用 --list-kb 查看，或先跑 prepare_demo_kb.py")
        return 1

    print("=" * 66)
    print("  SSE 真流式验证")
    print("=" * 66)
    print(f"  知识库 : {args.kb_id}")
    print(f"  检索模式: {args.mode}")
    print(f"  提问   : {args.question}")
    print()

    payload = {"message": args.question, "kb_id": args.kb_id, "mode": args.mode}

    frames: list[dict] = []
    first_byte_at: float | None = None
    started = time.perf_counter()

    try:
        with httpx.Client(timeout=httpx.Timeout(args.timeout, connect=10.0)) as client:
            with client.stream("POST", f"{args.base}/api/chat/stream", json=payload) as resp:
                if resp.status_code >= 400:
                    body = resp.read().decode("utf-8", "replace")
                    print(f"[错误] HTTP {resp.status_code}：{body[:300]}")
                    return 1

                buffer = ""
                # iter_bytes 而非 iter_lines：要精确捕捉「第一个字节到达」的时刻
                for chunk in resp.iter_bytes():
                    if first_byte_at is None and chunk:
                        first_byte_at = time.perf_counter() - started
                    buffer += chunk.decode("utf-8", errors="replace")

                    while "\n\n" in buffer:
                        frame, _, buffer = buffer.partition("\n\n")
                        for line in frame.split("\n"):
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if not raw:
                                continue
                            try:
                                frames.append(json.loads(raw))
                            except json.JSONDecodeError:
                                pass
    except Exception as exc:
        print(f"[错误] 请求失败：{type(exc).__name__}: {exc}")
        return 1

    total = time.perf_counter() - started

    # ---------------------------------------------------------------- 统计
    kinds: dict[str, int] = {}
    for f in frames:
        kinds[f.get("type", "?")] = kinds.get(f.get("type", "?"), 0) + 1

    print("  [帧统计]")
    for name in ("start", "rewrite", "citations", "trace", "delta", "done", "error", "eos"):
        if kinds.get(name):
            print(f"    {name:<10} × {kinds[name]}")
    print(f"    {'合计':<10} × {len(frames)}")
    print()

    answer = "".join(f.get("text", "") for f in frames if f.get("type") == "delta")
    citations = next((f.get("citations", []) for f in frames if f.get("type") == "citations"), [])
    trace = next((f for f in frames if f.get("type") == "trace"), {})
    done = next((f for f in frames if f.get("type") == "done"), {})
    error = next((f for f in frames if f.get("type") == "error"), None)

    if trace:
        print(
            f"  [检索] 向量 {trace.get('dense_hits')} 条 / BM25 {trace.get('sparse_hits')} 条 / "
            f"融合 {trace.get('fused_hits')} 条，{trace.get('elapsed_ms')} ms"
        )
    if citations:
        print(f"  [引用] {len(citations)} 条")
        for c in citations[:4]:
            page = f" 第{c['page']}页" if c.get("page") else ""
            print(f"        [{c['index']}] {c['filename']}{page}  来源={c['source']}")
    if done:
        print(
            f"  [生成] {done.get('provider')}/{done.get('model')}  "
            f"{done.get('latency_ms')} ms  tokens {done.get('prompt_tokens')}→{done.get('completion_tokens')}"
        )
    if error:
        print(f"  [错误] {error.get('message')}")
    print()

    print("  [回答]")
    for line in answer.strip().splitlines() or ["（空）"]:
        print(f"    {line}")
    print()

    # ---------------------------------------------------------------- 判定
    print("=" * 66)
    ttfb = first_byte_at if first_byte_at is not None else -1.0
    delta_count = kinds.get("delta", 0)

    print(f"  首字节 {ttfb:.3f}s   总耗时 {total:.3f}s   delta 帧 {delta_count} 个")
    print()

    failed = False

    if ttfb > 0 and total > 0 and ttfb < total * 0.7:
        print(f"  [通过] 首字节远早于响应结束 → 确实在流式推送")
    else:
        print(f"  [失败] 首字节接近总耗时 → 疑似被缓冲")
        failed = True

    if delta_count > 5:
        print(f"  [通过] 回答被拆成 {delta_count} 个增量帧（而非一整块）")
    else:
        print(f"  [失败] delta 帧仅 {delta_count} 个，可能没有逐 token 推送")
        failed = True

    if answer.strip():
        print("  [通过] 增量帧能拼出完整回答")
    else:
        print("  [失败] 没有拼出任何回答")
        failed = True

    print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
