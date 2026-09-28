"""双通道验证：证明两条 LLM 通道都能真正跑通，而不是「写了但没用过」。

技巧：Ollama 自带 OpenAI 兼容端点 `/v1/chat/completions`，
所以可以把它当作一个「本地假云端」来验证 `OpenAICompatProvider` ——
不需要任何真实 API Key，也不需要联网。

顺带验证四类错误路径能否给出**人话**（这是前端体验的关键，
报错写成堆栈就等于没有报错）。

用法（在 backend 目录下，需 Ollama 已启动）：
    .venv\\Scripts\\python.exe scripts\\check_providers.py
    .venv\\Scripts\\python.exe scripts\\check_providers.py --base-url https://api.deepseek.com/v1 --api-key sk-xxx --model deepseek-chat
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.llm import ChatMessage, LLMError, OllamaProvider, OpenAICompatProvider  # noqa: E402

OK = "  [通过]  "
BAD = "  [失败]  "
INFO = "  [信息]  "

failures: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> bool:
    if cond:
        print(f"{OK}{label}")
    else:
        print(f"{BAD}{label}" + (f"\n           {detail}" if detail else ""))
        failures.append(label)
    return cond


async def check_provider(provider, label: str) -> None:
    print(f"\n{'=' * 68}")
    print(f"  {label}  (name={provider.name}, model={provider.model})")
    print("=" * 68)

    # ---- 1. 健康检查 ----
    info = await provider.health()
    print(f"{INFO}health: ok={info.get('ok')}  {info.get('detail', '')[:90]}")
    if not check(bool(info.get("ok")), "健康检查通过", str(info.get("detail"))):
        return
    models = info.get("models") or []
    print(f"{INFO}可用模型 {len(models)} 个：{'、'.join(models[:6])}")

    msgs = [ChatMessage(role="user", content="用一句话说明什么是 BM25。")]

    # ---- 2. 非流式 ----
    t0 = time.perf_counter()
    try:
        result = await provider.chat(msgs, temperature=0.1, max_tokens=120)
    except LLMError as exc:
        check(False, "非流式调用", str(exc))
        return
    ms = int((time.perf_counter() - t0) * 1000)
    print(f"{INFO}非流式回复（{ms}ms，token {result.prompt_tokens}->{result.completion_tokens}）：")
    print(f"           {result.content.strip()[:150]}")
    check(bool(result.content.strip()), "非流式返回了非空内容")
    check(result.provider == provider.name, f"provider 标记正确（{result.provider}）")
    check(result.latency_ms > 0, "记录了耗时")
    check(
        result.prompt_tokens is not None and result.completion_tokens is not None,
        "拿到了 token 用量（双通道对比实验依赖它）",
        f"prompt={result.prompt_tokens} completion={result.completion_tokens}",
    )

    # ---- 3. 流式 ----
    t0 = time.perf_counter()
    pieces: list[str] = []
    chunk_count = 0
    usage: tuple[int | None, int | None] = (None, None)
    try:
        async for chunk in provider.stream(msgs, temperature=0.1, max_tokens=120):
            if chunk.text:
                pieces.append(chunk.text)
                chunk_count += 1
            if chunk.finished:
                usage = (chunk.prompt_tokens, chunk.completion_tokens)
    except LLMError as exc:
        check(False, "流式调用", str(exc))
        return
    ms = int((time.perf_counter() - t0) * 1000)
    text = "".join(pieces)
    print(f"{INFO}流式回复（{ms}ms，{chunk_count} 个分片）：{text.strip()[:100]}")
    check(chunk_count > 1, f"确实被拆成多个分片（{chunk_count} 个），而非一次性返回")
    check(bool(text.strip()), "流式拼接结果非空")
    print(f"{INFO}流式 usage：prompt={usage[0]} completion={usage[1]}")
    check(usage[1] is not None, "流式也拿到了 completion token 数")

    # ---- 4. 错误路径：不存在的模型 ----
    try:
        bad = type(provider)(
            "definitely-not-a-real-model-xyz",
            **{k: v for k, v in provider.options.items()},
            base_url=getattr(provider, "base_url", ""),
            timeout=30.0,
            api_key=getattr(provider, "api_key", None),
            num_ctx=getattr(provider, "num_ctx", None),
        )
        await bad.chat([ChatMessage(role="user", content="hi")], max_tokens=8)
        check(False, "模型不存在时应报错", "居然没报错")
    except LLMError as exc:
        msg = str(exc)
        print(f"{INFO}模型不存在的报错：{msg[:130]}")
        check(
            "找不到" in msg or "不存在" in msg or "404" in msg,
            "错误信息说明是「模型不存在」",
        )
        check(len(msg) > 10 and "Traceback" not in msg, "错误信息是人话而不是堆栈")
    except Exception as exc:
        check(False, "模型不存在时应抛 LLMError", f"抛了 {type(exc).__name__}: {exc}")


async def check_unreachable() -> None:
    print(f"\n{'=' * 68}")
    print("  不可达地址的错误处理")
    print("=" * 68)
    provider = OllamaProvider("qwen2.5:7b-instruct", base_url="http://127.0.0.1:59999", timeout=8.0)
    try:
        await provider.chat([ChatMessage(role="user", content="hi")], max_tokens=8)
        check(False, "连接失败时应报错", "居然没报错")
    except LLMError as exc:
        msg = str(exc)
        print(f"{INFO}报错内容：{msg[:170]}")
        check("无法连接" in msg, "提示「无法连接」")
        check("ollama serve" in msg or "启动" in msg, "给出了可执行的下一步建议")
        check("Traceback" not in msg, "没有把堆栈丢给用户")


async def main() -> int:
    parser = argparse.ArgumentParser(description="双通道 LLM 验证")
    parser.add_argument("--ollama-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1",
                        help="OpenAI 兼容端点；默认指向 Ollama 自带的兼容接口")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--model", default=None, help="兼容端点的模型名，默认同 --ollama-model")
    parser.add_argument("--skip-openai", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    args = parser.parse_args()

    print("=" * 68)
    print("  双通道 LLM 验证")
    print("=" * 68)

    if not args.skip_ollama:
        await check_provider(
            OllamaProvider(args.ollama_model, base_url="http://127.0.0.1:11434", timeout=180.0, num_ctx=4096),
            "通道一：本地 Ollama（原生 /api/chat）",
        )

    if not args.skip_openai:
        await check_provider(
            OpenAICompatProvider(
                args.model or args.ollama_model,
                base_url=args.base_url,
                api_key=args.api_key,
                timeout=180.0,
            ),
            f"通道二：OpenAI 兼容（{args.base_url}）",
        )

    await check_unreachable()

    print("\n" + "=" * 68)
    if failures:
        print(f"  未通过 {len(failures)} 项：")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  双通道全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
