"""环境自检：确认依赖、路径、Ollama 连通性都正常。

用法（在 backend 目录下）：
    .venv\\Scripts\\python.exe scripts\\check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接以脚本方式运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK = "  [ok]  "
BAD = "  [!!]  "

REQUIRED = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic_settings",
    "sqlalchemy",
    "chromadb",
    "pypdf",
    "docx",
    "numpy",
    "jieba",
    "rank_bm25",
    "httpx",
]


def check_deps() -> bool:
    print("== 依赖 ==")
    all_ok = True
    for name in REQUIRED:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "?")
            print(f"{OK}{name:<20} {version}")
        except Exception as exc:
            all_ok = False
            print(f"{BAD}{name:<20} {type(exc).__name__}: {exc}")
    return all_ok


def check_paths() -> bool:
    print("\n== 路径与配置 ==")
    try:
        from app.config import get_settings

        settings = get_settings()
    except Exception as exc:
        print(f"{BAD}配置加载失败：{type(exc).__name__}: {exc}")
        return False

    print(f"{OK}data_dir    {settings.data_dir}")
    print(f"{OK}upload_dir  {settings.upload_dir}")
    print(f"{OK}chroma_dir  {settings.chroma_dir}")
    print(f"{OK}db_path     {settings.db_path}")
    print(f"{OK}LLM 通道    {settings.llm_provider}")
    print(f"{OK}对话模型    {settings.ollama_chat_model}")
    print(f"{OK}向量模型    {settings.ollama_embed_model}")

    settings.ensure_dirs()
    for p in (settings.data_dir, settings.upload_dir, settings.chroma_dir):
        if not p.exists():
            print(f"{BAD}目录创建失败：{p}")
            return False
    print(f"{OK}目录已就绪")
    return True


def check_db() -> bool:
    print("\n== 数据库 ==")
    try:
        from app.db import init_db, session_scope
        from app.db.models import KnowledgeBase
        from sqlalchemy import func, select

        init_db()
        with session_scope() as db:
            count = db.execute(select(func.count(KnowledgeBase.id))).scalar_one()
        print(f"{OK}建表成功，当前知识库数量：{count}")
        return True
    except Exception as exc:
        print(f"{BAD}数据库初始化失败：{type(exc).__name__}: {exc}")
        return False


def check_chroma() -> bool:
    print("\n== 向量库 ==")
    try:
        from app.config import get_settings
        from app.core.vectorstore import ChromaVectorStore

        store = ChromaVectorStore(get_settings().chroma_dir)
        print(f"{OK}ChromaDB 持久化客户端可用（{store.path}）")
        return True
    except Exception as exc:
        print(f"{BAD}ChromaDB 初始化失败：{type(exc).__name__}: {exc}")
        return False


def check_tokenizer() -> bool:
    print("\n== 中文分词与 BM25 ==")
    try:
        from app.core.retrieval.bm25 import BM25Index, tokenize
        from app.core.retrieval.hybrid import rrf_fuse

        tokens = tokenize("梯度下降法是训练神经网络最常用的优化算法之一。")
        print(f"{OK}jieba 分词：{'/'.join(tokens)}")

        # 关键校验：必须保留多字词。曾经因为正则只匹配单字，
        # 把「梯度下降法」拆成「梯/度/下/降/法」，分词等于白做。
        multi = [t for t in tokens if len(t) > 1 and not t.isascii()]
        print(f"{OK}其中多字词 {len(multi)} 个：{'/'.join(multi)}")
        if not multi:
            print(f"{BAD}分词结果全是单字，说明分词/过滤逻辑有问题")
            return False
        if "梯度下降" not in tokens and "梯度" not in tokens:
            print(f"{BAD}未能切出「梯度下降」等关键术语，分词质量不达标")
            return False

        index = BM25Index(
            ["a", "b", "c"],
            [
                tokenize("梯度下降法用于优化神经网络损失函数"),
                tokenize("今天天气不错适合出门散步"),
                tokenize("神经网络的反向传播算法推导"),
            ],
        )
        hits = index.search("神经网络优化", top_k=3)
        names = ", ".join(f"{h.chunk_id}({h.score:.2f})" for h in hits)
        print(f"{OK}BM25 检索命中：{names or '（无）'}")
        if not hits or hits[0].chunk_id == "b":
            print(f"{BAD}BM25 排序不符合预期，相关性最高的应是 a 或 c")
            return False

        dense = [("a", 0.9), ("b", 0.8)]
        sparse = [type("H", (), {"chunk_id": "c", "rank": 1, "score": 5.0})()]
        fused = rrf_fuse(dense, sparse, k=60)
        print(f"{OK}RRF 融合结果：{ {k: round(v, 5) for k, v in fused.items()} }")
        return True
    except Exception as exc:
        print(f"{BAD}分词/BM25 检查失败：{type(exc).__name__}: {exc}")
        return False


def check_splitter() -> bool:
    print("\n== 切分器 ==")
    try:
        from app.core.ingest.splitter import split_text

        sample = (
            "第一章 引言\n\n"
            "近年来，检索增强生成（RAG）技术受到广泛关注。它通过在生成前引入外部知识，"
            "有效缓解了大语言模型的幻觉问题。\n\n"
            "第二章 相关工作\n\n"
            "稀疏检索以 BM25 为代表，稠密检索以双塔向量模型为代表。两者各有优劣，"
            "因此出现了混合检索与融合排序的研究方向。"
        )
        chunks = split_text(sample, chunk_size=60, chunk_overlap=15)
        print(f"{OK}切分得到 {len(chunks)} 片：")
        for i, c in enumerate(chunks, 1):
            preview = c.replace("\n", "⏎")[:50]
            print(f"        [{i}] len={len(c):3d}  {preview}…")
        if not chunks:
            print(f"{BAD}切分结果为空")
            return False

        # 大量短段落必须被「打包」成接近 chunk_size 的片，
        # 而不是每个短段各占一片（那会让 Top-K 全被碎片占满）
        short_paras = "\n\n".join(
            f"第{i}条：这是一个用于测试的短段落，内容不长但也不至于只占几个字。"
            for i in range(1, 21)
        )
        packed = split_text(short_paras, chunk_size=200, chunk_overlap=30)
        sizes = [len(c) for c in packed]
        print(f"{OK}短段落打包：20 段 -> {len(packed)} 片，长度 {sizes}")
        avg = sum(sizes) / len(sizes) if sizes else 0
        print(f"{OK}平均片长 {avg:.0f} / 目标 200（{avg / 200:.0%}）")
        if avg < 120:
            print(f"{BAD}平均片长明显偏小，短段落没有被有效打包")
            return False
        return True
    except Exception as exc:
        print(f"{BAD}切分检查失败：{type(exc).__name__}: {exc}")
        return False


def check_ollama() -> bool:
    print("\n== Ollama 连通性 ==")
    try:
        import httpx

        from app.config import get_settings

        settings = get_settings()
        base = settings.ollama_base_url.rstrip("/")
        resp = httpx.get(f"{base}/api/tags", timeout=8.0)
        if resp.status_code >= 400:
            print(f"{BAD}Ollama 返回 HTTP {resp.status_code}")
            return False
        models = [m.get("name", "") for m in (resp.json().get("models") or [])]
        print(f"{OK}Ollama 可用（{base}），已安装 {len(models)} 个模型：")
        for m in models:
            print(f"        - {m}")

        def has(name: str) -> bool:
            return any(m == name or m.split(":")[0] == name.split(":")[0] for m in models)

        ok = True
        if not has(settings.ollama_chat_model):
            print(f"{BAD}缺少对话模型 {settings.ollama_chat_model} -> ollama pull {settings.ollama_chat_model}")
            ok = False
        if not has(settings.ollama_embed_model):
            print(f"{BAD}缺少向量模型 {settings.ollama_embed_model} -> ollama pull {settings.ollama_embed_model}")
            ok = False
        return ok
    except Exception as exc:
        print(f"{BAD}无法连接 Ollama：{type(exc).__name__}: {exc}")
        print("        请先安装并启动 Ollama，或检查 .env 里的 OLLAMA_BASE_URL")
        return False


def main() -> int:
    print("=" * 62)
    print("  本地 RAG 系统 - 环境自检")
    print("=" * 62)

    results = {
        "依赖": check_deps(),
        "路径配置": check_paths(),
        "数据库": check_db(),
        "向量库": check_chroma(),
        "分词/BM25": check_tokenizer(),
        "切分器": check_splitter(),
        "Ollama": check_ollama(),
    }

    print("\n" + "=" * 62)
    print("  汇总")
    print("=" * 62)
    for name, ok in results.items():
        print(f"{'  [通过]  ' if ok else '  [失败]  '}{name}")

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n未通过：{'、'.join(failed)}")
        return 1
    print("\n全部通过，可以启动服务。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
