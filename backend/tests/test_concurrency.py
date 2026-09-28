"""并发与阻塞相关的回归测试。

这一组测试的每一条都对应 **2026-09-25 那次「上传过多卡死」事故**的一个成因。
那次的现象极具迷惑性：界面完全没反应、连 /api/health 都要 60 秒，
但 **GPU 利用率只有 6%** —— 说明瓶颈根本不在模型推理，而在下面这些地方：

1. `get_or_create_collection` 每次读写都调一遍，几十个入库任务反复打
   Chroma 内部的 SQLite 元数据表，把锁争用放大到整个服务无法响应。
2. PDF 解析、切分、Chroma 读写这些**同步**操作直接写在校程里，
   把事件循环占死 —— 异步 HTTP 请求全部排队。
3. 没有并发上限，111 个文档的后台任务同时开跑，一起抢 Ollama 和向量库。
4. 进程退出后，数据库里的状态永远停在 PARSING/EMBEDDING，没人接手。

这些性质用「跑一遍看看快不快」是测不稳的，必须断言具体行为。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from app.core.ingest import pipeline
from app.core.retrieval import hybrid
from app.core.vectorstore import ChromaVectorStore

# ======================================================================
# 1. Chroma collection 句柄必须缓存
# ======================================================================
class TestCollectionCaching:
    def test_重复取同一个_collection_只建一次(self, tmp_path: Path):
        """回归：原先每次 add/query/delete 都调 get_or_create_collection，
        批量入库时几十个任务反复打同一张元数据表。"""
        store = ChromaVectorStore(tmp_path / "chroma")

        calls = {"n": 0}
        real = store._client.get_or_create_collection

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        store._client.get_or_create_collection = counting  # type: ignore[method-assign]

        for _ in range(5):
            store._collection("kb_test")

        assert calls["n"] == 1, f"应当只创建一次，实际 {calls['n']} 次"

    def test_不同_collection_各自缓存(self, tmp_path: Path):
        store = ChromaVectorStore(tmp_path / "chroma")
        a = store._collection("kb_a")
        b = store._collection("kb_b")
        assert a is not b
        assert store._collection("kb_a") is a
        assert store._collection("kb_b") is b

    def test_drop_后缓存失效(self, tmp_path: Path):
        """drop 之后句柄失效，必须从缓存摘掉；
        否则重建同名 collection 时还在用那个悬空句柄，写入会静默丢失。"""
        store = ChromaVectorStore(tmp_path / "chroma")
        first = store._collection("kb_x")
        store.drop("kb_x")
        assert "kb_x" not in store._collections
        second = store._collection("kb_x")
        assert second is not first

    def test_并发取同一_collection_不重复创建(self, tmp_path: Path):
        """多线程同时取：双检锁要保证只建一个。"""
        store = ChromaVectorStore(tmp_path / "chroma")
        results: list[object] = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(store._collection("kb_race"))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len({id(r) for r in results}) == 1, "八个线程应当拿到同一个句柄"


# ======================================================================
# 2. 入库并发必须有闸门
# ======================================================================
class TestIngestGate:
    def test_并发上限为正且不大(self):
        """上限存在且刻意取小值：本机只有一个 Ollama，高并发只会互相排队。"""
        assert 1 <= pipeline.MAX_CONCURRENT_INGEST <= 4, pipeline.MAX_CONCURRENT_INGEST

    @pytest.mark.asyncio
    async def test_同时最多只有_上限_个入库在跑(self, monkeypatch):
        """回归：原先 111 个文档的后台任务会同时开跑。"""
        peak = {"value": 0}
        current = {"value": 0}
        lock = threading.Lock()

        async def fake_run(doc_id: str, cfg: dict):
            with lock:
                current["value"] += 1
                peak["value"] = max(peak["value"], current["value"])
            await asyncio.sleep(0.05)
            with lock:
                current["value"] -= 1
            return None  # type: ignore[return-value]

        monkeypatch.setattr(pipeline, "_run_ingest", fake_run)

        await asyncio.gather(
            *(pipeline.ingest_document(f"doc{i}", {}) for i in range(12))
        )

        assert peak["value"] <= pipeline.MAX_CONCURRENT_INGEST, (
            f"并发峰值 {peak['value']} 超过上限 {pipeline.MAX_CONCURRENT_INGEST}"
        )
        assert peak["value"] >= 2, "闸门不该把并发压到 1，否则白白浪费等待时间"

    @pytest.mark.asyncio
    async def test_闸门下的任务最终都会跑完(self, monkeypatch):
        """闸门只该限流，不该丢任务。"""
        done: list[str] = []

        async def fake_run(doc_id: str, cfg: dict):
            await asyncio.sleep(0.01)
            done.append(doc_id)
            return None  # type: ignore[return-value]

        monkeypatch.setattr(pipeline, "_run_ingest", fake_run)
        await asyncio.gather(
            *(pipeline.ingest_document(f"doc{i}", {}) for i in range(10))
        )
        assert sorted(done) == sorted(f"doc{i}" for i in range(10))


# ======================================================================
# 3. BM25 索引构建不能并发重复
# ======================================================================
class TestBm25BuildLock:
    def test_同一知识库的构建锁是同一把(self):
        assert hybrid._build_lock("kb1") is hybrid._build_lock("kb1")

    def test_不同知识库的构建锁相互独立(self):
        assert hybrid._build_lock("kb1") is not hybrid._build_lock("kb2")

    def test_并发建索引时只真正构建一次(self, monkeypatch):
        """回归：检索现在跑在线程池里，若多个请求同时发现缓存失效，
        就会各自为整库做一遍 jieba 分词 —— 1 万条切片时足以打满 CPU。"""
        builds = {"n": 0}
        real_build = hybrid.build_bm25_index

        def counting_build(db, kb_id):
            builds["n"] += 1
            return real_build(db, kb_id)

        monkeypatch.setattr(hybrid, "build_bm25_index", counting_build)
        hybrid.registry.clear()

        # 直接测 get_bm25_index 的加锁逻辑：需要真实 db，这里退而求其次，
        # 用桩替换 _kb_signature 与 build_bm25_index，聚焦并发控制本身。
        monkeypatch.setattr(hybrid, "_kb_signature", lambda db, kb_id: (3, 1))
        monkeypatch.setattr(
            hybrid, "build_bm25_index", lambda db, kb_id: hybrid.BM25Index(["a"], [["x"]])
        )

        results: list[object] = []
        barrier = threading.Barrier(6)

        def worker():
            barrier.wait()
            results.append(hybrid.get_bm25_index(None, "kb_race"))  # type: ignore[arg-type]

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 6
        assert len({id(r) for r in results}) == 1, "六个线程应当共享同一份索引"
