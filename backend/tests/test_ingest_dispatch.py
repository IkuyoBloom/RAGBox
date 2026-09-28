"""入库调度方式的回归测试。

对应 2026-09-25 那次「上传的文档永远停在 pending」的事故。

事故经过：为了让上传请求不再占着数据库连接（BackgroundTasks 会让
`get_db` 的连接一直占到后台任务结束），把入库改成了独立的 asyncio 任务。
改动中把 `_run_ingest` 写成了 `async def`，但端点里仍然按同步方式调用：

    _run_ingest(doc_id, cfg)      # 协程被创建，从未 await -> 永远不执行

后果是**新上传的文档永远停在 pending**，界面上看起来只是在排队，
日志里只有一行容易被忽略的：

    RuntimeWarning: coroutine '_run_ingest' was never awaited

这组测试盯住的就是「调度方式」，因为它的失败模式是**静默的** ——
不报错、不崩溃，只是什么都不发生。
"""

from __future__ import annotations

import asyncio
import inspect

from app.api import documents
from app.core.ingest import pipeline


class TestDispatchIsNotACoroutine:
    def test_run_ingest_是同步函数(self):
        """回归：写成 async def 却不 await，协程永远不执行。"""
        assert not inspect.iscoroutinefunction(documents._run_ingest), (
            "_run_ingest 不该是协程函数 —— 端点里是按同步方式调用它的，"
            "写成 async def 会导致入库静默地永不执行"
        )

    def test_spawn_ingest_也是同步的(self):
        """它内部用 asyncio.create_task，本身不该是协程。"""
        assert not inspect.iscoroutinefunction(pipeline.spawn_ingest)

    def test_调用_run_ingest_会真的调度一次(self):
        """真调用一次，断言落到了 spawn 上。

        注意 `_spawn` **本来就应该**收到协程对象（它内部 create_task 需要它），
        所以这里不检查类型，只检查「确实调度了一次」。
        「写成 async def 却不 await」由 test_run_ingest_是同步函数 负责拦。
        """
        calls: list[str] = []
        original = pipeline._spawn

        def fake_spawn(coro):
            calls.append("spawn")
            coro.close()  # type: ignore[union-attr]

        pipeline._spawn = fake_spawn  # type: ignore[assignment]
        try:
            documents._run_ingest("doc_x", {})
        finally:
            pipeline._spawn = original  # type: ignore[assignment]

        assert calls == ["spawn"], "应当恰好调度了一次入库"


class TestUploadEndpointSchedulesIngestion:
    def test_上传端点不再依赖_BackgroundTasks(self):
        """回归：BackgroundTasks 会让请求占着数据库连接直到后台任务结束，
        批量导入时几十个请求叠加就会打满连接池。"""
        source = inspect.getsource(documents.upload_documents)
        assert "background" not in source, "上传端点不该再使用 BackgroundTasks"
        assert "_run_ingest" in source, "上传端点应当调度入库"

    def test_重建索引端点不再依赖_BackgroundTasks(self):
        source = inspect.getsource(documents.reindex)
        assert "background" not in source
        assert "_run_ingest" in source

    def test_重建索引端点是异步的(self):
        """同步端点跑在线程池里，没有运行中的事件循环，
        spawn_ingest 内的 asyncio.create_task 会直接抛 RuntimeError。"""
        assert inspect.iscoroutinefunction(documents.reindex)

    def test_上传端点是异步的(self):
        assert inspect.iscoroutinefunction(documents.upload_documents)


class TestSpawnActuallyRuns:
    def test_spawn_ingest_能真正跑起来(self):
        """端到端地验证调度链路：spawn 出去的任务确实被执行了。"""
        ran: list[str] = []

        async def fake_ingest(doc_id: str, cfg: dict):
            ran.append(doc_id)

        original = pipeline._safe_ingest
        pipeline._safe_ingest = fake_ingest  # type: ignore[assignment]

        async def main():
            pipeline.spawn_ingest("doc_a", {})
            pipeline.spawn_ingest("doc_b", {})
            # 让出控制权，等被调度的任务跑完
            await asyncio.sleep(0.05)

        try:
            asyncio.run(main())
        finally:
            pipeline._safe_ingest = original  # type: ignore[assignment]

        assert sorted(ran) == ["doc_a", "doc_b"], f"两个任务都该被执行，实际 {ran}"

    def test_被调度的任务会被持有强引用(self):
        """asyncio 只持弱引用；不留一份的话任务可能在跑完前被回收。"""

        async def main():
            before = len(pipeline._background_tasks)
            pipeline.spawn_ingest("doc_c", {})
            assert len(pipeline._background_tasks) > before, (
                "应当把任务放进 _background_tasks，否则可能被垃圾回收"
            )
            await asyncio.sleep(0.05)

        asyncio.run(main())
