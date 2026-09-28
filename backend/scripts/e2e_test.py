"""端到端验收：走真实 HTTP 接口跑通整条链路。

覆盖：
    建知识库 -> 上传文档 -> 后台建索引 -> 三种检索模式对比
    -> 流式问答（SSE） -> 引用溯源 -> 多轮追问改写 -> 引用正文可回查

用法（在 backend 目录下，需要 Ollama 已启动且模型已拉取）：
    .venv\\Scripts\\python.exe scripts\\e2e_test.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[2] / "scripts" / "sample_docs" / "检索增强生成技术综述.md"

PASS = "  [通过]  "
FAIL = "  [失败]  "
INFO = "  [信息]  "

failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    if condition:
        print(f"{PASS}{label}")
    else:
        print(f"{FAIL}{label}" + (f"  <- {detail}" if detail else ""))
        failures.append(label)
    return condition


def parse_sse(raw: str) -> list[dict]:
    events: list[dict] = []
    for frame in raw.split("\n\n"):
        for line in frame.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        pass
    return events


def main() -> int:
    print("=" * 70)
    print("  本地 RAG 系统 - 端到端验收")
    print("=" * 70)

    if not SAMPLE.exists():
        print(f"{FAIL}样例文档不存在：{SAMPLE}")
        return 1

    client = TestClient(app)
    t_start = time.perf_counter()

    # ---------------- 0. 健康检查 ----------------
    print("\n[0] 健康检查")
    health = client.get("/api/health").json()
    print(f"{INFO}status={health['status']} 通道={health['active_provider']} 模型={health['active_model']}")
    ollama = next((p for p in health["providers"] if p["name"] == "ollama"), None)
    if not check(bool(ollama and ollama["ok"]), "Ollama 通道可用", ollama["detail"] if ollama else ""):
        print("\nOllama 不可用，后续测试无法进行。请确认已启动 Ollama 并拉取模型。")
        return 1
    print(f"{INFO}已安装模型：{'、'.join(ollama['models'])}")

    # ---------------- 1. 建知识库 ----------------
    print("\n[1] 创建知识库")
    kb_name = f"E2E验收-{int(time.time())}"
    resp = client.post("/api/kb", json={"name": kb_name, "description": "端到端测试"})
    if not check(resp.status_code == 201, f"POST /api/kb -> {resp.status_code}", resp.text[:300]):
        return 1
    kb = resp.json()
    kb_id = kb["id"]
    print(f"{INFO}kb_id={kb_id}  embed_model={kb['embed_model']}  "
          f"chunk_size={kb['chunk_size']} overlap={kb['chunk_overlap']}")

    try:
        # ---------------- 2. 上传文档 ----------------
        print("\n[2] 上传样例文档")
        with SAMPLE.open("rb") as fh:
            resp = client.post(
                f"/api/kb/{kb_id}/documents",
                files=[("files", (SAMPLE.name, fh, "text/markdown"))],
            )
        if not check(resp.status_code == 200, f"上传 -> {resp.status_code}", resp.text[:300]):
            return 1
        results = resp.json()
        check(results[0]["ok"], "文件接收成功", str(results[0].get("error")))
        doc_id = results[0]["document"]["id"]
        print(f"{INFO}doc_id={doc_id}  size={results[0]['document']['size_bytes']} 字节")

        # ---------------- 3. 等待索引 ----------------
        print("\n[3] 等待后台建立索引（解析 -> 切分 -> 向量化）")
        deadline = time.time() + 240
        doc = None
        last_status = None
        while time.time() < deadline:
            doc = client.get(f"/api/documents/{doc_id}").json()
            if doc["status"] != last_status:
                print(f"{INFO}状态：{doc['status']}")
                last_status = doc["status"]
            if doc["status"] in ("ready", "failed"):
                break
            time.sleep(1.5)

        if not check(doc is not None and doc["status"] == "ready",
                     "文档索引完成", (doc or {}).get("error", "超时")):
            return 1

        print(f"{INFO}切片 {doc['chunk_count']} 片，共 {doc['char_count']} 字")
        ingest = (doc.get("meta") or {}).get("ingest", {})
        if ingest:
            print(f"{INFO}耗时：解析 {ingest.get('parse_ms')}ms  切分 {ingest.get('split_ms')}ms  "
                  f"向量化 {ingest.get('embed_ms')}ms  写向量库 {ingest.get('vector_ms')}ms  "
                  f"总计 {ingest.get('total_ms')}ms")
        check(doc["chunk_count"] > 0, "切分产出非空")
        check(ingest.get("embed_ms", 0) > 0, "向量化确实执行过")

        # ---------------- 4. 切片可回查 ----------------
        print("\n[4] 切片回查")
        chunks = client.get(f"/api/documents/{doc_id}/chunks?limit=3").json()
        check(len(chunks) > 0, f"能取到切片（{len(chunks)} 条）")
        if chunks:
            one = client.get(f"/api/chunks/{chunks[0]['id']}").json()
            check(bool(one.get("content")), "单个切片正文可读取")
            check(one.get("filename") == SAMPLE.name, "切片带了来源文件名")

        # ---------------- 5. 流式问答 ----------------
        print("\n[5] 流式问答（SSE）")
        question = "BM25 检索有什么优点和缺点？"
        print(f"{INFO}提问：{question}")
        resp = client.post(
            "/api/chat/stream",
            json={"message": question, "kb_id": kb_id, "provider": "ollama", "mode": "hybrid"},
        )
        check(resp.status_code == 200, f"POST /api/chat/stream -> {resp.status_code}", resp.text[:300])
        events = parse_sse(resp.text)
        kinds = [e.get("type") for e in events]
        print(f"{INFO}收到事件：{' -> '.join(kinds)}")

        check("citations" in kinds, "推送了引用列表")
        check("delta" in kinds, "推送了流式增量")
        check("done" in kinds, "推送了结束事件")

        answer = "".join(e.get("text", "") for e in events if e.get("type") == "delta")
        citations = next((e["citations"] for e in events if e.get("type") == "citations"), [])
        trace = next((e for e in events if e.get("type") == "trace"), {})
        done = next((e for e in events if e.get("type") == "done"), {})

        print(f"{INFO}检索：向量 {trace.get('dense_hits')} 条 / BM25 {trace.get('sparse_hits')} 条 "
              f"/ 融合 {trace.get('fused_hits')} 条，耗时 {trace.get('elapsed_ms')}ms")
        print(f"{INFO}引用 {len(citations)} 条：")
        for c in citations:
            page = f" 第{c['page']}页" if c.get("page") else ""
            print(f"        [{c['index']}] {c['filename']}{page}  "
                  f"来源={c['source']:<6s} 向量#{c['dense_rank']} BM25#{c['sparse_rank']}  "
                  f"分数={c['score']:.5f}")
        print(f"{INFO}生成：{done.get('provider')}/{done.get('model')} "
              f"耗时 {done.get('latency_ms')}ms  "
              f"tokens {done.get('prompt_tokens')}->{done.get('completion_tokens')}")
        print(f"\n{'-' * 70}\n{answer.strip()}\n{'-' * 70}\n")

        check(bool(answer.strip()), "生成了非空回答")
        check(len(citations) > 0, "检索到了引用片段")
        check("[" in answer, "回答中包含 [n] 引用标注")

        # 引用编号必须落在真实存在的片段范围内，越界说明模型在编造来源
        import re

        used = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
        valid = {c["index"] for c in citations}
        out_of_range = used - valid
        check(not out_of_range, "引用编号都在实际片段范围内",
              f"越界编号：{sorted(out_of_range)}（说明模型编造了来源）")
        print(f"{INFO}回答引用了编号：{sorted(used)}  可用编号：{sorted(valid)}")

        # ---------------- 6. 检索模式消融 ----------------
        print("\n[6] 检索模式对比（同一问题，三种模式）")
        mode_stats: dict[str, dict] = {}
        for mode in ("hybrid", "dense_only", "sparse_only"):
            r = client.post(
                "/api/chat/stream",
                json={"message": question, "kb_id": kb_id, "provider": "ollama", "mode": mode},
            )
            evs = parse_sse(r.text)
            tr = next((e for e in evs if e.get("type") == "trace"), {})
            mode_stats[mode] = tr
            print(f"{INFO}{mode:12s} 向量 {tr.get('dense_hits'):>3} 条  "
                  f"BM25 {tr.get('sparse_hits'):>3} 条  融合 {tr.get('fused_hits'):>3} 条  "
                  f"{tr.get('elapsed_ms'):>5} ms")

        check(mode_stats["dense_only"].get("sparse_hits", 0) == 0, "dense_only 模式不跑 BM25")
        check(mode_stats["sparse_only"].get("dense_hits", 0) == 0, "sparse_only 模式不跑向量检索")
        check(
            mode_stats["hybrid"].get("dense_hits", 0) > 0
            and mode_stats["hybrid"].get("sparse_hits", 0) > 0,
            "hybrid 模式两路都跑",
        )

        # ---------------- 7. 多轮追问与查询改写 ----------------
        print("\n[7] 多轮追问（考验查询改写）")
        conv_id = done.get("conversation_id") or None
        # 上一轮的 conversation_id 要从 start 事件取
        start_ev = next((e for e in events if e.get("type") == "start"), {})
        conv_id = start_ev.get("conversation_id")
        print(f"{INFO}会话 id = {conv_id}")

        follow_up = "那它的缺点呢？"
        print(f"{INFO}追问：{follow_up}（依赖上一轮的『BM25 检索』）")
        r = client.post(
            "/api/chat/stream",
            json={"message": follow_up, "conversation_id": conv_id, "mode": "hybrid"},
        )
        evs2 = parse_sse(r.text)
        rewrite_ev = next((e for e in evs2 if e.get("type") == "rewrite"), None)
        if rewrite_ev:
            print(f"{INFO}改写结果：{rewrite_ev['query']}")
            check(rewrite_ev["query"] != follow_up, "追问被改写成了独立检索式")
            check("BM25" in rewrite_ev["query"] or "bm25" in rewrite_ev["query"].lower(),
                  "改写结果包含被指代的对象（BM25）")
        else:
            print(f"{INFO}本轮未触发改写（可能判定为问题已足够独立），跳过改写断言")

        # ---------------- 8. 会话历史落库 ----------------
        print("\n[8] 会话历史")
        conv = client.get(f"/api/conversations/{conv_id}").json()
        roles = [m["role"] for m in conv["messages"]]
        print(f"{INFO}会话「{conv['title']}」共 {len(conv['messages'])} 条消息：{roles}")
        check(len(conv["messages"]) >= 4, "多轮消息都已落库")
        assistant_msgs = [m for m in conv["messages"] if m["role"] == "assistant"]
        check(any(m["citations"] for m in assistant_msgs), "助手消息带回了引用数据")
        check(any(m["rewritten_query"] for m in assistant_msgs), "追问的改写结果被持久化")

        # ---------------- 9. 设置接口 ----------------
        print("\n[9] 运行时设置")
        st = client.get("/api/settings").json()
        print(f"{INFO}可调项 {len(st['items'])} 个，已覆盖 {len(st['overridden_keys'])} 个")
        r = client.put("/api/settings", json={"values": {"final_top_k": 8, "rrf_k": 30}})
        check(r.status_code == 200, "PUT /api/settings -> 200", r.text[:200])
        st2 = r.json()
        got = {i["key"]: i["value"] for i in st2["items"]}
        check(got.get("final_top_k") == 8, "final_top_k 已更新为 8", str(got.get("final_top_k")))
        check(got.get("rrf_k") == 30, "rrf_k 已更新为 30", str(got.get("rrf_k")))
        r = client.put("/api/settings", json={"values": {"final_top_k": 1, "rrf_k": 999}})
        print(f"{INFO}非法值校验：HTTP {r.status_code}  {r.json().get('detail', '')[:80]}")
        check(r.status_code == 422, "非法配置被拒绝（422）")
        client.post("/api/settings/reset", json={"keys": ["final_top_k", "rrf_k"]})
        st3 = client.get("/api/settings").json()
        check("final_top_k" not in st3["overridden_keys"], "reset 能还原为默认值")

        # ---------------- 10. 空知识库兜底 ----------------
        print("\n[10] 空知识库兜底行为")
        empty_kb = client.post("/api/kb", json={"name": f"空库-{int(time.time())}"}).json()
        r = client.post("/api/chat/stream", json={"message": "你好", "kb_id": empty_kb["id"]})
        evs3 = parse_sse(r.text)
        txt = "".join(e.get("text", "") for e in evs3 if e.get("type") == "delta")
        check("没有" in txt or "上传" in txt, "空知识库返回了友好提示而不是报错", txt[:80])
        print(f"{INFO}提示文案：{txt.strip()[:60]}...")
        client.delete(f"/api/kb/{empty_kb['id']}")

        # ---------------- 11. 去重 ----------------
        print("\n[11] 重复上传去重（秒传）")
        with SAMPLE.open("rb") as fh:
            r = client.post(
                f"/api/kb/{kb_id}/documents",
                files=[("files", (SAMPLE.name, fh, "text/markdown"))],
            )
        dup = r.json()[0]
        check(dup.get("duplicated") is True, "同内容再次上传被识别为重复")
        print(f"{INFO}duplicated={dup.get('duplicated')} 复用已有文档 id={dup['document']['id']}")
        check(dup["document"]["id"] == doc_id, "复用的正是首次上传的那份文档")

    finally:
        # ---------------- 清理 ----------------
        print("\n[清理] 删除测试知识库")
        r = client.delete(f"/api/kb/{kb_id}")
        print(f"{INFO}DELETE /api/kb/{kb_id} -> HTTP {r.status_code}")
        check(r.status_code == 204, "测试知识库已删除")

    # ---------------- 汇总 ----------------
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"  验收结果（总耗时 {elapsed:.1f} 秒）")
    print("=" * 70)
    if failures:
        print(f"未通过 {len(failures)} 项：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("  全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
