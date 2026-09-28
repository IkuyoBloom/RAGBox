"""验证「向量模型可在设置里切换」的语义是否真的正确。

要证明三件事，缺一不可：
  1. 改了全局设置后，**新建**知识库确实用新模型；
  2. **已有**知识库的模型保持不变（建库时冻结，否则旧向量会与新查询不在同一语义空间）；
  3. 用新模型建库并真正索引一份文档，整条链路能跑通（而不只是把名字存进数据库）。

跑完会把测试知识库删掉，并把设置还原。

用法（需后端与 Ollama 已启动）：
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\check_embed_switch.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import httpx  # noqa: E402

BASE = "http://127.0.0.1:8000"
ALT_MODEL = "qwen3-embedding:4b"  # 与默认的 bge-m3 不同，且维度不同（2560 vs 1024）

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


def setting_value(client: httpx.Client, key: str):
    data = client.get(f"{BASE}/api/settings").json()
    return next(i for i in data["items"] if i["key"] == key)


def main() -> int:
    print("=" * 68)
    print("  向量模型切换语义验证")
    print("=" * 68)
    print()

    created: list[str] = []
    client = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))

    try:
        # ---------------- 前置：确认候选里有备选模型 ----------------
        embed_item = setting_value(client, "ollama_embed_model")
        current = embed_item["value"]
        options = embed_item["choices"] or []
        print(f"{INFO}当前向量模型 : {current}")
        print(f"{INFO}候选（已装） : {'、'.join(options)}")

        if not check(bool(options), "拿到了已装向量模型列表", str(options)):
            return 1
        if ALT_MODEL not in options:
            print(f"{INFO}候选里没有 {ALT_MODEL}，跳过后半段（先 ollama pull {ALT_MODEL}）")
            return 0
        if current == ALT_MODEL:
            print(f"{INFO}当前已是被测模型，先切回默认以形成对照")
            client.put(f"{BASE}/api/settings", json={"values": {"ollama_embed_model": "bge-m3"}})
            current = "bge-m3"

        # ---------------- 1. 用当前模型建库 ----------------
        print("\n[1] 用当前模型建一个知识库")
        kb_a = client.post(
            f"{BASE}/api/kb", json={"name": f"模型切换验证A-{int(time.time())}"}
        ).json()
        created.append(kb_a["id"])
        print(f"{INFO}A: {kb_a['name']}  载入库的模型 = {kb_a['embed_model']}")
        check(kb_a["embed_model"] == current, f"A 用了当前模型（{current}）", kb_a["embed_model"])

        # ---------------- 2. 切换全局设置 ----------------
        print(f"\n[2] 在设置里把向量模型切成 {ALT_MODEL}")
        resp = client.put(
            f"{BASE}/api/settings", json={"values": {"ollama_embed_model": ALT_MODEL}}
        )
        if not check(resp.status_code == 200, "PUT /api/settings -> 200", resp.text[:200]):
            return 1
        after = next(i for i in resp.json()["items"] if i["key"] == "ollama_embed_model")
        check(after["value"] == ALT_MODEL, f"设置值已变为 {ALT_MODEL}", str(after["value"]))
        check(after["overridden"] is True, "该项被标记为「已覆盖默认」")

        # ---------------- 3. 新建库用新模型 ----------------
        print("\n[3] 再建一个知识库，应该用新模型")
        kb_b = client.post(
            f"{BASE}/api/kb", json={"name": f"模型切换验证B-{int(time.time())}"}
        ).json()
        created.append(kb_b["id"])
        print(f"{INFO}B: {kb_b['name']}  载入库的模型 = {kb_b['embed_model']}")
        check(kb_b["embed_model"] == ALT_MODEL, f"B 用了新模型（{ALT_MODEL}）", kb_b["embed_model"])

        # ---------------- 4. 已有库不受影响（冻结语义）----------------
        print("\n[4] 已有的 A 库不应被改动")
        kb_a_now = client.get(f"{BASE}/api/kb/{kb_a['id']}").json()
        check(
            kb_a_now["embed_model"] == current,
            f"A 仍保持 {current}（向量模型建库时冻结）",
            kb_a_now["embed_model"],
        )

        # ---------------- 5. 真正索引进一份文档 ----------------
        print(f"\n[5] 往 B 库索引一份文档（验证新模型整条链路可用）")
        sample = next(
            (p for p in (BACKEND.parent / "scripts" / "sample_docs").glob("*.md")),
            None,
        )
        if sample is None:
            print(f"{INFO}找不到样例文档，跳过索引验证")
        else:
            with sample.open("rb") as fh:
                resp = client.post(
                    f"{BASE}/api/kb/{kb_b['id']}/documents",
                    files=[("files", (sample.name, fh, "text/markdown"))],
                )
            up = resp.json()[0]
            if check(up["ok"], "上传成功", str(up.get("error"))):
                doc_id = up["document"]["id"]
                deadline = time.time() + 300
                doc = None
                while time.time() < deadline:
                    doc = client.get(f"{BASE}/api/documents/{doc_id}").json()
                    if doc["status"] in ("ready", "failed"):
                        break
                    time.sleep(2)
                print(f"{INFO}状态={doc['status']}  切片={doc['chunk_count']}  错误={doc['error'][:120]}")
                if check(doc["status"] == "ready", "用新模型索引成功", doc.get("error", "")):
                    kb_b_now = client.get(f"{BASE}/api/kb/{kb_b['id']}").json()
                    print(f"{INFO}B 库记录的向量维度 = {kb_b_now['embed_dim']}")
                    check(
                        kb_b_now["embed_dim"] == 2560,
                        f"维度确为 {ALT_MODEL} 的 2560（bge-m3 是 1024，可据此区分）",
                        str(kb_b_now["embed_dim"]),
                    )

        # ---------------- 6. 还原设置 ----------------
        print("\n[6] 还原设置")
        resp = client.post(
            f"{BASE}/api/settings/reset", json={"keys": ["ollama_embed_model"]}
        )
        back = next(i for i in resp.json()["items"] if i["key"] == "ollama_embed_model")
        check(back["value"] == current, f"已还原为 {current}", str(back["value"]))
        check(back["overridden"] is False, "「已覆盖」标记已清除")

    finally:
        print("\n[清理] 删除验证用的知识库")
        for kb_id in created:
            r = client.delete(f"{BASE}/api/kb/{kb_id}")
            print(f"{INFO}DELETE /api/kb/{kb_id[:8]}… -> {r.status_code}")
        client.close()

    print("\n" + "=" * 68)
    if failures:
        print(f"未通过 {len(failures)} 项：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("  全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
