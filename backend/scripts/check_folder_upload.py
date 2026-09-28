"""文件夹导入的端到端验证。

模拟浏览器「选择文件夹」时发出的多文件上传：每个文件都带相对路径作为文件名。
要证明三件事：

  1. 相对路径被**保留**为文档展示名 —— 否则 `论文/开题报告.md` 与
     `参考文献/开题报告.md` 会变成两个同名条目，界面上分不清；
  2. 路径穿越被挡住 —— `../../etc/evil.md` 不能变成可以写到上层目录的路径；
  3. 同名不同目录的文件确实成为两条独立记录，且各自能正常索引。

用法（需后端与 Ollama 已启动）：
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\check_folder_upload.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import httpx  # noqa: E402

BASE = "http://127.0.0.1:8000"

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


# 四段**各不相同**的内容。第一版让 `论文/开题报告.md` 与 `../../etc/evil.md`
# 共用同一段文字，结果触发了后端的内容去重（秒传），
# 把「同名不同目录要成为两条独立记录」这个要验证的点搅混了。
BODY_A = "量子计算利用叠加与纠缠特性处理信息。与经典比特不同，量子比特可以同时处于零和一的叠加态，这是量子并行性的来源。" * 2
BODY_B = "检索增强生成把外部知识引入生成过程。它先在知识库里检索相关片段，再让模型依据这些片段作答，因此答案可以溯源。" * 2
BODY_C = "向量数据库通过近似最近邻索引加速检索。常见的索引结构包括 HNSW、IVF 与乘积量化，三者在召回率与内存之间权衡。" * 2
BODY_D = "文本切分决定检索质量的上限。切得过粗会引入噪声，过细又会割裂语义，常见做法是保留十个百分点的重叠。" * 2

# (上报的文件名, 内容)
UPLOADS = [
    ("论文/开题报告.md", BODY_A),
    ("参考文献/开题报告.md", BODY_B),  # 与上面同名，不同目录
    ("论文/2024/综述.md", BODY_C),
    ("../../etc/evil.md", BODY_D),  # 路径穿越
]


def fetch_document(client: httpx.Client, doc_id: str) -> dict:
    """取文档状态，并在接口返回错误体时给出可读信息。

    直接 `.json()["status"]` 会在文档不存在时抛 KeyError，
    把「接口返回了错误」伪装成「脚本有 bug」，很难定位。
    """
    resp = client.get(f"{BASE}/api/documents/{doc_id}")
    data = resp.json()
    if not isinstance(data, dict) or "status" not in data:
        raise RuntimeError(f"GET /api/documents/{doc_id} -> HTTP {resp.status_code}：{data}")
    return data


def main() -> int:
    print("=" * 70)
    print("  文件夹导入端到端验证")
    print("=" * 70)
    print()

    client = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
    kb_id = ""

    try:
        kb = client.post(
            f"{BASE}/api/kb", json={"name": f"文件夹导入验证-{int(time.time())}"}
        ).json()
        kb_id = kb["id"]
        print(f"{INFO}测试知识库：{kb['name']}  向量模型={kb['embed_model']}")
        print()

        # ---------------- 上传 ----------------
        print("[1] 按浏览器的形式上传（文件名带相对路径）")
        files = [
            ("files", (name, body.encode("utf-8"), "text/markdown"))
            for name, body in UPLOADS
        ]
        resp = client.post(f"{BASE}/api/kb/{kb_id}/documents", files=files)
        if not check(resp.status_code == 200, f"上传 -> HTTP {resp.status_code}", resp.text[:300]):
            return 1
        results = resp.json()
        for r in results:
            flag = "✓" if r["ok"] else "✗"
            print(f"       {flag} 上报名={r['filename']!r}" + ("" if r["ok"] else f"  {r['error']}"))

        by_name = {r["filename"]: r for r in results}

        # ---------------- 路径保留 ----------------
        print("\n[2] 相对路径被保留为文档名")
        check("论文/开题报告.md" in by_name, "子目录路径保留（论文/开题报告.md）")
        check("参考文献/开题报告.md" in by_name, "另一子目录的路径保留（参考文献/开题报告.md）")
        check("论文/2024/综述.md" in by_name, "多层嵌套路径保留（论文/2024/综述.md）")
        check(
            len({r["filename"] for r in results}) == len(results),
            "四个上报名的文档名互不相同",
        )

        # ---------------- 路径穿越 ----------------
        print("\n[3] 路径穿越被挡住")
        traversal_names = [r["filename"] for r in results if "evil" in r["filename"]]
        check(bool(traversal_names), "穿越样本有返回结果")
        if traversal_names:
            name = traversal_names[0]
            print(f"{INFO}../../etc/evil.md 被规范化为 {name!r}")
            check(".." not in name, "结果里不含 `..`")
            check(not name.startswith("/"), "结果不是绝对路径")
            check(name.endswith("evil.md"), "文件名部分保留")
            check(
                len(name) < len("../../etc/evil.md") + 8,
                "路径被收敛而不是原样保留",
            )

        # ---------------- 落盘位置 ----------------
        print("\n[4] 落盘不受相对路径影响（不会真的建出嵌套目录）")
        upload_root = BACKEND.parent / "data" / "uploads" / kb_id
        if upload_root.exists():
            entries = list(upload_root.rglob("*"))
            nested = [p for p in entries if p.is_dir()]
            print(f"{INFO}上传目录内容：{[p.name for p in entries]}")
            check(
                not nested,
                "上传目录里没有子目录（落盘用 <doc_id><后缀>，与展示名无关）",
                str([p.name for p in nested]),
            )
        else:
            print(f"{INFO}未找到上传目录 {upload_root}，跳过")

        # ---------------- 索引与区分 ----------------
        print("\n[5] 两个同名文件成为独立记录，且都能索引")
        doc_ids = [r["document"]["id"] for r in results if r["ok"]]
        check(len(doc_ids) == len(set(doc_ids)), f"文档 id 互不相同（{len(doc_ids)} 条）")

        deadline = time.time() + 300
        pending = set(doc_ids)
        final: dict[str, dict] = {}
        while pending and time.time() < deadline:
            for doc_id in list(pending):
                doc = fetch_document(client, doc_id)
                if doc["status"] in ("ready", "failed"):
                    pending.discard(doc_id)
                    final[doc_id] = doc
            if pending:
                time.sleep(2)

        for r in results:
            if not r["ok"]:
                continue
            doc = final.get(r["document"]["id"]) or fetch_document(
                client, r["document"]["id"]
            )
            print(f"       {r['filename']!r:<34} 状态={doc['status']}  切片={doc['chunk_count']}")
            check(doc["status"] == "ready", f"{r['filename']} 索引完成", doc.get("error", "")[:120])

    finally:
        print("\n[清理] 删除测试知识库")
        if kb_id:
            r = client.delete(f"{BASE}/api/kb/{kb_id}")
            print(f"{INFO}DELETE /api/kb/{kb_id[:8]}… -> {r.status_code}")
        client.close()

    print("\n" + "=" * 70)
    if failures:
        print(f"未通过 {len(failures)} 项：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("  全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
