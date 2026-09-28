"""一键准备演示知识库：建库 + 上传样例文档 + 等待索引完成。

用途：
  * 答辩演示前快速铺好数据，不用手动点界面上传；
  * 给评估脚本提供 --kb-id（脚本会打印出来）。

用法（在 backend 目录下）：
    .venv\\Scripts\\python.exe scripts\\prepare_demo_kb.py
    .venv\\Scripts\\python.exe scripts\\prepare_demo_kb.py --name "我的资料库" --reset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

DEFAULT_DOC_DIR = Path(__file__).resolve().parents[2] / "scripts" / "sample_docs"
SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".txt", ".text", ".log"}


def main() -> int:
    parser = argparse.ArgumentParser(description="准备演示知识库")
    parser.add_argument("--name", default="演示知识库", help="知识库名称")
    parser.add_argument("--doc-dir", type=Path, default=DEFAULT_DOC_DIR, help="样例文档目录")
    parser.add_argument("--reset", action="store_true", help="若同名知识库已存在则先删除")
    parser.add_argument("--timeout", type=int, default=300, help="等待索引的超时秒数")
    args = parser.parse_args()

    if not args.doc_dir.exists():
        print(f"[错误] 样例文档目录不存在：{args.doc_dir}")
        return 1

    files = [p for p in sorted(args.doc_dir.iterdir()) if p.suffix.lower() in SUPPORTED]
    if not files:
        print(f"[错误] {args.doc_dir} 下没有受支持的文件")
        return 1

    client = TestClient(app)

    # ---- 已存在同名知识库？----
    existing = next((kb for kb in client.get("/api/kb").json() if kb["name"] == args.name), None)
    if existing:
        if not args.reset:
            print(f"[提示] 知识库「{args.name}」已存在（id={existing['id']}），直接复用。")
            print(f"       如需重建请加 --reset")
            print(f"\nKB_ID={existing['id']}")
            return 0
        print(f"[清理] 删除已有的「{args.name}」…")
        client.delete(f"/api/kb/{existing['id']}")

    # ---- 建库 ----
    resp = client.post("/api/kb", json={"name": args.name, "description": "由 prepare_demo_kb.py 自动创建"})
    if resp.status_code != 201:
        print(f"[错误] 建库失败 HTTP {resp.status_code}：{resp.text[:300]}")
        return 1
    kb = resp.json()
    kb_id = kb["id"]
    print(f"[建库] {kb['name']}  id={kb_id}")
    print(f"       向量模型={kb['embed_model']}  切片={kb['chunk_size']}  重叠={kb['chunk_overlap']}")

    # ---- 上传 ----
    print(f"\n[上传] 共 {len(files)} 个文件")
    handles = []
    try:
        for path in files:
            handles.append((path, path.open("rb")))
        payload = [
            ("files", (path.name, fh, "application/octet-stream")) for path, fh in handles
        ]
        resp = client.post(f"/api/kb/{kb_id}/documents", files=payload)
        if resp.status_code != 200:
            print(f"[错误] 上传失败 HTTP {resp.status_code}：{resp.text[:300]}")
            return 1
        results = resp.json()
    finally:
        for _path, fh in handles:
            fh.close()

    doc_ids: list[str] = []
    for r in results:
        if r["ok"]:
            tag = "（内容重复，复用已有）" if r["duplicated"] else ""
            print(f"       ✓ {r['filename']}{tag}")
            doc_ids.append(r["document"]["id"])
        else:
            print(f"       ✗ {r['filename']}  {r['error']}")

    if not doc_ids:
        print("[错误] 没有任何文件上传成功")
        return 1

    # ---- 等索引 ----
    print(f"\n[索引] 等待后台完成（解析 → 切分 → 向量化）…")
    deadline = time.time() + args.timeout
    pending = set(doc_ids)
    done_info: dict[str, dict] = {}
    while pending and time.time() < deadline:
        for doc_id in list(pending):
            doc = client.get(f"/api/documents/{doc_id}").json()
            if doc["status"] in ("ready", "failed"):
                pending.discard(doc_id)
                done_info[doc_id] = doc
        if pending:
            time.sleep(1.5)

    print()
    total_chunks = 0
    for doc_id in doc_ids:
        doc = done_info.get(doc_id) or client.get(f"/api/documents/{doc_id}").json()
        if doc["status"] == "ready":
            total_chunks += doc["chunk_count"]
            ingest = (doc.get("meta") or {}).get("ingest", {})
            print(
                f"       ✓ {doc['filename']}  {doc['chunk_count']} 片  "
                f"{doc['char_count']} 字  "
                f"（向量化 {ingest.get('embed_ms', '?')} ms / 总计 {ingest.get('total_ms', '?')} ms）"
            )
        else:
            print(f"       ✗ {doc['filename']}  状态={doc['status']}  {doc['error'][:120]}")

    if pending:
        print(f"\n[警告] 还有 {len(pending)} 个文档在 {args.timeout} 秒内没索引完")

    print("\n" + "=" * 66)
    print(f"  演示知识库已就绪：{kb['name']}")
    print(f"  文档 {len(doc_ids)} 份，切片 {total_chunks} 片")
    print("=" * 66)
    print(f"\n下一步可以直接跑评估：")
    print(f"    .venv\\Scripts\\python.exe scripts\\run_eval.py --kb-id {kb_id} --dataset sample_retrieval")
    print(f"\nKB_ID={kb_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
