"""评估管线测试：数据集、执行器、报告序列化、接口。

与 `test_eval_metrics.py` 的分工：
  * 那边测**公式**（纯函数，零依赖）；
  * 这边测**管线**（数据集读写、报告结构、降级路径、HTTP 契约）。

同样**不需要 Ollama / 网络 / 真知识库**：
  * 执行器用 `offline_contexts=True` 走不调检索与生成的分支；
  * 数据库用内存 SQLite（`StaticPool` 保证多连接共享同一份内存库），
    不碰项目真实的 `data/rag.db`。

这里最有价值的用例是 `test_offline_report_*` 系列：它们把「报告结构」
钉死，而报告正是论文里做表和画图的数据源 —— 结构一变形，
画图脚本就会静默出错（拿到空数组画出一张空图，比报错更难发现）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.eval.dataset import (
    EvalSample,
    builtin_datasets,
    extract_keywords,
    get_builtin_dataset,
    list_datasets,
    load_dataset,
    resolve_dataset,
    save_dataset,
)
from app.core.eval.runner import (
    ANSWER_OVERLAP_THRESHOLD,
    keyword_hits,
    resolve_gold,
    run_evaluation,
    summarize,
)
from app.db.base import Base
from app.db.models import KnowledgeBase, new_id


# ======================================================================
# 夹具
# ======================================================================
@pytest.fixture()
def db_session() -> Session:
    """内存 SQLite 会话。

    必须用 `StaticPool` + `check_same_thread=False`：
    SQLite 的 `:memory:` 是**每连接一个独立库**，默认连接池会为不同连接
    造出不同的空库，于是"建表成功但查不到数据"。StaticPool 强制复用同一个连接。
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def kb_id(db_session: Session) -> str:
    """建一个空知识库（离线模式不需要任何 chunk）。"""
    kb = KnowledgeBase(
        id=new_id(),
        name=f"测试库-{new_id()[:6]}",
        description="供评估管线测试使用",
        embed_model="bge-m3",
        collection_name=f"kb_{new_id()}",
    )
    db_session.add(kb)
    db_session.commit()
    return kb.id


# ======================================================================
# 数据集：内置集与克隆隔离
# ======================================================================
def test_内置数据集内容合理() -> None:
    datasets = builtin_datasets()
    assert set(datasets) == {"sample_ml", "sample_retrieval"}
    samples = datasets["sample_ml"]
    assert len(samples) >= 5
    for sample in samples:
        assert sample.question.strip()
        assert sample.gold_answer.strip()
        # 每条都必须有可用的标准答案，否则检索指标无定义
        assert sample.has_gold
        # 带 context 才能走离线自检
        assert sample.context.strip()


def test_内置数据集返回副本() -> None:
    """改一份不能污染另一份，否则第二次运行结果会变。"""
    first = get_builtin_dataset("sample_ml")
    first[0].gold_keywords.append("被污染的临时词")
    first[0].gold_chunk_ids.append("fake-id")
    second = get_builtin_dataset("sample_ml")
    assert "被污染的临时词" not in second[0].gold_keywords
    assert "fake-id" not in second[0].gold_chunk_ids


def test_内置数据集名不存在时报错() -> None:
    with pytest.raises(KeyError) as exc:
        get_builtin_dataset("不存在的数据集")
    assert "sample_ml" in str(exc.value)


# ======================================================================
# 数据集：读写
# ======================================================================
def test_保存与载入往返一致(tmp_path: Path) -> None:
    samples = [
        EvalSample(question="问题一？", gold_answer="答案一", gold_chunk_ids=["c1"]),
        EvalSample(question="问题二？", gold_keywords=["关键词"], notes="备注"),
    ]
    path = save_dataset(tmp_path / "datasets" / "mine.json", samples, description="手写集")
    assert path.is_file()

    loaded = load_dataset(path)
    assert [s.question for s in loaded] == ["问题一？", "问题二？"]
    assert loaded[0].gold_chunk_ids == ["c1"]
    assert loaded[1].gold_keywords == ["关键词"]
    assert loaded[1].notes == "备注"


def test_数据集文件元信息完整(tmp_path: Path) -> None:
    """元信息（版本 / 生成方式 / 时间）必须落盘，否则半年后无法复现。"""
    path = save_dataset(tmp_path / "m.json", [EvalSample(question="q")], generator="llm:ollama")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] >= 1
    assert payload["generator"] == "llm:ollama"
    assert payload["created_at"]
    assert payload["sample_count"] == 1


def test_载入时跳过脏样本(tmp_path: Path) -> None:
    """一条脏数据不该毁掉整份数据集。"""
    path = tmp_path / "dirty.json"
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {"question": "好样本？", "gold_answer": "a"},
                    {"gold_answer": "缺少 question"},
                    "整个是字符串",
                    {"question": "   "},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loaded = load_dataset(path)
    assert [s.question for s in loaded] == ["好样本？"]


def test_载入裸数组格式(tmp_path: Path) -> None:
    """手写数据集常直接写数组，这里容忍。"""
    path = tmp_path / "bare.json"
    path.write_text(json.dumps([{"question": "q1"}, {"question": "q2"}], ensure_ascii=False), encoding="utf-8")
    assert len(load_dataset(path)) == 2


def test_载入空数据集报错(tmp_path: Path) -> None:
    """空数据集要报错，不能返回空列表让调用方跑出一份全是 None 的报告。"""
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"samples": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_载入非法_json_报错(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{不是 json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_载入不存在的文件报错() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset("D:/绝对不存在的路径/nope.json")


def test_resolve_dataset_支持内置名与路径(tmp_path: Path) -> None:
    samples, name = resolve_dataset("sample_ml")
    assert name == "sample_ml"
    assert len(samples) > 0

    path = save_dataset(tmp_path / "custom.json", [EvalSample(question="q")])
    samples2, name2 = resolve_dataset(str(path))
    assert name2 == "custom"
    assert len(samples2) == 1


def test_resolve_dataset_优先数据目录下的文件(tmp_path: Path) -> None:
    target = tmp_path / "eval" / "datasets" / "stored.json"
    save_dataset(target, [EvalSample(question="来自数据目录")])
    samples, name = resolve_dataset("stored", data_dir=tmp_path)
    assert name == "stored"
    assert samples[0].question == "来自数据目录"


def test_resolve_dataset_找不到时报错并列出可选(tmp_path: Path) -> None:
    with pytest.raises(KeyError) as exc:
        resolve_dataset("没有这个集", data_dir=tmp_path)
    assert "sample_ml" in str(exc.value)


def test_list_datasets_包含内置与文件(tmp_path: Path) -> None:
    save_dataset(tmp_path / "eval" / "datasets" / "x.json", [EvalSample(question="q")])
    items = list_datasets(tmp_path)
    by_name = {item["name"]: item for item in items}
    assert by_name["sample_ml"]["source"] == "builtin"
    assert by_name["x"]["source"] == "file"
    assert by_name["x"]["sample_count"] == 1


def test_list_datasets_坏文件不导致接口崩(tmp_path: Path) -> None:
    bad = tmp_path / "eval" / "datasets" / "broken.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{坏的", encoding="utf-8")
    items = list_datasets(tmp_path)
    broken = next(i for i in items if i["name"] == "broken")
    assert broken["sample_count"] == 0
    assert "读取失败" in broken["description"]


def test_extract_keywords_都是能命中的子串() -> None:
    """抽出来的每个关键词都必须是原文的子串，否则回退判定会永远失效。"""
    text = "过拟合是指模型在训练数据上表现很好，可以通过正则化缓解。"
    keywords = extract_keywords(text)
    assert keywords
    assert all(k in text for k in keywords)
    # 只要能命中「拟合」，包含「过拟合」的片段就会被判为相关 ——
    # 关键词回退靠的就是子串包含，不追求抽出的词本身完整
    assert "拟合" in keywords
    assert all(len(k) >= 2 for k in keywords)
    assert "可以" not in keywords
    assert len(keywords) <= 8


def test_extract_keywords_英文术语() -> None:
    keywords = extract_keywords("BM25 is a sparse retriever; RAG uses vectors.")
    assert "BM25" in keywords or "bm25" in keywords.lower()
    assert all(k in "BM25 is a sparse retriever; RAG uses vectors." for k in keywords)


def test_extract_keywords_空输入() -> None:
    assert extract_keywords("") == []
    assert extract_keywords("   ") == []
    assert extract_keywords("。，、！") == []


# ======================================================================
# 标准答案降级判定
# ======================================================================
def test_keyword_hits_命中与未命中() -> None:
    assert keyword_hits("过拟合需要正则化缓解", ["正则化", "神经网络"]) == ["正则化"]
    assert keyword_hits("完全无关的内容", ["正则化"]) == []


def test_keyword_hits_大小写不敏感() -> None:
    assert keyword_hits("BM25 是稀疏检索", ["bm25"]) == ["bm25"]


def test_resolve_gold_优先使用_chunk_ids() -> None:
    sample = EvalSample(
        question="q",
        gold_answer="a",
        gold_chunk_ids=["c1"],
        gold_keywords=["过拟合"],
    )
    gold, mode = resolve_gold(sample, [{"chunk_id": "c9", "content": "过拟合"}])
    assert mode == "chunk_ids"
    assert gold == ["c1"]


def test_resolve_gold_无_chunk_id_时用答案重合() -> None:
    sample = EvalSample(
        question="q",
        gold_answer="过拟合是模型过度拟合训练数据，可以用正则化缓解",
        gold_keywords=["什么的"],
    )
    retrieved = [
        {"chunk_id": "c1", "content": "过拟合是模型过度拟合训练数据，可以用正则化缓解"},
        {"chunk_id": "c2", "content": "今天天气不错，适合出门散步散心"},
    ]
    gold, mode = resolve_gold(sample, retrieved)
    assert mode == "answer_overlap"
    assert gold == ["c1"]


def test_resolve_gold_答案与关键词都不命中时用关键词档() -> None:
    """关键词档即使一条都没命中也要如实标注 mode，不能悄悄降级成 none。"""
    sample = EvalSample(question="q", gold_answer="", gold_keywords=["过拟合"])
    retrieved = [{"chunk_id": "c1", "content": "过拟合"}, {"chunk_id": "c2", "content": "无关"}]
    gold, mode = resolve_gold(sample, retrieved)
    assert mode == "keywords"
    assert gold == ["c1"]


def test_resolve_gold_什么都没有返回_none() -> None:
    sample = EvalSample(question="q")
    gold, mode = resolve_gold(sample, [{"chunk_id": "c1", "content": "内容"}])
    assert mode == "none"
    assert gold == []


def test_答案重合阈值在合理区间() -> None:
    assert 0.0 < ANSWER_OVERLAP_THRESHOLD < 1.0


# ======================================================================
# 执行器：离线模式（不调检索与生成）
# ======================================================================
async def test_offline_report_结构与指标完整(db_session: Session, kb_id: str) -> None:
    samples = get_builtin_dataset("sample_ml")[:3]
    report = await run_evaluation(
        db_session,
        kb_id=kb_id,
        samples=samples,
        modes=["hybrid", "dense_only"],
        offline_contexts=True,
    )

    assert report.kb_id == kb_id
    assert report.kb_name
    assert report.modes == ["hybrid", "dense_only"]
    assert report.sample_count == 3
    assert report.offline_contexts is True
    # 3 样本 × 2 模式
    assert len(report.samples) == 6

    for mode in report.modes:
        agg = report.aggregate[mode]
        # 生成类指标三件套必须存在
        for key in ("citation_coverage", "citation_validity", "answer_overlap"):
            assert key in agg
        # 检索指标块必须存在（离线模式下 gold 来自内置关键词，可能为 None）
        for key in ("hit_rate", "recall", "precision", "mrr", "ndcg"):
            assert key in agg
            assert set(agg[key]) == {"mean", "std", "min", "max", "n"}
        assert report.timings[mode]["elapsed_ms"]["mean"] is not None


async def test_offline_report_指标值合理(db_session: Session, kb_id: str) -> None:
    """回答直接复用 gold_answer，且参考资料就是该答案的 context，重合度必须很高。"""
    samples = get_builtin_dataset("sample_ml")
    report = await run_evaluation(
        db_session, kb_id=kb_id, samples=samples, modes=["hybrid"], offline_contexts=True
    )
    agg = report.aggregate["hybrid"]
    overlap = agg["answer_overlap"]["mean"]
    assert overlap is not None
    assert overlap > 0.5, f"离线自检的答案重合度过低（{overlap}），说明指标或数据有问题"
    # 内置样本的 gold_answer 全部没有角标，所以覆盖率是**真实的 0.0**
    # 而不是"无定义"——有实义句子但全都没标来源，正是覆盖率要抓的毛病。
    assert agg["citation_coverage"]["n"] == len(samples)
    assert agg["citation_coverage"]["mean"] == 0.0
    # 没有角标 = 没有编造来源，引用有效率定义为 1.0
    assert agg["citation_validity"]["mean"] == 1.0


async def test_离线报告能算引用指标(db_session: Session, kb_id: str) -> None:
    """把 gold_answer 换成带角标的文本，覆盖覆盖率 / 有效率这两条路径。

    内置样本本身不含角标，若不单独构造，报告里这两个字段永远是 0，
    真出 bug 也发现不了。
    """
    sample = EvalSample(
        question="什么是过拟合？",
        gold_answer="过拟合是模型过度拟合了训练数据[1]。缓解手段包括正则化[7]。",
        gold_keywords=["过拟合"],
        context="过拟合是模型过度拟合训练数据。缓解手段包括正则化和 Dropout。",
    )
    report = await run_evaluation(
        db_session, kb_id=kb_id, samples=[sample], modes=["hybrid"], offline_contexts=True
    )
    agg = report.aggregate["hybrid"]
    assert agg["citation_coverage"]["mean"] == 1.0
    # [7] 越界：参考资料只有 1 条 -> 2 个角标里有 1 个非法
    assert agg["citation_validity"]["mean"] == pytest.approx(0.5)


async def test_offline_report_可序列化为_json(db_session: Session, kb_id: str) -> None:
    """报告必须能无损落盘 —— 论文做表画图全靠它。"""
    report = await run_evaluation(
        db_session,
        kb_id=kb_id,
        samples=get_builtin_dataset("sample_retrieval"),
        modes=["hybrid", "sparse_only"],
        offline_contexts=True,
    )
    report.dataset_name = "sample_retrieval"

    text = report.to_json()
    restored = json.loads(text)
    assert restored["dataset_name"] == "sample_retrieval"
    assert restored["modes"] == ["hybrid", "sparse_only"]
    # 2 个模式 × 数据集样本数
    assert len(restored["samples"]) == 2 * len(get_builtin_dataset("sample_retrieval"))
    assert isinstance(restored["notes"], list) and restored["notes"]
    # 逐样本明细必须带 gold_mode，否则读者无法判断指标可信度
    assert all(row["gold_mode"] in {"chunk_ids", "answer_overlap", "keywords", "none"} for row in restored["samples"])
    # to_dict 与 to_json 必须一致
    assert report.to_dict() == restored


async def test_offline_report_notes_声明降级策略(db_session: Session, kb_id: str) -> None:
    report = await run_evaluation(
        db_session,
        kb_id=kb_id,
        samples=get_builtin_dataset("sample_ml")[:1],
        modes=["hybrid"],
        offline_contexts=True,
    )
    joined = "\n".join(report.notes)
    assert "离线模式" in joined
    assert "裁判" in joined  # 未启用裁判时也要说明忠实度是代理指标


async def test_报告汇总表可渲染(db_session: Session, kb_id: str) -> None:
    report = await run_evaluation(
        db_session,
        kb_id=kb_id,
        samples=get_builtin_dataset("sample_ml")[:2],
        modes=["hybrid", "dense_only"],
        offline_contexts=True,
    )
    table = summarize(report)
    assert "hybrid" in table
    assert "dense_only" in table
    assert "命中率" in table
    assert "耗时" in table


async def test_进度回调异常不影响评估(db_session: Session, kb_id: str) -> None:
    """进度条坏了不该让实验失败。"""
    seen: list[tuple[str, int, int]] = []

    def bad_callback(mode: str, done: int, total: int) -> None:
        seen.append((mode, done, total))
        raise RuntimeError("进度条炸了")

    report = await run_evaluation(
        db_session,
        kb_id=kb_id,
        samples=get_builtin_dataset("sample_ml")[:2],
        modes=["hybrid"],
        offline_contexts=True,
        progress_cb=bad_callback,
    )
    assert seen == [("hybrid", 1, 2), ("hybrid", 2, 2)]
    assert len(report.samples) == 2


async def test_知识库不存在时报错(db_session: Session) -> None:
    with pytest.raises(ValueError) as exc:
        await run_evaluation(
            db_session,
            kb_id="不存在的知识库id",
            samples=get_builtin_dataset("sample_ml")[:1],
            offline_contexts=True,
        )
    assert "知识库不存在" in str(exc.value)


async def test_空样本列表报错(db_session: Session, kb_id: str) -> None:
    with pytest.raises(ValueError):
        await run_evaluation(db_session, kb_id=kb_id, samples=[], offline_contexts=True)


# ======================================================================
# HTTP 接口契约
# ======================================================================
def test_接口_列出数据集() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.get("/api/eval/datasets")
    assert resp.status_code == 200
    names = {item["name"] for item in resp.json()}
    assert "sample_ml" in names


def test_接口_内置示例数据集() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.get("/api/eval/sample-dataset", params={"name": "sample_ml"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sample_count"] >= 5
    first = body["samples"][0]
    assert first["question"]
    assert first["gold_answer"]
    assert first["gold_keywords"]


def test_接口_内置数据集名不存在返回_404() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.get("/api/eval/sample-dataset", params={"name": "不存在"})
    assert resp.status_code == 404


def test_接口_评估运行缺少知识库返回_404() -> None:
    """kb_id 不存在必须是 404，而不是 500 —— 前端要能区分"配置错了"和"服务坏了"。"""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/eval/run",
        json={"kb_id": "一定不存在的kb", "dataset": "sample_ml"},
    )
    assert resp.status_code == 404
