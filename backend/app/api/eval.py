"""评估接口：跑离线实验、看可用数据集、取内置示例集。

## 为什么评估接口是同步返回，而不是做成后台任务

完整的后台任务需要任务表、状态轮询、取消、进程重启后的恢复 —— 这是一套
独立的基础设施。评估场景的使用者是**论文作者本人**，跑一次几十秒到几分钟，
同步等待（前端转圈）完全够用，也能立刻拿到报告。

如果将来样本涨到几百条，正确做法是把 `run_evaluation` 挪到后台线程 /
任务队列里，再提供一个 `GET /api/eval/report/{id}` —— 但那属于工程扩展，
当前规模下引入只会增加出错面。

## 接口设计上的一点坚持

`POST /eval/run` 的请求体里**没有** `temperature` / `top_k` 这类生成参数：
它们属于「系统配置」，评估必须跑在**与线上完全一致的配置**下才有意义。
想对比不同配置，就改设置页再跑一次，并把两次报告分别存档。
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import DbSession, get_kb_or_404, problem
from app.core.eval.dataset import (
    get_builtin_dataset,
    list_datasets,
    resolve_dataset,
)
from app.core.eval.runner import DEFAULT_MODES, run_evaluation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eval", tags=["评估"])

SearchModeName = Literal["hybrid", "dense_only", "sparse_only"]

# 单次请求的样本上限。没有上限的话，一个手滑的 dataset 加上三个模式
# 就能让本机跑上几小时，而 HTTP 连接早就超时了 —— 不如在入口就拦下。
MAX_SAMPLES_PER_RUN = 200


# ======================================================================
# 契约
# ======================================================================
class EvalDatasetInfo(BaseModel):
    """可用数据集的一条描述。"""

    name: str
    source: Literal["builtin", "file"]
    description: str = ""
    sample_count: int = 0
    path: str | None = None
    generator: str | None = None


class EvalRunRequest(BaseModel):
    """启动一次评估。

    dataset 可以是内置名（`sample_ml`）、`data/eval/datasets/` 下的文件名，
    或一个绝对/相对路径 —— 三种写法由 `dataset.resolve_dataset` 统一解析，
    前端不用关心区别。
    """

    kb_id: str = Field(min_length=1, description="要评估的知识库 id")
    dataset: str = Field(default="sample_ml", description="数据集名或 JSON 文件路径")
    modes: list[SearchModeName] = Field(
        default_factory=lambda: list(DEFAULT_MODES),
        description="要对比的检索模式，默认三种全跑（消融实验）",
    )
    provider: Literal["ollama", "openai"] | None = Field(
        default=None, description="覆盖生成通道；留空用设置里的当前通道"
    )
    use_llm_judge: bool = Field(
        default=False, description="是否额外调用 LLM 打分（更慢，但能拿到忠实度/相关性）"
    )
    limit: int | None = Field(
        default=None, ge=1, le=MAX_SAMPLES_PER_RUN, description="只跑前 N 条样本"
    )
    offline_contexts: bool = Field(
        default=False,
        description="离线自检：用样本自带 context 当参考资料，不调检索与生成",
    )


class EvalRunResponse(BaseModel):
    """评估报告 + 本次实际使用的数据集名。

    报告本体是自由结构（`dict`）而不是强类型模型，出于两个考虑：
      1. 指标的增删（比如以后加 BLEU）不该让接口契约跟着改；
      2. 论文里要拿这份 JSON 直接做表画图，保持与落盘文件**逐字节一致**
         比分成两套结构更可靠。
    `report` 的具体结构见 `core.eval.runner.EvalReport`。
    """

    dataset_name: str
    sample_count: int
    modes: list[str]
    report: dict[str, Any]


# ======================================================================
# 接口
# ======================================================================
@router.get(
    "/datasets",
    response_model=list[EvalDatasetInfo],
    summary="列出可用评估数据集",
)
def list_eval_datasets() -> list[EvalDatasetInfo]:
    """内置数据集 + `data/eval/datasets/*.json`。

    坏掉的 JSON 不会被隐藏：它以 `sample_count=0` 和一段说明出现在列表里，
    这样用户知道"文件在那里但读不了"，而不是以为文件丢了。
    """
    return [EvalDatasetInfo(**item) for item in list_datasets()]


@router.get(
    "/sample-dataset",
    summary="内置示例数据集（无需 LLM，无 Ollama 也能用）",
)
def sample_dataset(name: str = "sample_ml") -> dict[str, Any]:
    """返回内置示例数据集，可直接拿去 `POST /eval/run`。

    额外回传 `samples` 的完整字段（含 `context` 与 `gold_keywords`），
    既方便前端展示"评估集长什么样"，也方便用户存成文件后自己改。
    """
    try:
        samples = get_builtin_dataset(name)
    except KeyError as exc:
        raise problem(404, str(exc)) from exc
    return {
        "name": name,
        "description": "内置示例数据集：不依赖 LLM 与外部服务，用于打通评估流程。",
        "sample_count": len(samples),
        "samples": [s.to_dict() for s in samples],
    }


@router.post(
    "/run",
    response_model=EvalRunResponse,
    summary="运行评估（同步返回报告）",
)
async def run_eval(payload: EvalRunRequest, db: DbSession) -> EvalRunResponse:
    """在指定知识库上跑一轮检索消融评估。

    流程与 `scripts/run_eval.py` 完全一致（都调 `core.eval.runner.run_evaluation`），
    保证「页面上跑出来的数」和「命令行跑出来的数」一模一样。

    失败处理：
      * 知识库不存在 -> 404；
      * 数据集找不到 / 不可解析 -> 404；
      * 数据集为空、或全部模式都跑不通（通常是生成通道没起、知识库没建索引）
        -> 503，并在 detail 里给出**下一步该做什么**。
        这条很重要：一份"所有指标都是 0 或缺失"的报告很容易被误读成
        "混合检索没用"，而真实原因是模型没启动。
    """
    kb = get_kb_or_404(db, payload.kb_id)

    try:
        samples, dataset_name = resolve_dataset(payload.dataset)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise problem(404, str(exc)) from exc

    if payload.limit:
        samples = samples[: payload.limit]
    if not samples:
        raise problem(422, "数据集里没有样本")
    if len(samples) > MAX_SAMPLES_PER_RUN:
        raise problem(
            422,
            f"单次评估最多 {MAX_SAMPLES_PER_RUN} 条样本，当前 {len(samples)} 条。"
            f"请用 limit 参数分批跑，避免一次请求跑太久。",
        )

    modes = list(payload.modes) or list(DEFAULT_MODES)

    # 进度只写日志：接口是同步返回的，没有地方推进度条
    def on_progress(mode: str, done: int, total: int) -> None:
        if done == total or done % 5 == 0:
            logger.info("[eval] %s %d/%d", mode, done, total)

    try:
        report = await run_evaluation(
            db,
            kb_id=payload.kb_id,
            samples=samples,
            modes=modes,
            provider_name=payload.provider,
            use_llm_judge=payload.use_llm_judge,
            progress_cb=on_progress,
            offline_contexts=payload.offline_contexts,
        )
    except ValueError as exc:
        raise problem(422, str(exc)) from exc

    report.dataset_name = dataset_name

    # 全军覆没时给出可执行的建议，而不是让用户盯着一份空报告发呆
    if report.failed_count and report.failed_count == len(report.samples) and not payload.offline_contexts:
        first_error = next((r.error for r in report.samples if r.error), "")
        raise problem(
            503,
            f"所有样本都执行失败，评估无有效数据。首个错误：{first_error}\n"
            f"排查方向：① 设置页里的生成通道是否可用（Ollama 是否已启动、模型是否已拉取）；"
            f"② 知识库「{kb.name}」是否已上传文档并完成索引；"
            f"③ 若只想验证检索指标，可改用纯检索脚本或勾选离线自检模式。",
        )

    return EvalRunResponse(
        dataset_name=dataset_name,
        sample_count=len(samples),
        modes=report.modes,
        report=report.to_dict(),
    )


__all__ = [
    "EvalDatasetInfo",
    "EvalRunRequest",
    "EvalRunResponse",
    "router",
]
