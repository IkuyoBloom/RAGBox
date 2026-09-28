"""FastAPI 应用入口。

启动：
    cd backend
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload --port 8000

接口文档：http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import chat, documents, eval as eval_api, kb, settings as settings_api, system
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    cfg.ensure_dirs()

    from app.db import init_db

    init_db()
    logger.info("数据库已就绪：%s", cfg.db_path)
    logger.info("数据目录：%s", cfg.data_dir)
    logger.info("默认生成通道：%s（%s）", cfg.llm_provider, cfg.ollama_chat_model)
    logger.info("向量模型：%s", cfg.ollama_embed_model)

    # 恢复上次没跑完的文档。
    #
    # 入库是后台任务，进程一退出它们就没了，但数据库里的状态会停在
    # PARSING / EMBEDDING —— 界面上表现为文档永远「向量化中」，
    # 用户不能重试也不知道为什么。启动时统一捞回来重新排队。
    try:
        from app.core.ingest.pipeline import requeue_unfinished

        recovered = await requeue_unfinished()
        if recovered:
            logger.info("已恢复 %d 份上次未完成的文档，正在后台重新入库", recovered)
    except Exception:
        logger.exception("恢复未完成文档失败（不影响启动）")

    # 启动时探一次 Ollama，把「模型没拉」这类问题提前暴露在日志里，
    # 而不是等用户提问时才报错。
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{cfg.ollama_base_url.rstrip('/')}/api/tags")
        if resp.status_code < 400:
            models = [m.get("name", "") for m in (resp.json().get("models") or [])]
            logger.info("Ollama 可用，已安装模型：%s", "、".join(models) or "（无）")
            for need in (cfg.ollama_chat_model, cfg.ollama_embed_model):
                if not any(m == need or m.split(":")[0] == need.split(":")[0] for m in models):
                    logger.warning("缺少模型 %s，请执行：ollama pull %s", need, need)
        else:
            logger.warning("Ollama 返回 HTTP %s", resp.status_code)
    except Exception as exc:
        logger.warning(
            "启动时无法连接 Ollama（%s）：%s。服务仍会启动，但本地通道不可用。",
            cfg.ollama_base_url,
            exc,
        )

    yield

    logger.info("服务关闭")


settings = get_settings()

app = FastAPI(
    title="本地 RAG 知识库问答系统",
    description=(
        "完全本地部署的检索增强生成系统。\n\n"
        "- **混合检索**：BM25 稀疏 + 向量稠密，RRF 融合\n"
        "- **引用溯源**：回答标注 [n]，可回溯到原文切片与页码\n"
        "- **双通道生成**：本地 Ollama / 任意 OpenAI 兼容接口，可切换对比\n"
        "- **多轮改写**：追问自动改写为独立检索式"
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 统一挂到 /api 下，前端只需要一个 baseURL
app.include_router(system.router, prefix="/api")
app.include_router(settings_api.router, prefix="/api")
app.include_router(kb.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(eval_api.router, prefix="/api")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底异常处理：给前端一句人话，同时把堆栈留在服务端日志里。"""
    logger.exception("未处理异常 %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务端内部错误：{type(exc).__name__}: {exc}"},
    )


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api": "/api",
    }
