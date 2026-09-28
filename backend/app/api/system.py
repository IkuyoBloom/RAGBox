"""系统状态与模型列表。"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from sqlalchemy import select, text

from app.api.deps import DbSession, SettingsDep
from app.core.embedding import build_embedder
from app.core.llm import build_provider
from app.core.ingest import supported_extensions
from app.core.llm import PROVIDER_NAMES
from app.schemas import HealthOut, ProviderHealth

logger = logging.getLogger(__name__)
router = APIRouter(tags=["系统"])


@router.get("/health", response_model=HealthOut, summary="系统健康检查")
async def health(db: DbSession, settings: SettingsDep) -> HealthOut:
    cfg = settings.effective()

    # 数据库
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
        db.execute(select(1)).first()
    except Exception:
        logger.exception("数据库健康检查失败")
        db_ok = False

    # 向量库
    vector_ok = True
    vector_detail = ""
    try:
        from app.core.vectorstore import get_vector_store

        get_vector_store()
    except Exception as exc:
        vector_ok = False
        vector_detail = f"{type(exc).__name__}: {exc}"
        logger.exception("向量库健康检查失败")

    # 两条生成通道分别探测，界面上要能一眼看出哪条可用
    providers: list[ProviderHealth] = []
    for name in PROVIDER_NAMES:
        try:
            provider = build_provider(cfg, provider=name)
            info = await provider.health()
            providers.append(
                ProviderHealth(
                    name=name,
                    ok=bool(info.get("ok")),
                    detail=str(info.get("detail") or ""),
                    model=provider.model,
                    models=list(info.get("models") or [])[:200],
                    base_url=str(info.get("base_url") or ""),
                )
            )
        except Exception as exc:
            providers.append(
                ProviderHealth(
                    name=name,
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                    model="",
                )
            )

    active_provider = cfg.get("llm_provider") or "ollama"
    active = next((p for p in providers if p.name == active_provider), None)

    return HealthOut(
        status="ok" if db_ok and vector_ok else "degraded",
        version=str(cfg.get("app_version") or "0.1.0"),
        database=db_ok,
        vector_store=vector_ok,
        active_provider=active_provider,
        active_model=active.model if active else "",
        embed_model=str(cfg.get("ollama_embed_model") or ""),
        providers=providers,
    )


@router.get("/models", summary="列出可用的对话模型与向量模型")
async def list_models(settings: SettingsDep) -> dict:
    """把两条通道能用的模型都列出来，供设置页做下拉选择。"""
    cfg = settings.effective()
    out: dict[str, object] = {"providers": {}, "supported_file_types": supported_extensions()}

    for name in PROVIDER_NAMES:
        try:
            provider = build_provider(cfg, provider=name)
            info = await provider.health()
            out["providers"][name] = {  # type: ignore[index]
                "ok": bool(info.get("ok")),
                "detail": info.get("detail", ""),
                "models": list(info.get("models") or []),
                "configured_model": provider.model,
            }
        except Exception as exc:
            out["providers"][name] = {  # type: ignore[index]
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
                "models": [],
                "configured_model": "",
            }

    # 向量模型也探一下
    try:
        embedder = build_embedder(cfg)
        out["embedding"] = {
            "provider": embedder.name,
            "configured_model": embedder.model,
            "detail": (await embedder.health()).get("detail", ""),
        }
    except Exception as exc:
        out["embedding"] = {"error": f"{type(exc).__name__}: {exc}"}

    return out
