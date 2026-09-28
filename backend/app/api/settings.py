"""运行时设置接口。"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter

from app.api.deps import SettingsDep, problem
from app.core.settings_service import (
    SOURCE_OLLAMA_CHAT_MODELS,
    SOURCE_OLLAMA_EMBED_MODELS,
    SettingsValidationError,
)
from app.schemas import (
    SettingGroup,
    SettingItem,
    SettingsOut,
    SettingsReset,
    SettingsUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/settings", tags=["设置"])


# ----------------------------------------------------------------------
async def _dynamic_choices(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """向 Ollama 查已装模型，按 capabilities 分成「对话模型」与「向量模型」。

    为什么要分类而不是把全部模型都列出来：让用户在一个只含 7B 对话模型的
    下拉里选「向量模型」，或者反过来，都会导致配置了一个根本跑不通的组合。
    Ollama 的 /api/tags 带 capabilities 字段，可以精确判断，不用靠名字猜。

    任何失败（Ollama 没启动、超时、返回格式变了）都返回空字典 ——
    界面会退化成「纯自由输入」。**连不上 Ollama 时绝不能让设置页报错**，
    因为「改模型名」正是用户从这个状态里自救的主要手段。
    """
    base = str(cfg.get("ollama_base_url") or "").rstrip("/")
    if not base:
        return {}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0)) as client:
            resp = await client.get(f"{base}/api/tags")
        if resp.status_code >= 400:
            logger.debug("查询 Ollama 模型列表返回 HTTP %s", resp.status_code)
            return {}
        models = resp.json().get("models") or []
    except Exception as exc:
        logger.debug("查询 Ollama 模型列表失败（不影响设置读取）：%s", exc)
        return {}

    chat: list[str] = []
    embed: list[str] = []
    for model in models:
        name = str(model.get("name") or "").strip()
        if not name:
            continue
        caps = set(model.get("capabilities") or [])
        if "embedding" in caps:
            embed.append(name)
        # 老版本 Ollama 不返回 capabilities，此时不做分类，一律当作可用
        if "completion" in caps or not caps:
            chat.append(name)

    return {
        SOURCE_OLLAMA_CHAT_MODELS: sorted(set(chat)),
        SOURCE_OLLAMA_EMBED_MODELS: sorted(set(embed)),
    }


async def _snapshot(settings: SettingsDep) -> dict[str, Any]:
    return settings.public_snapshot(await _dynamic_choices(settings.effective()))


def _to_out(snapshot: dict[str, Any]) -> SettingsOut:
    """快照 -> 响应模型。

    三个接口共用一份转换逻辑。之前是三处复制粘贴，
    加字段时漏改一处的风险很高 —— `groups` 就差点只在 GET 里出现，
    导致「保存后层级消失」这种诡异表现。
    """
    return SettingsOut(
        items=[SettingItem(**item) for item in snapshot["items"]],
        groups=[SettingGroup(**g) for g in snapshot.get("groups", [])],
        readonly=snapshot["readonly"],
        overridden_keys=snapshot["overridden_keys"],
    )


# ----------------------------------------------------------------------
@router.get("", response_model=SettingsOut, summary="读取当前有效配置")
async def read_settings(settings: SettingsDep) -> SettingsOut:
    return _to_out(await _snapshot(settings))


@router.put("", response_model=SettingsOut, summary="更新配置（写入数据库覆盖层）")
async def update_settings(payload: SettingsUpdate, settings: SettingsDep) -> SettingsOut:
    try:
        settings.set_many(payload.values)
    except SettingsValidationError as exc:
        raise problem(422, str(exc)) from exc
    return _to_out(await _snapshot(settings))


@router.post("/reset", response_model=SettingsOut, summary="还原为 .env 默认值")
async def reset_settings(
    payload: SettingsReset | None, settings: SettingsDep
) -> SettingsOut:
    settings.reset(payload.keys if payload else None)
    return _to_out(await _snapshot(settings))
