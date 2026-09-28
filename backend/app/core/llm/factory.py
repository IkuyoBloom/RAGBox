"""按有效配置构造 LLM 通道实例。"""

from __future__ import annotations

from typing import Any

from app.core.llm.base import LLMError, LLMProvider
from app.core.llm.ollama_provider import OllamaProvider
from app.core.llm.openai_provider import OpenAICompatProvider

PROVIDER_NAMES = ("ollama", "openai")


def build_provider(cfg: dict[str, Any], *, provider: str | None = None) -> LLMProvider:
    """cfg 是 SettingsService 的有效配置快照（dict）。

    provider 参数允许单次请求覆盖配置里的通道 —— 界面上「这条对话用云端跑」
    就是靠它，也是双通道对比实验的实现方式。
    """
    name = (provider or cfg.get("llm_provider") or "ollama").lower()
    timeout = float(cfg.get("llm_timeout") or 300.0)

    if name == "ollama":
        return OllamaProvider(
            model=cfg.get("ollama_chat_model") or "qwen2.5:7b-instruct",
            base_url=cfg.get("ollama_base_url") or "http://127.0.0.1:11434",
            timeout=timeout,
            num_ctx=int(cfg.get("num_ctx") or 8192),
        )

    if name == "openai":
        return OpenAICompatProvider(
            model=cfg.get("openai_model") or "deepseek-chat",
            base_url=cfg.get("openai_base_url") or "https://api.deepseek.com/v1",
            api_key=cfg.get("openai_api_key") or "",
            timeout=timeout,
        )

    raise LLMError(
        f"未知的 LLM 通道 `{name}`，可选：{'、'.join(PROVIDER_NAMES)}",
        provider=name,
    )


__all__ = ["PROVIDER_NAMES", "build_provider", "OllamaProvider", "OpenAICompatProvider"]
