"""LLM 通道抽象层。

为什么要抽象：本课题需要**同一条检索链路接不同生成后端**做对比实验
（本地 Ollama vs 云端 OpenAI 兼容接口）。所以：
  * 上层（RAG 引擎、API）只认 `LLMProvider` 接口；
  * 换后端只改配置，不改业务代码；
  * 每条通道的真实耗时与 token 用量都被记录，直接喂给评估模块。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


class LLMError(RuntimeError):
    """LLM 调用失败。

    message 会被直接展示到前端，所以必须是「人能看懂 + 给出下一步动作」的中文，
    不要把原始堆栈丢出去。
    """

    def __init__(self, message: str, *, provider: str = "", status: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class StreamChunk:
    """流式输出的一个分片。

    末尾那一片 `finished=True`，并携带用量；中间片只有 text。
    """

    text: str = ""
    finished: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def has_usage(self) -> bool:
        return self.prompt_tokens is not None or self.completion_tokens is not None


@dataclass(slots=True)
class ChatResult:
    content: str
    provider: str
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider(ABC):
    """生成后端统一接口。"""

    name: str = "base"

    def __init__(self, model: str, *, timeout: float = 300.0, **options: Any) -> None:
        self.model = model
        self.timeout = timeout
        self.options = options

    # ---- 核心能力 ----
    @abstractmethod
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """一次性拿到完整回答（用于查询改写、评估等不需要流式的场景）。"""

    @abstractmethod
    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """流式生成。实现为 async generator。"""

    # ---- 运维能力 ----
    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """探测通道可用性，返回 {ok: bool, detail: str, models: [...]}。"""

    async def list_models(self) -> list[str]:
        info = await self.health()
        return list(info.get("models") or [])

    # ---- 工具 ----
    @staticmethod
    def _friendly_connect_error(exc: Exception, provider: str, target: str) -> LLMError:
        return LLMError(
            f"无法连接 {provider} 服务（{target}）。"
            f"请确认服务已启动；若用本地 Ollama，可执行 `ollama serve` 或启动 Ollama 应用。"
            f"原始错误：{type(exc).__name__}: {exc}",
            provider=provider,
        )
