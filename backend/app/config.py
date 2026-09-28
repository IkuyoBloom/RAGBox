"""应用配置。

分层原则：
  * 本模块只读「环境变量 / .env」，属于**出厂默认值**，进程启动后不变。
  * 用户在界面上改的设置写进数据库 `app_settings` 表，由
    `app.core.settings_service.SettingsService` 在请求时合并覆盖。
  * 业务代码一律通过 SettingsService 取「有效配置」，不要直接读本模块，
    否则界面上的开关会失效。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

LLMProviderName = Literal["ollama", "openai"]
EmbeddingProviderName = Literal["ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # `.env` 里写了 `KEY=` 这种空值时必须忽略，让代码默认值生效。
        # 否则 `EMBED_DIM=`（空串）会被当成 int 去解析，直接启动失败。
        env_ignore_empty=True,
    )

    # ---------------- 服务 ----------------
    app_name: str = "Local RAG"
    app_version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---------------- 路径 ----------------
    data_dir: Path = PROJECT_ROOT / "data"

    # ---------------- LLM ----------------
    llm_provider: LLMProviderName = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen2.5:7b-instruct"
    openai_base_url: str = "https://api.deepseek.com/v1"
    openai_api_key: str = ""
    openai_model: str = "deepseek-chat"

    # 供生成时用的采样参数
    temperature: float = 0.2
    max_tokens: int = 1024
    num_ctx: int = 8192
    # 单次请求最长等待秒数（本地 7B 模型首字可能较慢）
    llm_timeout: float = 300.0

    # ---------------- Embedding ----------------
    embedding_provider: EmbeddingProviderName = "ollama"
    ollama_embed_model: str = "bge-m3"
    embed_batch_size: int = 16
    # None = 首次调用时探测并缓存
    embed_dim: int | None = None

    # ---------------- 切分 ----------------
    chunk_size: int = 500
    chunk_overlap: int = 80

    # ---------------- 检索 ----------------
    enable_hybrid: bool = True
    dense_top_k: int = 20
    sparse_top_k: int = 20
    final_top_k: int = 6
    rrf_k: int = 60

    # ---------------- 生成阶段开关 ----------------
    enable_query_rewrite: bool = True
    # 允许回答「资料中没有」而不是硬编
    strict_grounding: bool = True

    # ---------------- 校验 ----------------
    @field_validator("embed_dim", mode="before")
    @classmethod
    def _blank_dim_to_none(cls, v: object) -> object:
        """空串 / "none" / "auto" 一律视为「自动探测」。"""
        if isinstance(v, str) and v.strip().lower() in ("", "none", "auto", "null"):
            return None
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError(f"chunk_overlap({v}) 必须小于 chunk_size({size})")
        return v

    # ---------------- 派生路径 ----------------
    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rag.db"

    @property
    def sqlalchemy_url(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.upload_dir, self.chroma_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程级单例。测试中可用 get_settings.cache_clear() 重置。"""
    return Settings()
