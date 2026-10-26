"""应用配置：统一从环境变量与 ``.env`` 文件读取。"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class Settings(BaseSettings):
    """运行时配置。

    字段名即环境变量名（大小写不敏感），例如 ``MILVUS_URI`` 映射到 ``milvus_uri``。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用
    app_name: str = "企业级智能文档检索与问答平台"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    request_timeout_seconds: float = Field(default=3.0, gt=0)

    # 依赖服务
    milvus_uri: str = "http://127.0.0.1:19530"
    redis_url: str = "redis://127.0.0.1:6379/0"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct"

    # 模型与存储
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = Field(default=1024, gt=0)
    sqlite_path: str = "data/app.sqlite3"

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in _LOG_LEVELS:
            raise ValueError(f"log_level 必须是 {sorted(_LOG_LEVELS)} 之一，实际为 {value!r}")
        return level

    @field_validator("milvus_uri", "redis_url", "ollama_base_url")
    @classmethod
    def _normalize_service_url(cls, value: str) -> str:
        url = value.strip().rstrip("/")
        if not url:
            raise ValueError("服务地址不能为空")
        return url


@lru_cache
def get_settings() -> Settings:
    """返回进程内单例配置，便于 FastAPI 依赖注入与测试覆盖。"""

    return Settings()
