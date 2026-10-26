"""应用配置：统一从环境变量与 ``.env`` 文件读取。"""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_COLLECTION_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")


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
    embedding_model: str = "bge-m3"
    embedding_dim: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    embedding_timeout_seconds: float = Field(default=120.0, gt=0)
    sqlite_path: str = "data/app.sqlite3"
    milvus_collection: str = "document_chunks"

    # 文档入库
    upload_dir: str = "data/uploads"
    max_upload_mb: int = Field(default=20, ge=1, le=100)
    parent_chunk_chars: int = Field(default=1800, ge=200)
    child_chunk_chars: int = Field(default=400, ge=100)
    child_chunk_overlap: int = Field(default=80, ge=0)

    # 混合检索
    retrieval_top_k: int = Field(default=20, ge=1, le=100)
    retrieval_result_limit: int = Field(default=8, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=1000)

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

    @field_validator("upload_dir")
    @classmethod
    def _normalize_upload_dir(cls, value: str) -> str:
        directory = value.strip()
        if not directory:
            raise ValueError("upload_dir 不能为空")
        return directory

    @field_validator("embedding_model")
    @classmethod
    def _normalize_embedding_model(cls, value: str) -> str:
        model = value.strip()
        if not model:
            raise ValueError("embedding_model 不能为空")
        return model

    @field_validator("milvus_collection")
    @classmethod
    def _validate_collection_name(cls, value: str) -> str:
        name = value.strip()
        if not _COLLECTION_PATTERN.match(name):
            raise ValueError("milvus_collection 只能包含字母、数字、下划线，且以字母或下划线开头")
        return name

    @model_validator(mode="after")
    def _check_chunk_sizes(self) -> "Settings":
        if self.child_chunk_chars >= self.parent_chunk_chars:
            raise ValueError("child_chunk_chars 必须小于 parent_chunk_chars")
        if self.child_chunk_overlap >= self.child_chunk_chars:
            raise ValueError("child_chunk_overlap 必须小于 child_chunk_chars")
        return self


@lru_cache
def get_settings() -> Settings:
    """返回进程内单例配置，便于 FastAPI 依赖注入与测试覆盖。"""

    return Settings()
