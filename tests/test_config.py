"""配置加载与校验测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_defaults_are_local_friendly() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_version == "0.1.0"
    assert settings.log_level == "INFO"
    assert settings.milvus_uri == "http://127.0.0.1:19530"
    assert settings.redis_url == "redis://127.0.0.1:6379/0"
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ollama_model == "qwen2.5:7b-instruct"
    assert settings.embedding_dim == 1024


def test_env_override_and_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("MILVUS_URI", "http://10.0.0.5:19530/")

    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"
    assert settings.milvus_uri == "http://10.0.0.5:19530"


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="verbose")


def test_non_positive_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, embedding_dim=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, request_timeout_seconds=0)


def test_empty_service_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, redis_url="   ")


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
