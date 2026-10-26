"""依赖服务健康检查：Milvus / Redis / Ollama。

所有检查都自行捕获异常并返回结构化结果，避免单个依赖故障拖垮 ``/health``。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx
import redis.asyncio as aioredis
from pymilvus import MilvusClient

from app.core.config import Settings

DETAIL_MAX_LENGTH = 300


@dataclass(slots=True)
class DependencyStatus:
    """单个依赖的检查结果。"""

    name: str
    ok: bool
    latency_ms: float
    detail: str = ""


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _describe(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message[:DETAIL_MAX_LENGTH]


async def check_milvus(settings: Settings) -> DependencyStatus:
    started = time.perf_counter()
    try:
        client = MilvusClient(uri=settings.milvus_uri)
        try:
            collections = await asyncio.wait_for(
                asyncio.to_thread(client.list_collections),
                timeout=settings.request_timeout_seconds,
            )
        finally:
            await asyncio.to_thread(client.close)
        return DependencyStatus(
            name="milvus",
            ok=True,
            latency_ms=_elapsed_ms(started),
            detail=f"已连接，现有 collection 数量: {len(collections)}",
        )
    except Exception as exc:  # noqa: BLE001 - 健康检查需要吞掉所有异常
        return DependencyStatus(name="milvus", ok=False, latency_ms=_elapsed_ms(started), detail=_describe(exc))


async def check_redis(settings: Settings) -> DependencyStatus:
    started = time.perf_counter()
    client = aioredis.from_url(
        settings.redis_url,
        socket_connect_timeout=settings.request_timeout_seconds,
        socket_timeout=settings.request_timeout_seconds,
    )
    try:
        await asyncio.wait_for(client.ping(), timeout=settings.request_timeout_seconds)
        return DependencyStatus(name="redis", ok=True, latency_ms=_elapsed_ms(started), detail="PING 成功")
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(name="redis", ok=False, latency_ms=_elapsed_ms(started), detail=_describe(exc))
    finally:
        await client.aclose()


async def check_ollama(settings: Settings) -> DependencyStatus:
    started = time.perf_counter()
    url = f"{settings.ollama_base_url}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(name="ollama", ok=False, latency_ms=_elapsed_ms(started), detail=_describe(exc))

    models = {item.get("name", "") for item in payload.get("models", []) if isinstance(item, dict)}
    model_ready = any(name == settings.ollama_model or name.startswith(f"{settings.ollama_model}:") for name in models)
    detail = (
        f"已连接，模型 {settings.ollama_model} 就绪"
        if model_ready
        else f"已连接，但模型 {settings.ollama_model} 未下载（ollama pull {settings.ollama_model}）"
    )
    return DependencyStatus(name="ollama", ok=True, latency_ms=_elapsed_ms(started), detail=detail)


async def check_all(settings: Settings) -> list[DependencyStatus]:
    """并发检查全部依赖，顺序固定为 milvus / redis / ollama。"""

    return list(
        await asyncio.gather(
            check_milvus(settings),
            check_redis(settings),
            check_ollama(settings),
        )
    )
