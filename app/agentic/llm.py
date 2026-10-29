"""Agent 使用的 LLM 封装：文本、结构化 JSON、流式三种调用。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content)


class AgentLLM:
    def __init__(self, model: ChatOllama) -> None:
        self._model = model

    async def text(self, prompt: str, *, system: str | None = None) -> str:
        messages: list[Any] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        result = await self._model.ainvoke(messages)
        return _content_text(result.content).strip()

    async def json(self, prompt: str, schema: type[BaseModel]) -> BaseModel | None:
        raw = await self.text(prompt)
        match = _JSON_BLOCK.search(raw)
        candidate = match.group(0) if match else raw
        try:
            return schema.model_validate(json.loads(candidate))
        except (ValueError, ValidationError):
            logger.warning("结构化输出解析失败: %s", raw[:200])
            return None

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        messages: list[Any] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        async for chunk in self._model.astream(messages):
            text = _content_text(chunk.content)
            if text:
                yield text
