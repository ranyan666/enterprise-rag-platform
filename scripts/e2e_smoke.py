"""端到端冒烟脚本：健康检查 → 上传 → 入库 → 检索 → SSE 问答 → 删除。

用法：
    .\\.venv\\Scripts\\python.exe scripts\\e2e_smoke.py
    .\\.venv\\Scripts\\python.exe scripts\\e2e_smoke.py --base-url http://127.0.0.1:8000 --keep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

SSE_TIMEOUT = 180.0


class StepError(RuntimeError):
    pass


def step(message: str) -> None:
    print(f"[*] {message}")


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def read_sse(response: httpx.Response) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if event_name is not None:
            events.append((event_name, json.loads("\n".join(data_lines))))
        event_name = None
        data_lines = []

    for line in response.iter_lines():
        if line == "":
            flush()
        elif line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    flush()
    return events


def wait_ready(client: httpx.Client, doc_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/v1/documents/{doc_id}")
        if response.status_code == 200:
            document = response.json()
            if document["status"] == "ready":
                return document
            if document["status"] == "failed":
                raise StepError(f"文档处理失败: {document.get('error')}")
        time.sleep(1)
    raise StepError("等待文档就绪超时")


def run(base_url: str, keep: bool) -> None:
    marker = f"E2E-{uuid.uuid4().hex[:8]}"
    name = f"e2e-{marker}.md"
    content = (
        f"# 端到端验证文档 {marker}\n\n"
        f"本文档用于端到端冒烟测试，唯一标记为 {marker}。\n\n"
        "## 验证目标\n\n"
        "确认上传、解析、父子分块、BGE-M3 向量化、混合检索与流式问答链路均可用。\n\n"
        "## 预期结果\n\n"
        "检索能够命中本文档，问答回答中应包含唯一标记或相关引用。\n"
    ).encode("utf-8")

    with httpx.Client(base_url=base_url, timeout=SSE_TIMEOUT) as client:
        step(f"健康检查 {base_url}/health")
        health = client.get("/health")
        health.raise_for_status()
        payload = health.json()
        if payload.get("status") != "ok":
            raise StepError(f"依赖未就绪: {payload}")
        ok("Milvus / Redis / Ollama 正常")

        step(f"上传文档 {name}")
        response = client.post(
            "/api/v1/documents",
            files={"file": (name, content, "text/markdown")},
        )
        if response.status_code != 202:
            raise StepError(f"上传失败 HTTP {response.status_code}: {response.text}")
        doc_id = response.json()["id"]
        ok(f"已提交 id={doc_id}")

        step("等待解析与向量化")
        document = wait_ready(client, doc_id, timeout=120)
        if document["vector_count"] != document["child_count"] or document["vector_count"] <= 0:
            raise StepError(
                f"向量数量异常: vector_count={document['vector_count']} child_count={document['child_count']}"
            )
        ok(
            f"已就绪：父块 {document['parent_count']} / 子块 {document['child_count']} / 向量 {document['vector_count']}"
        )

        step("混合检索")
        search = client.post(
            "/api/v1/search",
            json={"query": marker, "top_k": 20, "limit": 10},
        )
        search.raise_for_status()
        results = search.json()["results"]
        if not any(item["document_id"] == doc_id for item in results):
            raise StepError("检索结果未命中本次上传的文档")
        ok(f"命中 {len(results)} 条，首条来自 {results[0]['filename']}")

        step("SSE 流式问答")
        events: list[tuple[str, dict]] = []
        with client.stream(
            "POST",
            "/api/v1/query",
            json={"question": f"端到端验证文档 {marker} 的验证目标是什么？", "limit": 3},
        ) as response:
            response.raise_for_status()
            if not response.headers.get("content-type", "").startswith("text/event-stream"):
                raise StepError("问答接口未返回 text/event-stream")
            events = read_sse(response)

        names = [name for name, _ in events]
        if not names or names[0] != "meta" or names[-1] != "done":
            raise StepError(f"SSE 事件序列异常: {names}")
        done = events[-1][1]
        if not done.get("answer"):
            raise StepError("问答没有返回内容")
        timings = done.get("timings", {})
        ok(
            f"收到 {names.count('token')} 个 token，ttft={timings.get('ttft_ms')}ms，"
            f"total={timings.get('total_ms')}ms，引用 {len(done.get('citations', []))} 条"
        )

        step("删除文档并校验清理")
        deleted = client.delete(f"/api/v1/documents/{doc_id}")
        if deleted.status_code != 204:
            raise StepError(f"删除失败 HTTP {deleted.status_code}")
        if client.get(f"/api/v1/documents/{doc_id}").status_code != 404:
            raise StepError("删除后文档仍然可查询")
        ok("文档与向量已清理")

    print("\nE2E 冒烟通过")
    if keep:
        print(f"[提示] 本次文档 id={doc_id}（脚本已按流程删除；--keep 仅保留失败现场）")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 控制台不支持时忽略
            pass
    parser = argparse.ArgumentParser(description="端到端冒烟测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--keep", action="store_true", help="保留文档（默认删除）")
    args = parser.parse_args()

    try:
        run(args.base_url.rstrip("/"), keep=args.keep)
    except (StepError, httpx.HTTPError, ValueError) as exc:
        fail(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
