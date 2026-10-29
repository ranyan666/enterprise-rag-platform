"""Agentic RAG 评测：路由准确率、拒答准确率、引用命中、反思轮次与首字延迟。

用法：
    .\\.venv\\Scripts\\python.exe scripts\\evaluate_agentic.py

前置：服务已启动，且知识库中已导入 eval/corpus 语料（可先跑 evaluate_retrieval --reset --keep-docs）。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, Sequence

import httpx

DEFAULT_DATASET = Path("eval/agentic_dataset.jsonl")
DEFAULT_REPORT = Path(".tmp/agentic_report.json")
REFUSAL_MARKERS = ("未找到", "抱歉", "无法", "没有找到", "不能回答", "超出")


def percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(ratio * len(ordered)) - 1))
    return ordered[index]


def read_sse(response: httpx.Response) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if event_name is not None:
            try:
                events.append((event_name, json.loads("\n".join(data_lines))))
            except ValueError:
                pass
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


def ask(client: httpx.Client, question: str, session_id: str) -> dict:
    started = time.perf_counter()
    with client.stream(
        "POST",
        "/api/v1/chat",
        json={"question": question, "session_id": session_id},
    ) as response:
        response.raise_for_status()
        events = read_sse(response)
    route = next((data for name, data in events if name == "route"), {})
    condense = next((data for name, data in events if name == "condense"), {})
    rewrites = [data for name, data in events if name == "rewrite"]
    done = next((data for name, data in reversed(events) if name == "done"), {})
    return {
        "route": route.get("route"),
        "reason": route.get("reason"),
        "condensed": condense.get("search_query"),
        "rewrite_count": done.get("rewrite_count", len(rewrites)),
        "citations": done.get("citations", []),
        "answer": done.get("answer", ""),
        "ttft_ms": (done.get("timings") or {}).get("ttft_ms"),
        "total_ms": (done.get("timings") or {}).get("total_ms")
        or round((time.perf_counter() - started) * 1000, 2),
    }


def baseline_refused(client: httpx.Client, question: str) -> bool:
    with client.stream("POST", "/api/v1/query", json={"question": question, "limit": 3}) as response:
        response.raise_for_status()
        events = read_sse(response)
    done = next((data for name, data in reversed(events) if name == "done"), {})
    answer = str(done.get("answer", ""))
    return any(marker in answer for marker in REFUSAL_MARKERS)


def summarize_group(cases: Iterable[dict]) -> dict[str, float]:
    items = list(cases)
    total = len(items) or 1
    return {
        "count": len(items),
        "route_accuracy": sum(1 for item in items if item["route_correct"]) / total,
        "citation_hit@1": sum(1 for item in items if item["citation_top1"]) / total,
        "citation_hit_any": sum(1 for item in items if item["citation_hit"]) / total,
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description="Agentic RAG 评测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-baseline", action="store_true", help="跳过线性 RAG 拒答基线")
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        print("[FAIL] 评测集为空")
        return 1

    records: list[dict] = []
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=300.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json().get("status") != "ok":
            print("[FAIL] 依赖未就绪，先检查 /health")
            return 1

        for case in cases:
            session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
            for seed in case.get("seed_questions") or []:
                ask(client, seed, session_id)
            result = ask(client, case["question"], session_id)
            expected_route = case["expected_route"]
            citations = result["citations"] or []
            top1 = citations[0].get("filename") if citations else None
            expected_doc = case.get("expected_doc")
            record = {
                **case,
                **result,
                "route_correct": result["route"] == expected_route,
                "citation_top1": bool(expected_doc) and top1 == expected_doc,
                "citation_hit": bool(expected_doc) and any(
                    item.get("filename") == expected_doc for item in citations
                ),
            }
            records.append(record)
            print(
                f"[{case['id']:>2}] {case['type']:<12} route={result['route']:<7} "
                f"ttft={result['ttft_ms']}ms rewrite={result['rewrite_count']} "
                f"{'OK' if record['route_correct'] else 'MISS'}"
            )

        out_of_scope = [item for item in records if item["type"] == "out_of_scope"]
        baseline_refusals = 0
        if not args.skip_baseline and out_of_scope:
            for item in out_of_scope:
                if baseline_refused(client, item["question"]):
                    baseline_refusals += 1

        by_type = {
            name: summarize_group(item for item in records if item["type"] == name)
            for name in ("in_scope", "direct", "out_of_scope", "multi_turn", "hard")
        }
        in_scope_like = [
            item for item in records if item["type"] in {"in_scope", "multi_turn", "hard"}
        ]
        ttfts = [item["ttft_ms"] for item in records if item.get("ttft_ms") is not None]
        rewrites = [item["rewrite_count"] for item in in_scope_like]

        report = {
            "cases": len(records),
            "route_accuracy": sum(1 for item in records if item["route_correct"]) / len(records),
            "by_type": by_type,
            "refusal_accuracy": (
                sum(1 for item in out_of_scope if item["route"] == "refuse") / len(out_of_scope)
                if out_of_scope
                else None
            ),
            "baseline_refusal_accuracy": (
                baseline_refusals / len(out_of_scope) if out_of_scope else None
            ),
            "avg_rewrite_count": statistics.mean(rewrites) if rewrites else 0.0,
            "ttft_p50_ms": percentile(ttfts, 0.5),
            "ttft_p95_ms": percentile(ttfts, 0.95),
            "details": records,
        }

        print(f"\n=== Agentic RAG 评测（{len(records)} 条）===")
        print(f"总路由准确率: {report['route_accuracy']:.3f}")
        for name, metrics in by_type.items():
            print(
                f"{name:<12} n={metrics['count']:<2} route={metrics['route_accuracy']:.3f} "
                f"citation@1={metrics['citation_hit@1']:.3f}"
            )
        print(f"拒答准确率（Agentic）: {report['refusal_accuracy']:.3f}")
        if report["baseline_refusal_accuracy"] is not None:
            print(f"拒答准确率（线性 RAG 基线，关键词判定）: {report['baseline_refusal_accuracy']:.3f}")
        print(
            f"平均改写轮次: {report['avg_rewrite_count']:.2f} | "
            f"TTFT p50={report['ttft_p50_ms']:.0f}ms p95={report['ttft_p95_ms']:.0f}ms"
        )

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] 报告写入 {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
