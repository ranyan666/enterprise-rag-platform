"""检索评测：在固定语料与标注集上对比 hybrid / BM25-only / dense-only。

用法：
    .\\.venv\\Scripts\\python.exe scripts\\evaluate_retrieval.py --reset

`--reset` 会先清空当前知识库（仅适合本地演示环境），再上传 eval/corpus 下的语料，
跑完评测后自动删除本次上传的文档。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import httpx

DEFAULT_CORPUS = Path("eval/corpus")
DEFAULT_DATASET = Path("eval/qa_set.jsonl")
DEFAULT_REPORT = Path(".tmp/eval_report.json")


def rank_for(results: Sequence[dict], filename: str, field: str) -> int | None:
    """返回命中目标文档的名次（1 起）；field=hybrid 用融合顺序，否则用该通道名次。"""

    if field == "hybrid":
        for index, item in enumerate(results, start=1):
            if item.get("filename") == filename:
                return index
        return None
    ranks = [
        item[field]
        for item in results
        if item.get("filename") == filename and item.get(field) is not None
    ]
    return min(ranks) if ranks else None


def hit(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None, k: int = 10) -> float:
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def ndcg(rank: int | None, k: int = 10) -> float:
    """二值相关性下的 nDCG@k（IDCG=1）。"""

    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(ratio * len(ordered)) - 1))
    return ordered[index]


def summarize(ranks: Iterable[int | None]) -> dict[str, float]:
    collected = list(ranks)
    total = len(collected) or 1
    return {
        "hit@1": sum(hit(rank, 1) for rank in collected) / total,
        "hit@3": sum(hit(rank, 3) for rank in collected) / total,
        "hit@5": sum(hit(rank, 5) for rank in collected) / total,
        "mrr@10": sum(reciprocal_rank(rank, 10) for rank in collected) / total,
        "ndcg@10": sum(ndcg(rank, 10) for rank in collected) / total,
    }


def wait_ready(client: httpx.Client, doc_ids: Sequence[str], timeout: float) -> None:
    deadline = time.time() + timeout
    pending = set(doc_ids)
    while pending and time.time() < deadline:
        for doc_id in list(pending):
            response = client.get(f"/api/v1/documents/{doc_id}")
            if response.status_code != 200:
                continue
            document = response.json()
            if document["status"] == "ready":
                pending.discard(doc_id)
            elif document["status"] == "failed":
                raise RuntimeError(f"文档 {document['filename']} 处理失败: {document.get('error')}")
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError(f"等待文档就绪超时: {sorted(pending)}")


def reset_knowledge_base(client: httpx.Client) -> None:
    documents: list[dict] = []
    offset = 0
    while True:
        response = client.get("/api/v1/documents", params={"limit": 100, "offset": offset})
        response.raise_for_status()
        page = response.json()["items"]
        documents.extend(page)
        if len(page) < 100:
            break
        offset += 100
    for document in documents:
        if document["status"] == "processing":
            continue
        client.delete(f"/api/v1/documents/{document['id']}")
    print(f"[*] 已清空 {len(documents)} 个文档")


def upload_corpus(client: httpx.Client, corpus_dir: Path) -> tuple[list[str], list[str]]:
    owned: list[str] = []
    reused: list[str] = []
    for path in sorted(corpus_dir.glob("*.md")):
        response = client.post(
            "/api/v1/documents",
            files={"file": (path.name, path.read_bytes(), "text/markdown")},
        )
        if response.status_code == 202:
            owned.append(response.json()["id"])
        elif response.status_code == 409:
            reused.append(response.json()["detail"]["existing_id"])
        else:
            raise RuntimeError(f"上传 {path.name} 失败: HTTP {response.status_code} {response.text}")
    return owned, reused


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 控制台不支持时忽略
            pass
    parser = argparse.ArgumentParser(description="检索评测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reset", action="store_true", help="评测前清空知识库（仅本地环境）")
    parser.add_argument("--keep-docs", action="store_true", help="评测后保留本次上传的语料")
    parser.add_argument("--rerank", action="store_true", help="追加 hybrid + 重排的对比（服务端需启用重排）")
    args = parser.parse_args()

    queries = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not queries:
        print("[FAIL] 评测集为空")
        return 1

    owned: list[str] = []
    latencies: list[float] = []
    details: list[dict] = []
    hybrid_ranks: list[int | None] = []
    bm25_ranks: list[int | None] = []
    dense_ranks: list[int | None] = []
    rerank_ranks: list[int | None] = []
    rerank_latencies: list[float] = []

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json().get("status") != "ok":
            print("[FAIL] 依赖未就绪，先检查 /health")
            return 1

        if args.reset:
            reset_knowledge_base(client)

        print(f"[*] 上传语料 {args.corpus_dir}")
        owned, reused = upload_corpus(client, args.corpus_dir)
        print(f"[*] 新建 {len(owned)} 篇，复用 {len(reused)} 篇，等待向量化完成")
        wait_ready(client, [*owned, *reused], timeout=180)

        for item in queries:
            started = time.perf_counter()
            baseline_payload = {
                "query": item["query"],
                "top_k": args.top_k,
                "limit": args.limit,
            }
            if args.rerank:
                # 显式关闭重排，避免服务端默认开启时把 baseline 也重排了
                baseline_payload["rerank"] = False
            response = client.post("/api/v1/search", json=baseline_payload)
            response.raise_for_status()
            latency = (time.perf_counter() - started) * 1000
            latencies.append(latency)

            results = response.json()["results"]
            expected = item["expected_doc"]
            hybrid_rank = rank_for(results, expected, "hybrid")
            bm25_rank = rank_for(results, expected, "bm25_rank")
            dense_rank = rank_for(results, expected, "dense_rank")
            hybrid_ranks.append(hybrid_rank)
            bm25_ranks.append(bm25_rank)
            dense_ranks.append(dense_rank)

            rerank_rank = None
            if args.rerank:
                rerank_started = time.perf_counter()
                rerank_response = client.post(
                    "/api/v1/search",
                    json={
                        "query": item["query"],
                        "top_k": args.top_k,
                        "limit": args.limit,
                        "rerank": True,
                    },
                )
                rerank_response.raise_for_status()
                rerank_latencies.append((time.perf_counter() - rerank_started) * 1000)
                rerank_rank = rank_for(rerank_response.json()["results"], expected, "hybrid")
                rerank_ranks.append(rerank_rank)

            details.append(
                {
                    "query": item["query"],
                    "expected_doc": expected,
                    "hybrid_rank": hybrid_rank,
                    "bm25_rank": bm25_rank,
                    "dense_rank": dense_rank,
                    "rerank_rank": rerank_rank,
                    "latency_ms": round(latency, 2),
                }
            )

        hybrid = summarize(hybrid_ranks)
        bm25 = summarize(bm25_ranks)
        dense = summarize(dense_ranks)
        reranked = summarize(rerank_ranks) if args.rerank else None

        print(f"\n=== 检索评测（{len(queries)} 条查询，top_k={args.top_k}）===")
        print(f"{'system':<10}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'mrr@10':>9}{'ndcg@10':>9}")
        rows = [("hybrid", hybrid), ("bm25", bm25), ("dense", dense)]
        if reranked is not None:
            rows.append(("hybrid+rr", reranked))
        for name, metrics in rows:
            print(
                f"{name:<10}{metrics['hit@1']:>8.3f}{metrics['hit@3']:>8.3f}"
                f"{metrics['hit@5']:>8.3f}{metrics['mrr@10']:>9.3f}{metrics['ndcg@10']:>9.3f}"
            )
        print(
            f"\n延迟: avg={statistics.mean(latencies):.1f}ms "
            f"p95={percentile(latencies, 0.95):.1f}ms"
        )
        if rerank_latencies:
            print(
                f"重排延迟: avg={statistics.mean(rerank_latencies):.1f}ms "
                f"p95={percentile(rerank_latencies, 0.95):.1f}ms"
            )
        print(
            "hybrid vs dense: "
            f"hit@5 {(hybrid['hit@5'] - dense['hit@5']) * 100:+.1f}pp, "
            f"mrr@10 {hybrid['mrr@10'] - dense['mrr@10']:+.3f}"
        )
        if reranked is not None:
            print(
                "hybrid+rerank vs hybrid: "
                f"hit@1 {(reranked['hit@1'] - hybrid['hit@1']) * 100:+.1f}pp, "
                f"mrr@10 {reranked['mrr@10'] - hybrid['mrr@10']:+.3f}, "
                f"ndcg@10 {reranked['ndcg@10'] - hybrid['ndcg@10']:+.3f}"
            )

        report = {
            "queries": len(queries),
            "top_k": args.top_k,
            "metrics": {"hybrid": hybrid, "bm25": bm25, "dense": dense},
            "latency": {
                "avg_ms": round(statistics.mean(latencies), 2),
                "p95_ms": round(percentile(latencies, 0.95), 2),
            },
            "corpus_docs": len(owned) + len(reused),
            "details": details,
        }
        if reranked is not None:
            report["metrics"]["hybrid_rerank"] = reranked
            report["rerank_latency"] = {
                "avg_ms": round(statistics.mean(rerank_latencies), 2),
                "p95_ms": round(percentile(rerank_latencies, 0.95), 2),
            }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] 报告已写入 {args.report}")

        if not args.keep_docs:
            for doc_id in owned:
                client.delete(f"/api/v1/documents/{doc_id}")
            print(f"[*] 已清理本次上传的 {len(owned)} 篇文档")

    return 0


if __name__ == "__main__":
    sys.exit(main())
