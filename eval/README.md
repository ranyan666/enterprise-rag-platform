# 检索评测集

用于 `scripts/evaluate_retrieval.py` 的可复现评测资产。

## 目录

- `corpus/`：28 篇 Markdown 语料，覆盖权限、计费、部署、故障、数据字典、入职六个主题，
  另外包含 6 篇近似的服务接口说明、6 篇错误码文档（`ERR-6101`~`ERR-6106`）、
  以及安全分级、审计、密钥、监控、灰度、报销、设备、三张表结构等运营文档。
- `qa_set.jsonl`：46 条标注查询，每行一个 JSON：`{"query": "...", "expected_doc": "文件名.md"}`。

> `corpus/` 是人工构造的合成语料，用于稳定复现检索指标，不代表真实业务文档分布。

## 指标

- **hit@k**：目标文档出现在前 k 条结果中的比例（k=1/3/5）
- **MRR@10**：目标文档名次倒数的平均值，未出现在前 10 记为 0
- **nDCG@10**：二值相关性下的归一化折损累计增益
- 三个系统分别计算：`hybrid`（融合排序）、`bm25`（BM25 通道名次）、`dense`（向量通道名次）；
  加 `--rerank` 时额外计算 `hybrid+rr`

## 运行

```powershell
# 建议在干净实例上跑：--reset 会清空当前知识库（仅本地演示环境）
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --reset --keep-docs

# 同时对比二阶段重排（服务端需先启用 RERANKER_ENABLED=true）
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --reset --keep-docs --rerank
```

`--keep-docs` 保留语料便于用 `/demo` 页面体验；不加则评测后自动删除。

## 扩展示例

1. 在 `corpus/` 增加 Markdown 文档；
2. 在 `qa_set.jsonl` 增加 `{"query": "...", "expected_doc": "新文档.md"}`；
3. 重新运行脚本，报告写入 `.tmp/eval_report.json`（含每条查询的名次与延迟）。

## 最近结果

46 条查询、28 篇语料、`top_k=50`：

| 系统 | hit@1 | hit@5 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| hybrid | 0.913 | 0.978 | 0.948 | 0.961 |
| BM25-only | 0.848 | 0.957 | 0.890 | 0.907 |
| dense-only | **0.957** | 1.000 | **0.978** | **0.984** |
| hybrid + cross-encoder 重排 | 0.935 | 1.000 | 0.964 | 0.973 |

## 已知局限

- 稠密通道在本语料上反而最强，hybrid 被 BM25 噪声拖累；重排能把 hybrid 拉高 2.2pp，
  但仍未超过 dense-only，且 CPU 上每次查询多约 1.1s。
- 因此**没有测出“混合比纯向量提升 20%”**。要复现简历里的数字，需要调融合权重
  （给 dense 更高权重）、扩大语料与查询规模，并优化重排候选集。
