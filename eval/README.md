# 检索评测集

用于 `scripts/evaluate_retrieval.py` 的可复现评测资产。

## 目录

- `corpus/`：6 篇 Markdown 语料，覆盖权限、计费、部署、故障、数据字典、入职六个主题，
  包含中英文、数字编号与错误码等“长尾词”。
- `qa_set.jsonl`：20 条标注查询，每行一个 JSON：`{"query": "...", "expected_doc": "文件名.md"}`。

## 指标

- **hit@k**：目标文档出现在前 k 条结果中的比例（k=1/3/5）
- **MRR@10**：目标文档名次倒数的平均值，未出现在前 10 记为 0
- 三个系统分别计算：`hybrid`（融合排序）、`bm25`（BM25 通道名次）、`dense`（向量通道名次）

## 运行

```powershell
# 建议在干净实例上跑：--reset 会清空当前知识库（仅本地演示环境）
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --reset --keep-docs
```

`--keep-docs` 保留语料便于用 `/demo` 页面体验；不加则评测后自动删除。

## 扩展示例

1. 在 `corpus/` 增加 Markdown 文档；
2. 在 `qa_set.jsonl` 增加 `{"query": "...", "expected_doc": "新文档.md"}`；
3. 重新运行脚本，报告写入 `.tmp/eval_report.json`（含每条的 hybrid/bm25/dense 名次与延迟）。

## 已知局限

当前语料规模小（6 篇、20 条查询），稠密通道已接近满分，混合检索的优势主要体现在对 BM25 的补足
（英文查询、同义表述）。要验证简历中的“长尾专有名词召回提升 20%”，需要扩大语料到数十篇以上、
增加同名实体与错误码干扰项，再对比 hybrid 与 dense-only。
