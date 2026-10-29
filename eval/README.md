# 检索评测集

用于 `scripts/evaluate_retrieval.py` 的可复现评测资产。

## 目录

- `corpus/`：12 篇 Markdown 语料，覆盖权限、计费、部署、故障、数据字典、入职六个主题，
  并包含一组近似的错误码文档（`ERR-6101`~`ERR-6106`），用于模拟长尾技术标识符。
- `qa_set.jsonl`：26 条标注查询，每行一个 JSON：`{"query": "...", "expected_doc": "文件名.md"}`。

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

最近一次结果：hybrid 1.000 / BM25 0.923 / dense 1.000（hit@1），检索 p95 56.4ms。混合检索相对
BM25 提升 7.7pp，但稠密通道已满分，**没有测出相对 dense-only 的提升**。要验证简历中的
“长尾专有名词召回提升 20%”，需要把语料扩到数十篇以上、加入更多同义表述与近义干扰项；
当前 12 篇、26 条的规模不足以复现该数字。
