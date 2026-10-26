# 企业级智能文档检索与问答平台

基于 RAG 架构的企业内部知识库问答平台：解析多格式业务文档并知识入库，结合大模型与私有数据回答业务咨询，缓解内部信息孤岛。

## 技术栈

- Python
- LangChain
- FastAPI
- Milvus
- Qwen 大模型
- BGE-M3 嵌入（Ollama 本地推理）
- jieba + rank_bm25（关键词检索）
- Redis

## 规划能力

- 多格式文档解析与知识入库
- BM25 关键词检索 + Milvus 稠密向量检索，RRF 融合双路召回
- 父子文档检索，兼顾上下文连贯性与答案准确性
- FastAPI 异步接口 + SSE 流式输出

## 快速开始

前置条件：Python 3.10、Docker Desktop、已启动的 Ollama 服务。

```powershell
# 1. 启动 Milvus / Redis 依赖
docker compose up -d

# 2. 拉取问答与嵌入模型（约 4.7G + 1.2G）
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3

# 3. 安装依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 4. 可选：复制配置
Copy-Item .env.example .env

# 5. 启动服务
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# 6. 验证依赖状态
Invoke-RestMethod http://127.0.0.1:8000/health
```

> **镜像拉取说明**
> - 国内网络无法访问 Docker Hub 时，可在 Docker Desktop 的 *Settings → Docker Engine* 中配置
>   `registry-mirrors`（例如 `https://docker.1ms.run`），保存后重启引擎。
> - MinIO 官方镜像已于 2025 年从 Docker Hub 下架，`docker-compose.yml` 因此改用 Bitnami 归档镜像
>   `bitnamilegacy/minio`（服务端仍是 MinIO，接口与凭证不变）。

> **模型下载慢时**：`bge-m3` 可从 ModelScope 下载同一份 GGUF（`gpustack/bge-m3-GGUF`，FP16 与
> Ollama 官方模型一致），再执行 `ollama create bge-m3 -f Modelfile`（Modelfile 内容为
> `FROM ./bge-m3-FP16.gguf`）离线导入。

接口文档：<http://127.0.0.1:8000/docs>

`GET /health` 返回 `status=ok` 表示 Milvus / Redis / Ollama 全部可达；任一依赖异常时为 `degraded`，并在 `dependencies[].detail` 中给出原因。接口本身始终返回 200，便于排障。

## 文档入库

支持 PDF / DOCX / Markdown / TXT，单文件默认上限 20 MB。上传后异步解析并做父子分块：
父块按标题层级与段落边界切分（默认 1800 字），子块在父块内滑窗切分（默认 400 字、重叠 80 字），
子块通过 `parent_id` 指回父块，检索命中后可用父块回填上下文。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/documents` | multipart 上传，返回 202 + 文档 id |
| GET | `/api/v1/documents` | 列表（`limit` / `offset` / `status`） |
| GET | `/api/v1/documents/{id}` | 详情（状态、块数、解析错误） |
| GET | `/api/v1/documents/{id}/chunks` | 分块查询（`level=parent\|child`） |
| DELETE | `/api/v1/documents/{id}` | 删除文档与磁盘文件 |

```powershell
$resp = curl.exe -s -X POST http://127.0.0.1:8000/api/v1/documents -F "file=@.tmp/sample.md"
$id = ($resp | ConvertFrom-Json).id
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/documents/$id"
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/documents/$id/chunks?level=child"
curl.exe -s -X DELETE "http://127.0.0.1:8000/api/v1/documents/$id"
```

状态机：`pending → processing → ready | failed`，失败原因写入 `error` 字段。
文件与元数据分别落在 `data/uploads/{id}/` 与 `data/app.sqlite3`，均已在 `.gitignore` 中。
已知限制：扫描件不做 OCR；DOCX 表格按行拍平为普通段落；PDF 标题识别依赖编号/章节样式，不还原复杂版式。

## 向量入库

子块解析完成后会调用本地 `bge-m3`（基于 Ollama，1024 维）生成向量，写入 Milvus collection
`document_chunks`（HNSW + COSINE，字段与 SQLite 子块一一对应）。父块不向量化，检索命中子块后
按 `parent_id` 回 SQLite 取完整上下文。

- 文档详情新增 `vector_count`，应等于 `child_count`
- `POST /api/v1/documents/{id}/reindex` 可重新解析并写入向量（嵌入失败、Milvus 恢复后使用）
- 删除文档会同时删除 SQLite 记录、磁盘文件与 Milvus 向量
- 嵌入/写向量失败时文档状态为 `failed`，错误原因写入 `error`，修复后调 `reindex` 即可
- collection 已存在但维度与 `EMBEDDING_DIM` 不一致时会拒绝写入并报错，避免写坏数据
- 更换嵌入模型时需同步修改 `EMBEDDING_MODEL`/`EMBEDDING_DIM` 并重建 collection

```powershell
# 重新索引（例如 Milvus 重启后）
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/api/v1/documents/$id/reindex"
```

## 混合检索

`POST /api/v1/search` 同时跑两路召回，再用 RRF 融合，最后回填父块：

- **BM25 关键词**：jieba 搜索模式分词 + 停用词过滤，内存索引，按仓储指纹自动重建，只索引 ready 文档
- **稠密向量**：查询文本经 `bge-m3` 编码后在 Milvus 按 COSINE 检索
- **RRF 融合**：`score = Σ 1 / (k + rank)`，默认 `k=60`，避免两种分数量纲不可比
- **父块回填**：命中子块后按 `parent_id` 回 SQLite 取完整父段，`parent_text` 用于后续生成

```powershell
$body = @{ query = "权限模型包含哪些内容"; limit = 5 } | ConvertTo-Json
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/v1/search -Body $body -ContentType "application/json"
```

返回字段：`results[]`（`parent_text`/`child_text`/`score`/`bm25_rank`/`dense_rank`/来源文档）、
`timings`（两路召回与融合耗时）、`candidates`（各阶段候选数）。可调参数：
`RETRIEVAL_TOP_K`（每路召回数）、`RETRIEVAL_RESULT_LIMIT`（返回条数）、`RRF_K`。
接口支持单次覆盖 `top_k` / `limit`。BM25 索引为进程内缓存，暂不支持多 worker 部署。

## 问答（SSE）

`POST /api/v1/query` 先走混合检索，再把父块按 `[1] [2] …` 编号拼进 prompt，调用本地 Qwen 流式作答。
接口返回 `text/event-stream`，事件类型：

| 事件 | 数据 | 说明 |
| --- | --- | --- |
| `meta` | `query`、`citations[]`、`retrieval_ms`、`candidates` | 引用清单（文件名/章节/页码/分数）与检索耗时 |
| `token` | `text` | 增量文本，可多次 |
| `done` | `answer`、`citations[]`、`timings`、`eval_count` | 完整答案与 `retrieval_ms` / `ttft_ms` / `total_ms` |
| `error` | `message` | 检索或模型异常 |

```powershell
$json = '{"question":"权限模型包含什么","limit":5}'
[System.IO.File]::WriteAllText("$PWD\.tmp\q.json", $json, [System.Text.UTF8Encoding]::new($false))
curl.exe -N -X POST http://127.0.0.1:8000/api/v1/query -H "Content-Type: application/json" --data-binary "@$PWD\.tmp\q.json"
```

- prompt 要求模型只依据资料作答、用 `[n]` 标注来源；资料不足时明确回答不知道
- 知识库为空或检索无结果时，不调用模型，直接流式返回固定提示
- 模型异常会以 `error` 事件收尾，不会中断 HTTP 连接
- 可调参数：`QA_CONTEXT_LIMIT`（引用条数）、`QA_CONTEXT_CHARS`（每条字符上限）、
  `QA_MAX_TOKENS`、`QA_TEMPERATURE`、`OLLAMA_CHAT_TIMEOUT_SECONDS`

## 演示页面

<http://127.0.0.1:8000/demo> 提供最小可用界面：上传文档 → 轮询处理状态 → 提问 → 流式答案与引用列表，
并可查看/删除已入库文档。纯静态 HTML/CSS/JS，无构建步骤，由 FastAPI 托管在 `/static`。

## 脚本与评测

```powershell
# 端到端冒烟：健康检查 → 上传 → 入库 → 检索 → SSE 问答 → 删除
.\.venv\Scripts\python.exe scripts\e2e_smoke.py

# 检索评测：清空本地知识库（--reset）、上传 eval/corpus、跑标注查询并输出报告
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --reset --keep-docs
```

最近一次评测（20 条标注查询、6 篇语料、`top_k=50`，报告在 `.tmp/eval_report.json`）：

| 系统 | hit@1 | hit@3 | hit@5 | MRR@10 |
| --- | --- | --- | --- | --- |
| hybrid | 1.000 | 1.000 | 1.000 | 1.000 |
| BM25-only | 0.900 | 0.950 | 0.950 | 0.925 |
| dense-only | 1.000 | 1.000 | 1.000 | 1.000 |

检索延迟 avg 53.9ms / p95 59.4ms。结论：混合检索在本语料上达到满分，相对 BM25 的 top-1 提升 10pp；
稠密通道本身已经很强，因此“混合比纯向量提升 20%”需要更大、更偏长尾专有名词的评测集来验证，
评测脚本与标注格式已就绪（见 `eval/README.md`）。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试默认不依赖 Docker、GPU 与真实模型：外部依赖通过替身/mock 隔离。

## 目录结构

```
app/
  api/            # /health、/api/v1/documents、/api/v1/search、/api/v1/query
  core/           # 配置、日志
  providers/      # 嵌入（Ollama bge-m3）与对话（Qwen /api/chat）提供者
  repositories/   # SQLite 仓储（documents / chunks）
  services/       # 解析、父子分块、向量库、混合检索、RAG 问答、处理编排
  static/         # 演示页（index.html / app.css / app.js）
tests/            # pytest 用例（解析 / 分块 / 仓储 / 接口 / 向量 / 检索 / 问答 / 评测指标）
scripts/          # e2e_smoke.py、evaluate_retrieval.py
eval/             # corpus/ 语料与 qa_set.jsonl 标注集
docker-compose.yml # Milvus + etcd + MinIO + Redis
```

## 当前状态

MVP 六个里程碑全部完成：M1 骨架、M2 文档入库、M3 向量入库、M4 混合检索、M5 流式问答、M6 演示与评测。
链路：上传 → 解析 → 父子分块 → SQLite → BGE-M3 → Milvus → BM25 + 稠密 + RRF → Qwen 带引用 SSE 回答；
配套 `/demo` 页面、E2E 冒烟脚本与可复现的检索评测。
