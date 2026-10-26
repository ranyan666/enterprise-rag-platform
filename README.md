# 企业级智能文档检索与问答平台

基于 RAG 架构的企业内部知识库问答平台：解析多格式业务文档并知识入库，结合大模型与私有数据回答业务咨询，缓解内部信息孤岛。

## 技术栈

- Python
- LangChain
- FastAPI
- Milvus
- Qwen 大模型
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

# 2. 拉取问答模型（约 4.7G）
ollama pull qwen2.5:7b-instruct

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

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试默认不依赖 Docker、GPU 与真实模型：外部依赖通过替身/mock 隔离。

## 目录结构

```
app/
  api/            # /health 与 /api/v1/documents
  core/           # 配置、日志
  repositories/   # SQLite 仓储（documents / chunks）
  services/       # 解析、父子分块、处理编排、依赖检查
tests/            # pytest 用例（解析 / 分块 / 仓储 / 接口）
docker-compose.yml # Milvus + etcd + MinIO + Redis
```

## 当前状态

MVP 已完成 M1 骨架与 M2 文档入库：配置、日志、健康检查、依赖编排，以及上传 → 解析 → 父子分块 → SQLite 落库 → 查询/删除的完整链路。M3 起接入 BGE-M3 向量化与 Milvus collection。
