# knowledge_base

基于 **FastAPI + LangGraph + Milvus + MinIO + MongoDB + 大模型** 的知识库系统，包含两条主链路：

- **导入链路**：上传 PDF/MD -> 解析与切分 -> 向量化 -> 入库
- **查询链路**：问题改写与商品名确认 -> 多路检索 -> 融合重排 -> 生成答案（支持 SSE 流式输出）

## 1. 功能概览

### 1.1 导入服务（`app/import_process/api/file_import_service.py`）

默认端口：`8000`

- `GET /import.html`：导入页面
- `POST /upload`：上传一个或多个文件，后台启动 LangGraph 导入任务
- `GET /status/{task_id}`：查询任务状态、运行节点、已完成节点

导入流程图（代码中的主流程）：

1. `node_entry`
2. `node_pdf_to_md`（PDF 场景）
3. `node_md_img`
4. `node_document_split`
5. `node_item_name_recognition`
6. `node_bge_embedding`
7. `node_import_milvus`

### 1.2 查询服务（`app/query_process/api/query_service.py`）

默认端口：`8001`

- `GET /chat.html`：聊天页面
- `GET /health`：健康检查
- `POST /query`：发起问答（支持同步/异步）
- `GET /stream/{session_id}`：SSE 流式输出
- `GET /history/{session_id}`：查询会话历史
- `DELETE /history/{session_id}`：清空会话历史

查询流程图（设计目标）：

1. `node_item_name_confirm`
2. `node_multi_search`（并发多路）
3. `node_search_embedding` / `node_search_embedding_hyde` / `node_web_search_mcp` / `node_query_kg`
4. `node_rrf`
5. `node_rerank`
6. `node_answer_output`

## 2. 目录结构

```text
knowledge_base/
├─ app/
│  ├─ clients/           # Milvus / MinIO / Mongo / Neo4j 客户端
│  ├─ conf/              # 各类环境配置映射
│  ├─ core/              # logger、prompt加载
│  ├─ import_process/    # 导入链路（API + LangGraph + 前端页）
│  ├─ query_process/     # 查询链路（API + LangGraph + 前端页）
│  ├─ lm/                # LLM、Embedding、Reranker 封装
│  ├─ utils/             # 任务状态、SSE、限流、路径工具
│  └─ tool/              # 模型下载脚本
├─ prompts/              # Prompt 模板
├─ test/                 # 测试脚本
├─ pyproject.toml
└─ .env                  # 本地环境变量（已被 gitignore 忽略）
```

## 3. 运行环境

- Python `>=3.11`
- 推荐使用 `uv` 或 `venv + pip`

外部依赖服务：

- Milvus
- MinIO
- MongoDB
- 大模型 API（OpenAI 兼容接口）
- MinerU（PDF 解析服务）
- 百炼 MCP（联网检索）
- （可选）Neo4j

## 4. 安装依赖

### 4.1 使用 uv（推荐）

```bash
uv sync
```

### 4.2 使用 pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

## 5. 环境变量

请在项目根目录配置 `.env`（示例）：

```env
# 项目路径
PROJECT_ROOT=D:\Python\project\PythonProject\PythonProject\knowledge_base

# LLM
OPENAI_API_BASE=https://your-openai-compatible-endpoint/v1
OPENAI_API_KEY=your_api_key
LLM_DEFAULT_MODEL=qwen3-32b
VL_MODEL=qwen-vl-plus
LLM_DEFAULT_TEMPERATURE=0.1

# Embedding
BGE_M3_PATH=BAAI/bge-m3
BGE_M3=BAAI/bge-m3
BGE_DEVICE=cuda:0
BGE_FP16=1

# Reranker
BGE_RERANKER_LARGE=BAAI/bge-reranker-large
BGE_RERANKER_DEVICE=cuda:0
BGE_RERANKER_FP16=1

# Milvus
MILVUS_URL=http://127.0.0.1:19530
CHUNKS_COLLECTION=kb_chunks
ENTITY_NAME_COLLECTION=kb_entity_names
ITEM_NAME_COLLECTION=kb_item_names

# MinIO
MINIO_ENDPOINT=127.0.0.1:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=kb-import-bucket
MINIO_IMG_DIR=images
MINIO_PDF_DIR=pdf_files

# MongoDB
MONGO_URL=mongodb://127.0.0.1:27017
MONGO_DB_NAME=knowledge_base

# MinerU
MINERU_BASE_URL=https://your-mineru-endpoint
MINERU_API_TOKEN=your_mineru_token

# 百炼 MCP
MCP_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v2/mcps/WebSearch/sse

# Neo4j（可选）
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# 日志
LOG_CONSOLE_ENABLE=True
LOG_CONSOLE_LEVEL=INFO
LOG_FILE_ENABLE=True
LOG_FILE_LEVEL=INFO
LOG_FILE_RETENTION=7 days
```

## 6. 启动方式

在项目根目录执行：

### 6.1 启动导入服务（8000）

```bash
python -m app.import_process.api.file_import_service
```

### 6.2 启动查询服务（8001）

```bash
python -m app.query_process.api.query_service
```

访问：

- 导入页面：`http://127.0.0.1:8000/import.html`
- 查询页面：`http://127.0.0.1:8001/chat.html`

## 7. 接口示例

### 7.1 上传文件

```bash
curl -X POST "http://127.0.0.1:8000/upload" ^
  -F "files=@doc\example.pdf"
```

### 7.2 查询导入状态

```bash
curl "http://127.0.0.1:8000/status/{task_id}"
```

### 7.3 发起查询（非流式）

```bash
curl -X POST "http://127.0.0.1:8001/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"这款设备如何设置\",\"is_stream\":false}"
```

### 7.4 发起查询（流式）

1. 先调 `/query` 获取 `session_id`
2. 再连接 `/stream/{session_id}` 接收 SSE 事件

## 8. 数据与日志

- 本地中间产物：`output/YYYYMMDD/{task_id}/...`
- 日志目录：`logs/`
- Prompt 模板：`prompts/*.prompt`

## 9. 代码现状说明（阅读仓库后）

当前仓库中有若干实现不一致问题，部署前建议先修复：

1. `app/query_process/agent/main_graph.py` 仍在引用 `kb.query_process...` 路径，和当前项目目录 `app.query_process...` 不一致。
2. `app/query_process/agent/node/node_answer_output.py` 中主函数名为 `node_import_kg`，但流程图引用的是 `node_answer_output`。
3. `app/query_process/agent/node/node_search_embedding.py` 中主函数名为 `node_pdf_to_md`，和节点命名不一致。
4. `app/import_process/agent/nodes/node_bge_embedding.py`、`node_import_milvus.py` 存在导入/函数体重复等问题，需补全整理后再上线。
5. `node_md_img.py` 使用了 `minio_config.minio_secure`，但 `MinIOConfig` 未声明该字段。

如果你计划把该仓库用于生产，建议先做一次完整可运行性修复（导入、查询、流式、入库全链路回归）。

---

如需扩展：

- 将任务状态从内存字典迁移到 Redis（支持多进程、多实例）
- 为导入链路和查询链路补齐自动化测试与健康检查
- 对 `query_process` 流程图做统一命名与模块路径清理
