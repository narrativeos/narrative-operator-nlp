# Deployment

`narrative-operator-nlp` 的 Docker 部署指南。本地开发安装见 [install.md](install.md)。

## 架构

单容器部署，FastAPI（HTTP）与 gRPC 服务运行在同一容器内：

- **FastAPI**：前台进程，监听容器内 `8000` 端口（映射到宿主机 `HTTP_PORT`）
- **gRPC**：后台进程，默认通过 Unix Domain Socket（`/tmp/narrative-operator-nlp.sock`）提供零开销本地 IPC；需要跨主机时切换为 TCP

镜像定义：`deploy/Dockerfile`（基于 `python:3.10-slim`，内置 apt/pip/HF 镜像加速）。
入口脚本：`deploy/docker-entrypoint.sh`，支持四种运行模式：

| 模式 | 说明 |
|------|------|
| `fastapi` | 仅 FastAPI（默认，含 Demo 页面与 Swagger UI） |
| `grpc` | 仅 gRPC |
| `both` | FastAPI 前台 + gRPC 后台（compose 默认） |
| `mcp` | 仅 MCP server（stdio 传输，供 LLM 框架本地拉起） |

## 快速开始

```bash
cp .env.example .env          # 按需修改配置
docker compose up -d --build  # 构建并启动
```

首次启动会自动检查并下载 **MTL（现代汉语，必选，~500MB）** 模型；古汉语（LZH）与单任务（PIPELINE）模型需提前下载到模型卷中（在宿主机执行 `python scripts/setup_models.py --model LZH` 等，将 `~/.hanlp` 内容同步到卷，或临时挂载后下载）。模型缓存在命名卷 `narrative-operator-nlp-models`（挂载到容器内 `/root/.hanlp`），重启容器不会重复下载。

## 配置（.env）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HTTP_PORT` | `8000` | FastAPI 宿主机端口（容器内固定 8000） |
| `GRPC_PORT` | `50051` | gRPC TCP 端口（宿主机 → 容器）；默认走 UDS 时不生效 |
| `HANLP_MODEL_SET` | `ALL` | 期望的模型集：`MTL` / `LZH` / `PIPELINE` / `ALL`。⚠️ 目前仅作文档约定，entrypoint 实际只自动下载 MTL；其他模型集需预先放入模型卷（见上文） |
| `HANLP_HOME` | `/root/.hanlp` | 容器内模型缓存目录 |
| `PYTHONUNBUFFERED` | `1` | Python 日志实时输出 |
| `TOKENIZERS_PARALLELISM` | `false` | 禁用 tokenizer 多进程（避免 fork 警告） |

完整示例见 [`.env.example`](../.env.example)。

## 端点

| 端点 | 说明 |
|------|------|
| `GET /` | 重定向到 `/demo` |
| `GET /health` | 健康检查（compose healthcheck 使用） |
| `POST /analyze` | NLP 分析主端点，返回 NSP 标准化结果 |
| `POST /analyze/dep` | 依存句法分析（SVG 渲染用） |
| `POST /analyze/discover` | 新词发现（PMI + ConvSeg） |
| `POST /analyze/pretty` | HanLP 原生可视化（to_pretty） |
| `GET /demo` | 交互式 Demo 页面（含 API 接口说明） |
| `GET /docs` | Swagger UI |
| `GET /redoc` | ReDoc |
| `GET /openapi.json` | OpenAPI JSON Schema |

调用示例：

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"碳钢是钢的一种，具有高强度。"}'
```

## gRPC 接入

- **同机 Worker Runtime（默认）**：Unix Domain Socket，零网络开销。Worker 连接 `/tmp/narrative-operator-nlp.sock`（同一 compose 网络内可通过 `GRPC_SOCKET` 环境变量指定路径）。
- **跨主机**：设置 `GRPC_PORT=50051` 并通过 TCP 连接。

Python 客户端示例：`adapters/grpc_client.py`，调用方式见 [api_usage.md](api_usage.md)。

## 验证

```bash
curl http://localhost:8000/health
```

浏览器打开 `http://localhost:8000` 查看交互式 Demo。

## MCP 说明

MCP（Model Context Protocol）使用 **stdio 传输**，设计为本地子进程调用（Claude Desktop、Dify、LangGraph 等 LLM 框架直接拉起 `python adapters/mcp_server.py`），**不适合通过 Docker 网络暴露**。Docker 部署场景下，LLM/Agent 集成请使用 HTTP API（`POST /analyze`）。

## 运维

```bash
docker compose logs -f        # 查看日志
docker compose restart        # 重启
docker compose down           # 停止（模型卷保留）
docker compose down -v        # 停止并删除模型卷（下次启动重新下载）
```

健康检查：每 30s 探测 `/health`，启动宽限期 60s（模型加载需要时间）。
