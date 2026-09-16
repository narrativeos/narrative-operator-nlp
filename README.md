# Narrative Operator NLP

NLP operator for NarrativeOS — Protocol-First architecture with three standard protocol layers.

## Overview

`narrative-operator-nlp` provides NLP capabilities (tokenization, NER, dependency parsing, SRL, etc.) to the NarrativeOS ecosystem. It wraps [HanLP](https://github.com/hankcs/HanLP) as its underlying NLP engine and exposes analysis results through a unified **Narrative Schema Protocol (NSP)** format.

```
┌─────────────────────────────────────────────────┐
│  ① MCP (Model Context Protocol)                   │
│     → Standard interface for LLM/Agent            │
├─────────────────────────────────────────────────┤
│  ② gRPC (Protobuf)                                │
│     → Internal high-performance bus               │
│     → Unix Domain Socket                          │
├─────────────────────────────────────────────────┤
│  ③ FastAPI + JSON                                 │
│     → Dev/debug interface with Swagger UI         │
└─────────────────────────────────────────────────┘
```

## Project Structure

```
narrative-operator-nlp/
├── core/                  # Protocol-free NLP logic
│   ├── schema.py          # NSP Pydantic data models
│   ├── analyzer.py        # Unified analyze(text) → NarrativeDocument
│   ├── mapper.py          # HanlpSchemaMapper
│   ├── entity_mapper.py   # Entity mapping rules
│   └── relation_mapper.py # Relation extraction rules
├── adapters/              # Protocol adapters (one per protocol)
│   ├── mcp_server.py      # MCP Tool: analyze_text
│   ├── grpc_server.py     # gRPC NarrativeService
│   ├── grpc_client.py     # gRPC client for Worker Runtime
│   └── fastapi_app.py     # FastAPI POST /analyze
├── schemas/               # Core contracts
│   ├── narrative.proto    # Protobuf definition
│   └── narrative.schema.json  # JSON Schema
├── hanlp/                 # HanLP NLP engine (fork)
├── examples/              # Usage examples
│   ├── call_via_mcp.py
│   └── call_via_http.py
├── docs/
│   └── protocol.md        # NSP data structure standard
└── tests/
```

## Quick Start

### Prerequisites

- Python 3.10 (required by HanLP)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### 1. Create Virtual Environment

```bash
uv venv --python 3.10
source .venv/bin/activate
```

### 2. Install

```bash
uv pip install -e ".[narrative]"
```

### 3. Download Pre-trained Models

```bash
# Download all models (MTL + Classical Chinese + Pipeline)
python scripts/setup_models.py

# Check what's already downloaded
python scripts/setup_models.py --check

# Download specific model sets
python scripts/setup_models.py --model MTL      # Modern Chinese only
python scripts/setup_models.py --model LZH      # Classical Chinese only
python scripts/setup_models.py --model PIPELINE  # Single-task models
```

Models are cached in `~/.hanlp/` by default.

### 4. Verify & Analyze

```python
from core.analyzer import analyze

doc = analyze("碳钢是钢的一种，具有高强度和高韧性。")
print(doc.model_dump_json(indent=2))
```

### Start Services

```bash
# FastAPI dev server (with Swagger UI at /docs)
python adapters/fastapi_app.py

# MCP server (for LLM/Agent integration)
python adapters/mcp_server.py

# gRPC server (for Worker Runtime IPC)
python adapters/grpc_server.py
```

## Docker Deployment

Single-container deployment bundling the FastAPI and gRPC services.

```bash
cp .env.example .env
docker compose up -d --build
```

### Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_PORT` | `8000` | FastAPI host port |
| `GRPC_PORT` | `50051` | gRPC TCP port (default uses Unix Domain Socket) |
| `HANLP_MODEL_SET` | `MTL` | Model set auto-downloaded at startup: MTL / LZH / PIPELINE / ALL (see [docs/deployment.md](docs/deployment.md)) |

### Verify

```bash
curl http://localhost:8000/health
```

Open `http://localhost:8000` — redirects to the interactive demo page (`/demo`).

> **MCP (Model Context Protocol)** uses stdio transport and is designed for local
> subprocess invocation by LLM frameworks (Claude Desktop, Dify, LangGraph).
> In Docker, use the HTTP API (`POST /analyze`) instead.

## Protocol Selection

| Scenario | Protocol | Reason |
|----------|----------|--------|
| LLM/Agent calls | MCP | Standard interface |
| Workflow orchestration (Dify/LangGraph) | MCP | Ecosystem compatibility |
| Inter-operator communication | gRPC + UDS | High performance, low latency |
| Manual NLP testing | FastAPI + JSON | Quick validation |
| Studio UI prototyping | FastAPI + JSON | Fast iteration |
| Cross-language (Rust↔Python) | gRPC + Protobuf | Type-safe, high performance |

## Related Documentation

- [Install Guide](docs/install.md) — 本地安装与模型下载
- [Deployment Guide](docs/deployment.md) — Docker 部署（单容器 FastAPI + gRPC）
- [Configuration](docs/configure.md) — 抽取规则 / 部署 / 引擎三层配置
- [Implementation Summary](docs/implementation.md) — Full architecture & module reference
- [API Usage Guide](docs/api_usage.md) — MCP, gRPC, FastAPI call examples
- [Narrative Schema Protocol (NSP)](docs/protocol.md) — Data structure standard
- [Gap Analysis vs HanLP Demo](docs/gap_analysis.md) — Feature comparison
- [Architecture: NLP Operator](https://github.com/narrativeos/narrative-docs/blob/main/architecture/operator-nlp/README.md) — Design rationale
- [HanLP Documentation](https://hanlp.hankcs.com/docs/) — Underlying NLP engine

## License

Copyright (c) 2025 北京九录科技有限公司. Licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

### Third-Party Components

| Component | License | Usage |
|-----------|---------|-------|
| [HanLP](https://github.com/hankcs/HanLP) | Apache 2.0 | NLP engine |
| ELECTRA-small (Chinese MTL) | Apache 2.0 | Modern Chinese pipeline |
| [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) (Answer.AI) | Apache 2.0 | English pipeline |
| KYOTO-EVAHAN (LZH) | CC BY 4.0 | Classical Chinese pipeline |

### Citation

This project builds on [HanLP](https://github.com/hankcs/HanLP). If you use the underlying HanLP engine in your research, please also cite:

```bibtex
@inproceedings{he-choi-2021-stem,
    title = "The Stem Cell Hypothesis: Dilemma behind Multi-Task Learning with Transformer Encoders",
    author = "He, Han and Choi, Jinho D.",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.451",
}
```
