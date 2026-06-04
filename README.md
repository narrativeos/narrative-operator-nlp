# Narrative Operator NLP

NLP operator for NarrativeOS — Protocol-First architecture with three standard protocol layers.

## Overview

`narrative-operator-nlp` provides NLP capabilities (tokenization, NER, dependency parsing, SRL, etc.) to the NarrativeOS ecosystem. It wraps [HanLP](https://github.com/hankcs/HanLP) as its underlying NLP engine and exposes analysis results through a unified **Narrative Schema Protocol (NSP)** format.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   External World                 │
│  LLM / Agent / Dify / LangGraph / Studio UI      │
├─────────────────────────────────────────────────┤
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
│   └── PROTOCOL.md        # NSP data structure standard
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

### Quick Start

```bash
cp .env.example .env
docker compose up -d
```

First run auto-downloads models (~500 MB) into a persistent volume.

### Service Modes

```bash
docker compose up -d nlp-http        # FastAPI only (default)
docker compose up -d nlp-grpc        # gRPC only
```

### Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_PORT` | `8000` | FastAPI host port |
| `GRPC_PORT` | `50051` | gRPC host port |
| `HANLP_MODEL_SET` | `MTL` | Models: MTL / LZH / PIPELINE / ALL |

### Verify

```bash
curl http://localhost:8000/health
```

Open `http://localhost:8000/demo` for interactive visualization.

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

- [Implementation Summary](docs/IMPLEMENTATION.md) — Full architecture & module reference
- [API Usage Guide](docs/API-USAGE.md) — MCP, gRPC, FastAPI call examples
- [Narrative Schema Protocol (NSP)](docs/PROTOCOL.md) — Data structure standard
- [Gap Analysis vs HanLP Demo](docs/GAP-ANALYSIS.md) — Feature comparison
- [Architecture: NLP Operator](https://github.com/narrativeos/narrative-docs/blob/main/architecture/operator-nlp/README.md) — Design rationale
- [HanLP Documentation](https://hanlp.hankcs.com/docs/) — Underlying NLP engine

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details. HanLP is also licensed under Apache 2.0.
