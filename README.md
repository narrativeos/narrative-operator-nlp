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

### Install

```bash
pip install -e .
```

### Analyze Text

```python
from core.analyzer import analyze

doc = analyze("碳钢是钢的一种，具有高强度和高韧性。")
print(doc.model_dump_json(indent=2))
```

### Start Services

```bash
# FastAPI dev server
python adapters/fastapi_app.py

# MCP server
python adapters/mcp_server.py

# gRPC server
python adapters/grpc_server.py
```

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

- [Narrative Schema Protocol (NSP)](docs/PROTOCOL.md) — Data structure standard
- [Architecture: NLP Operator](https://github.com/narrativeos/narrative-docs/blob/main/architecture/operator-nlp/README.md) — Design rationale
- [HanLP Documentation](https://hanlp.hankcs.com/docs/) — Underlying NLP engine

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details. HanLP is also licensed under Apache 2.0.
