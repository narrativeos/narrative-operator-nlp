# Narrative Operator NLP

NLP operator for NarrativeOS — Protocol-First architecture with three standard protocol layers.

## Overview

`narrative-operator-nlp` provides NLP capabilities (tokenization, NER, dependency parsing, SRL, etc.) to the NarrativeOS ecosystem. It wraps [HanLP](https://github.com/hankcs/HanLP) as its underlying NLP engine and exposes analysis results through a unified **Narrative Schema Protocol (NSP)** format.


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
HanLP is the multilingual NLP library designed for researchers and enterprises, built on PyTorch and TensorFlow 2.x to advance state-of-the-art deep learning techniques in academia and industry. HanLP was designed from day one to be
efficient, user-friendly and extendable.

Thanks to open-access corpora like Universal Dependencies and OntoNotes, HanLP 2.1 now offers 10 joint tasks on [130
languages](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/mtl.html#hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6): tokenization, lemmatization, part-of-speech tagging, token feature extraction, dependency parsing,
constituency parsing, semantic role labeling, semantic dependency parsing, abstract meaning representation (AMR)
parsing.

For end users, HanLP offers light-weighted RESTful APIs and native Python APIs.

## RESTful APIs

Tiny packages in several KBs for agile development and mobile applications. Although anonymous users are welcomed, an
auth key is suggested
and [a free one can be applied here](https://bbs.hankcs.com/t/apply-for-free-hanlp-restful-apis/3178) under
the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

<details>
  <summary>Click to expand tutorials for RESTful APIs</summary>

  ### Python

  ```bash
  pip install hanlp_restful
  ```

  Create a client with our API endpoint and your auth.

  ```python
  from hanlp_restful import HanLPClient
  HanLP = HanLPClient('https://hanlp.hankcs.com/api', auth=None, language='mul') # Support en, ja, zh, mul
  ```

  ### Java

  Insert the following dependency into your `pom.xml`.

  ```xml
  <dependency>
    <groupId>com.hankcs.hanlp.restful</groupId>
    <artifactId>hanlp-restful</artifactId>
    <version>0.0.15</version>
  </dependency>
  ```

  Create a client with our API endpoint and your auth.

  ```java
  HanLPClient HanLP = new HanLPClient("https://hanlp.hankcs.com/api", null, "mul"); // Support en, ja, zh, mul
  ```

  ### Quick Start

  No matter which language you use, the same interface can be used to parse a document.

  ```python
  HanLP.parse(
      "In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments. 2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。")
  ```

  See [docs](https://hanlp.hankcs.com/docs/tutorial.html) for visualization, annotation guidelines and more details.

</details>


## Native APIs

```bash
pip install hanlp
```

HanLP requires Python 3.6 or higher. While GPU or TPU acceleration is recommended, it is not mandatory.

### Quick Start

```bash
cp .env.example .env
docker compose up -d --build
```

- In particular, the Python `HanLPClient` can also be used as a callable function following the same semantics.
  See [docs](https://hanlp.hankcs.com/docs/tutorial.html) for visualization, annotation guidelines and more details.
- To process English, Chinese or Japanese, HanLP provides mono-lingual models in each language which significantly outperform the
  multilingual model. See [docs](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/index.html) for the list of models.

### Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_PORT` | `8000` | FastAPI host port |
| `HANLP_MODEL_SET` | `MTL` | Models: MTL / LZH / PIPELINE / ALL |

### Verify

```bash
curl http://localhost:8000/health
```

Open `http://localhost:8000` — redirects to interactive demo page.

> **MCP (Model Context Protocol)** uses stdio transport and is designed for local
> subprocess invocation by LLM frameworks (Claude Desktop, Dify, LangGraph).
> In Docker, use the HTTP API (`POST /analyze`) instead.

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

Copyright (c) 2025 北京九录科技有限公司. Licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

### Third-Party Components

| Component | License | Usage |
|-----------|---------|-------|
| [HanLP](https://github.com/hankcs/HanLP) | Apache 2.0 | NLP engine |
| ELECTRA-small (Chinese MTL) | Apache 2.0 | Modern Chinese pipeline |
| [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) (Answer.AI) | Apache 2.0 | English pipeline |
| KYOTO-EVAHAN (LZH) | CC BY 4.0 | Classical Chinese pipeline |

### Citation

If you use HanLP in your research, please cite:

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
