# Contributing Guide

Thank you for being interested in contributing to `narrative-operator-nlp`! You
are awesome ✨.

This guideline contains information about our conventions around coding style, pull request workflow, commit messages and more.

This page also contains information to help you get started with development on this
project.

## Development

### Set-up

Get the source code of this project using git:

```bash
git clone https://github.com/narrativeos/narrative-operator-nlp
cd narrative-operator-nlp
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[narrative]"
```

To work on this project, you need Python 3.10.

Download the pre-trained models required by the HanLP engine:

```bash
python scripts/setup_models.py
```

### Running Tests

This project has a test suite to ensure certain important APIs work properly. The tests can be run using:

```bash
pytest tests
```

```{tip}
It's hard to cover every API especially those of deep learning models, due to the limited computation resource of CI. However, we suggest all inference APIs to be tested at least.
```

## Repository Structure

This repository is split into a few critical folders:

core/
: Protocol-free NLP logic. `schema.py` defines the NSP (Narrative Schema Protocol) Pydantic models, `analyzer.py` is the unified `analyze(text) → NarrativeDocument` entry point, and the entity/relation extraction components live alongside.

adapters/
: Protocol adapters, one per protocol — `mcp_server.py` (MCP), `grpc_server.py` / `grpc_client.py` (gRPC + Protobuf), `fastapi_app.py` (FastAPI + JSON).

schemas/
: Core contracts — `narrative.proto` (Protobuf) and `narrative.schema.json` (JSON Schema), plus generated `narrative_pb2*.py` files.

config/
: YAML/TXT configuration — entity mapping rules, merge rules, domain keywords, custom dictionaries, and classical Chinese resources.

hanlp/
: A vendored fork of the [HanLP](https://github.com/hankcs/HanLP) NLP engine. It keeps the upstream `hanlp` package name and its own version (`hanlp/version.py`). Treat it as a dependency: do not mix project-level code into it, and keep the boundary between this project and the engine explicit.

plugins/
: Upstream HanLP helper packages (`hanlp_common`, `hanlp_trie`, `hanlp_restful`, ...) kept for engine compatibility.

docs/
: The documentation for this project, in markdown format. File names use lowercase `snake_case`. The build configuration is contained in `conf.py`.

tests/
: Testing infrastructure that uses `pytest` to ensure the output of the API is what we expect it to be.

.github/
: Contains Continuous-integration (CI) workflows, run on commits/PRs to the GitHub repository.

## Naming Conventions

- Python modules: lowercase `snake_case` (e.g. `entity_mapper.py`).
- Classes: `PascalCase` (e.g. `EntityMerger`, `NarrativeDocument`).
- Functions/variables: `snake_case` (e.g. `analyze`, `summarize_text`).
- Documentation files: lowercase `snake_case` (e.g. `api_usage.md`, `protocol.md`).
- The `hanlp/` engine fork keeps its upstream naming; project code lives in `core/`, `adapters/`, and `schemas/`.

