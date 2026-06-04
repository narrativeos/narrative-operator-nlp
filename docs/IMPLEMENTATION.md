# Narrative Operator NLP — 实现总结

> 最后更新: 2026-06-04 | 分支: `doc-zh`

## 项目定位

基于 HanLP 2.1.x fork 构建的 NarrativeOS NLP 算子，遵循文档 `narrative-docs/architecture/operator-nlp/README.md` 中定义的**协议优先架构**标准。

## 架构概览

```
                  ┌──────────────────────────────────┐
                  │         External World            │
                  │   LLM / Agent / Dify / Studio     │
                  └────┬──────────┬──────────┬────────┘
                       │ MCP      │ HTTP     │ gRPC
                  ┌────▼────┐┌───▼────┐┌───▼──────┐
                  │  MCP    ││FastAPI ││  gRPC    │
                  │ Server  ││ + Demo ││ Server   │
                  └────┬────┘└───┬────┘└───┬──────┘
                       │         │         │
                  ┌────▼─────────▼─────────▼──────┐
                  │       core/analyzer.py         │
                  │  Two-Stage Hybrid Pipeline     │
                  │  ┌──────────┐ ┌────────────┐  │
                  │  │ Modern   │ │ Classical  │  │
                  │  │ MTL      │ │ LZH        │  │
                  │  └────┬─────┘ └────┬───────┘  │
                  │       └──────┬─────┘          │
                  │       ┌──────▼─────┐          │
                  │       │  Mapper    │          │
                  │       │ HanLP→NSP  │          │
                  │       └──────┬─────┘          │
                  └──────────────┼────────────────┘
                                 │
                         NarrativeDocument (NSP)
```

## 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| **NSP Schema** | `core/schema.py` | Token/Entity/Relation/NarrativeDocument Pydantic 模型。11 实体类别 + 11 关系类型 |
| **实体映射** | `core/entity_mapper.py` | PKU/MSRA/OntoNotes 三套 NER 标注 → 统一 NSP 实体，含去重 |
| **关系抽取** | `core/relation_mapper.py` | dep 依存句法 + SRL 语义角色 → NSP 关系三元组 |
| **格式转换** | `core/mapper.py` | `HanlpSchemaMapper` — HanLP MTL 原始输出 → NarrativeDocument |
| **混合分析** | `core/analyzer.py` | 断句→语言检测→模型路由→质量评估→回退→偏移重映射→合并 |
| **语言检测** | `core/language_detector.py` | 多特征置信度评分，分类 modern/classical |
| **契约定义** | `schemas/narrative.proto` | gRPC Protobuf 服务 + 消息定义 |
| **契约定义** | `schemas/narrative.schema.json` | NSP 输出 JSON Schema 验证 |

## 协议适配器

| 适配器 | 端点 | 协议 | 用途 |
|--------|------|------|------|
| **FastAPI** | `POST /analyze` | HTTP/JSON | NSP 结构化结果 |
| | `POST /analyze/pretty` | HTTP/JSON | HanLP `to_pretty()` 原生可视化 |
| | `POST /analyze/dep` | HTTP/JSON | 依存树数据（供 SVG 渲染） |
| | `GET /demo` | HTML | 交互式可视化页面 |
| | `GET /docs` | HTML | Swagger UI |
| | `GET /health` | HTTP | 健康检查 |
| **gRPC** | `Analyze` RPC | Protobuf/UDS | 内部高性能总线 |
| **MCP** | `analyze_text` Tool | JSON-RPC/stdio | LLM/Agent 标准接口 |

## Demo 可视化

`GET /demo` 自包含 HTML 页面，四个 tab：

| Tab | 内容 |
|-----|------|
| 📊 NSP 结构化 | 分词（POS 彩色标注）+ 实体卡片 + 关系三元组 |
| 🎨 HanLP 原生 | `to_pretty()` ASCII 艺术，含依存树/NER/SRL/成分树 |
| 🧬 依存树 SVG | 内联 SVG：彩色节点 + 曲线箭头 + 关系标签 |
| { } JSON Raw | 原始 NSP JSON |

含三个预设示例（材料+地点 / 组织机构 / 多任务），页面加载自动分析。

## 两阶段混合分析

```
输入: "孔子曰：学而时习之。小明在北大读书。"
  │
  ├─ 断句 → ["孔子曰：学而时习之。", "小明在北大读书。"]
  │
  ├─ 语言检测
  │   ├─ "孔子曰：学而时习之。" → classical (confidence: 0.55)
  │   └─ "小明在北大读书。"     → modern    (confidence: 0.00)
  │
  ├─ 分别处理
  │   ├─ classical → KYOTO_EVAHAN LZH 模型 → 键名归一化 → NSP
  │   └─ modern    → ELECTRA_SMALL MTL     → NSP
  │
  └─ 合并 → offset 重映射 → 单一 NarrativeDocument
       meta.source = "hanlp_v2+hanlp_lzh"
```

## 模型管理

`scripts/setup_models.py` 统一管理 6 个预训练模型：

| 模型 | 类别 | 大小 | 必选 |
|------|------|------|------|
| `CLOSE_..._ELECTRA_SMALL_ZH` | MTL | ~500MB | ✅ |
| `KYOTO_EVAHAN_..._LZH` | LZH | ~300MB | 推荐 |
| `FINE_ELECTRA_SMALL_ZH` | Pipeline | ~200MB | 可选 |
| `CTB9_POS_ELECTRA_SMALL` | Pipeline | ~200MB | 可选 |
| `MSRA_NER_ELECTRA_SMALL_ZH` | Pipeline | ~200MB | 可选 |
| `CTB9_DEP_ELECTRA_SMALL` | Pipeline | ~200MB | 可选 |

## 部署

- **本地开发**: Python 3.10 + uv venv
- **Docker**: `docker compose up -d` — 自动检查下载模型
- **端口**: HTTP_PORT / GRPC_PORT 通过 `.env` 配置
- **模型持久化**: Docker Volume `hanlp-models:/root/.hanlp`

## 测试

```bash
pytest tests/ -q
# 62 passed (schema 17 + mapper 12 + adapters 18 + HanLP MTL 15)
```

## 基线门禁 (9/9 ✅)

- [x] `docs/PROTOCOL.md`
- [x] `schemas/narrative.proto`
- [x] `schemas/narrative.schema.json`
- [x] `examples/call_via_mcp.py`
- [x] `examples/call_via_http.py`
- [x] `core/analyzer.py`
- [x] `adapters/mcp_server.py`
- [x] `adapters/grpc_server.py`
- [x] `adapters/fastapi_app.py`
