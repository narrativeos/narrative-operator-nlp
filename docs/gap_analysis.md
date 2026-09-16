# 与 HanLP 官方 Demo 的差异报告

> 对比对象: https://hanlp.hankcs.com/demos/  |  日期: 2026-06-04

## 概览

HanLP 官方 demo 站点提供 **18 类 NLP 任务** 的在线演示，分为**图形界面**（点击分析按钮）和**代码框**（Jupyter Notebook 在线执行）两种交互方式。narrative-operator-nlp 在此基础上封装了协议适配层和 NSP 标准化输出，聚焦于 **NarrativeOS 叙事分析场景**。

## 功能覆盖对比

| NLP 任务 | 官方 Demo | narrative-operator-nlp | 备注 |
|----------|-----------|----------------------|------|
| 中文分词 | ✅ GUI + Code | ✅ MTL `tok/fine` + `tok/coarse` | 一致 |
| 词性标注 | ✅ CTB/PKU/863 | ✅ `pos/ctb` + `pos/pku` + `pos/863` | 一致 |
| 命名实体识别 | ✅ MSRA/PKU/OntoNotes | ✅ 三标注集统一 → NSP Entity | **增强**: 去重合并 |
| 依存句法分析 | ✅ GUI + SVG | ✅ dep tuple → NSP Relation + SVG | 一致 |
| 成分句法分析 | ✅ Tree | ✅ con Tree | 一致 |
| 语义依存分析 | ✅ SDP | ✅ sdp (原始透传) | 未映射到 NSP |
| 语义角色标注 | ✅ SRL | ✅ srl → NSP Relation (ARG0-ARG1) | 简化映射 |
| 抽象意义表示 | ✅ AMR SVG | ❌ | MTL 模型不含 |
| 指代消解 | ✅ RESTful | ❌ | 闭源功能 |
| 语义文本相似度 | ✅ | ❌ | 不同任务域 |
| 文本风格转换 | ✅ | ❌ | 不同任务域 |
| 关键词提取 | ✅ | ❌ | 可后续扩展 |
| 抽取式摘要 | ✅ | ❌ | 可后续扩展 |
| 生成式摘要 | ✅ | ❌ | 可后续扩展 |
| 文本纠错 | ✅ | ❌ | 不同任务域 |
| 文本分类 | ✅ | ❌ | 可后续扩展 |
| 情感分析 | ✅ | ❌ | 可后续扩展 |
| 古汉语分析 | ❌ | ✅ | **独占能力** |
| 多语种 | ✅ EN/JA/MUL | ❌ | 仅中文 |

## 核心差异分析

### 1. 本地调用 vs 云端 API

| 维度 | HanLP 官方 Demo | narrative-operator-nlp |
|------|----------------|----------------------|
| **执行位置** | 云端 (Binder Jupyter + RESTful API) | 本地 (进程内 HanLP 模型) |
| **网络依赖** | 需要 (API 调用 + WebSocket) | 无需 (离线可用) |
| **模型加载** | 服务端预加载 | 首次 ~3s, 后续 ~0.06s/次 |
| **数据隐私** | 文本发送到 hanlp.com | 文本不离开本地 |
| **并发能力** | 受 API 配额限制 | 本地线程池 (gRPC 4 workers) |
| **自定义模型** | 不支持 | 支持 (HanLP fork 可替换) |

### 2. 输出格式

| 维度 | 官方 Demo | narrative-operator-nlp |
|------|----------|----------------------|
| **原始输出** | HanLP Document (dict) | ✅ 保留在 `structural` |
| **标准化输出** | 无 | ✅ NSP NarrativeDocument |
| **可视化** | `pretty_print()` ASCII 树 | ✅ `/demo` 页面 (4 tab) |
| **SVG 树** | Jupyter 内联渲染 | ✅ 自研内联 SVG 渲染 |
| **JSON Schema** | 无 | ✅ `narrative.schema.json` |
| **多协议** | RESTful only | ✅ MCP + gRPC + FastAPI |

### 3. 交互方式

| 维度 | 官方 Demo | narrative-operator-nlp |
|------|----------|----------------------|
| **Web GUI** | Jekyll 静态页 + Jupyter | FastAPI `/demo` 自包含 HTML |
| **代码执行** | Binder Jupyter Notebook | 本地 Python SDK |
| **API** | RESTful (`hanlp.com/api`) | REST + gRPC + MCP |
| **一键运行** | Binder 在线 | `docker compose up -d` |
| **示例切换** | 侧边栏导航 | 页面内按钮 (3 预设) |

### 4. 架构差异

| 维度 | 官方 Demo | narrative-operator-nlp |
|------|----------|----------------------|
| **设计理念** | 功能展示 + 教学 | 协议优先算子 |
| **模型隔离** | 无 (API 服务端透明) | 独立进程 + Python 3.10 锁定 |
| **NLP 引擎** | HanLP 2.1 RESTful | HanLP 2.1 MTL (本地 fork) |
| **依赖管理** | 无 | uv + setup.py extras |
| **容器化** | 无 | Dockerfile + docker-compose |
| **标准化契约** | 无 | Protobuf + JSON Schema |

## 本地调用的关键优势

### 零网络延迟

```
官方 Demo:  文本 → HTTP → hanlp.com API → HTTP ← 结果  (~200-500ms)
本地方案:   文本 → HanLP MTL 本地推理 → NSP  (~60ms)
```

本地推理比云端 API 快 **3-8 倍**，且不受网络抖动影响。

### 数据不出域

```
官方 Demo:  "碳钢是钢的一种" → 发送到 hanlp.com → 分析 → 返回
本地方案:   "碳钢是钢的一种" → 本地 HanLP → NSP
```

适用于涉密材料、企业内部文档、专利文本等场景。

### 协议标准化

```
官方 Demo:  单一 RESTful JSON 输出 (HanLP 原生格式)
本地方案:   NSP 标准格式 → MCP/gRPC/FastAPI 三通道同时可用
```

NarrativeOS 生态内所有组件通过统一契约通信，无需解析异构 JSON。

### 离线可用

```
官方 Demo:  需要互联网连接
本地方案:  模型下载后完全离线运行
```

适用于内网环境、安全隔离区、学术复现场景。

## 尚未覆盖的能力

| 缺失 | 影响 | 计划 |
|------|------|------|
| AMR 抽象意义表示 | 深层语义分析缺失 | HanLP RESTful API 闭源，需替代方案 |
| 指代消解 | 跨句实体链接缺失 | HanLP 闭源，可考虑 spaCy/coref 替代 |
| 多语种 (EN/JA) | 仅支持中文 | MTL 模型支持多语种但未集成 |
| 关键词提取 | 文档摘要能力缺失 | TextRank 等算法可独立实现 |
| 文本摘要 | 长文本处理缺失 | 可后续扩展单任务模型 |

## 建议

1. **近期**: 保持当前覆盖范围 (7 核心 NLP 任务 + 古汉语)，深化 NSP 映射质量
2. **中期**: 集成多语种 MTL 模型 (UD_ONTONOTES 系列)
3. **远期**: 基于 NSP 关系三元组做下游叙事推理 (因果链、时序线、人物网络)
