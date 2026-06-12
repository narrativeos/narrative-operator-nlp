# 古汉语 NLP 处理指南

## 概述

本文档介绍 narrative-operator-nlp 中对古汉语（文言文）的支持，包括架构设计、改进内容和配置方法。

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     narrative-operator-nlp                      │
│                                                                 │
│  ┌──────────────────┐     ┌──────────────────────────────────┐ │
│  │  HanLP lzh       │     │  Post-processing (后处理层)       │ │
│  │  Model (外部)    │────▶│                                  │ │
│  │                  │     │  - EntityMappingRules             │ │
│  │  - Tokenization  │     │  - RelationExtractionRules        │ │
│  │  - POS tagging   │     │  - ModifierExtractor              │ │
│  │  - NER           │     │  - NegationDetector               │ │
│  │  - Dependency    │     │  - CorefResolver                  │ │
│  │  - SRL           │     │  - DictionaryLoader               │ │
│  └──────────────────┘     └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

```
原始文本
    │
    ▼
HanLP lzh model (外部)
    │
    ├── tok/fine        ──────┐
    ├── pos/ctb            ───┤
    ├── pos/xpos           ───┤
    ├── ner/*               ──┤
    ├── dep                  ─┤
    └── srl                  ─┤
                              │
                              ▼
                    EntityMappingRules
                    (整合 NER + 种子词典 + xpos)
                              │
                              ▼
                    RelationExtractionRules
                    (SRL → 关系 + 动词映射)
                              │
                              ▼
                    ModifierExtractor
                    (证据文本 → 修饰符)
                              │
                              ▼
                    NegationDetector
                    (证据文本 → 否定调整)
                              │
                              ▼
                    CorefResolver
                    (代词 → 实体链接)
                              │
                              ▼
                    DictionaryLoader (新增)
                    (种子词典 → 实体增强)
                              │
                              ▼
                    mapper.py
                    (句型检测 + 韵律分析)
                              │
                              ▼
                    NarrativeDocument (最终输出)
```

## 改进内容

### 1. 实体类别扩展

新增了 4 个古汉语特定实体类别：

| 类别 | 说明 | 示例 |
|------|------|------|
| TITLE | 官职名、爵位 | 丞相、太守、将军 |
| ERA | 朝代名、时代 | 秦汉、隋唐、宋元 |
| INSTITUTION | 典章制度 | 科举、郡县制 |
| ASTRONOMY | 天文历法 | 星宿、干支 |

### 2. 种子词典管理

采用三层架构管理词典：

```
config/classical/
├── seed/          # 开源种子词典 (只读，定期同步)
│   ├── titles.yaml
│   ├── eras.yaml
│   ├── locations.yaml
│   ├── persons.yaml
│   ├── classics.yaml
│   └── institutions.yaml
├── custom/        # 自维护词典 (可手动编辑)
└── user/          # 用户自定义 (运行时注入)
```

**DictionaryLoader 特性：**
- 版本管理 (从注释提取版本号)
- MD5 校验和验证完整性
- 元数据追踪 (来源、更新时间)
- 错误处理和日志记录

### 3. 修饰符与否定增强

**古汉语修饰符词典：**
- 疑问词: 16个 (何, 胡, 奚, 曷, 安, 焉...)
- 时态副词: 19个 (尝, 曾, 已, 既, 方...)
- 程度副词: 10个 (甚, 极, 至, 颇...)
- 范围副词: 8个 (悉, 皆, 俱, 咸...)

**古汉语否定词：**
- 扩展 17 个否定词 (弗, 毋, 罔, 微, 未尝...)

### 4. 共指消解增强

**古汉语代词系统：**
- 18 个古汉语代词
- 每个代词有 case 属性 (obj/gen/subj/demonstrative)
- 之 → obj (宾语位置)
- 其 → gen (所有格/定语)

### 5. 关系提取增强

**古汉语动词映射：**
- 25 个动词到谓词的映射
- 判断/系词: 为→IS_A, 乃→IS_A
- 移动: 至→MOVED_TO, 往→MOVED_TO
- 使令: 使→CAUSES, 令→CAUSES

**古汉语连词分割：**
- 支持: 暨, 幷, 並, 共, 俱, 皆...

### 6. 句式模式检测

**检测的句式类型：**
- 判断句: "陈胜者，阳城人也"
- 被动句: "见欺于王", "为V所N"
- 反问句: "何陋之有", "不亦乐乎"
- 比较句: "孰与君少", "何...如"
- 否定判断句: "非君子也"

### 7. 韵律特征分析

**四字格检测：**
- 检测对偶/排比结构
- 识别四字节奏模式

## 配置方法

### 使用种子词典

词典在 `EntityMappingRules` 初始化时自动加载：

```python
from core.entity_mapper import EntityMappingRules

# 自动加载 config/classical/seed/ 下的词典
rules = EntityMappingRules()
```

### 添加自定义词典

```python
# 运行时注入用户词典
rules = EntityMappingRules(
    entity_categories={
        "TITLE": ["大将军", "骠骑将军"],
        "ERA": ["五代十国"],
    }
)
```

### 添加自定义词典文件

在 `config/classical/custom/` 目录下添加 YAML 文件：

```yaml
# config/classical/custom/titles.yaml
titles:
  - 大将军
  - 骠骑将军
  - 车骑将军
```

## 测试

运行测试套件：

```bash
cd narrative-operator-nlp
python3 tests/test_classical_chinese.py
```

## 数据源

| 文件 | 来源 | 更新方式 |
|------|------|---------|
| seed/titles.yaml | CLNER + 手动整理 | 定期同步 |
| seed/eras.yaml | 手动整理 | 手动更新 |
| seed/locations.yaml | CLNER + DuEE | 定期同步 |
| seed/persons.yaml | CLNER | 定期同步 |
| seed/classics.yaml | 手动整理 | 手动更新 |
| seed/institutions.yaml | 手动整理 | 手动更新 |

## 相关文件

| 文件 | 说明 |
|------|------|
| `core/schema.py` | 实体类别定义 |
| `core/entity_mapper.py` | 实体提取整合 |
| `core/dictionary_loader.py` | 词典加载器 |
| `core/modifier_extractor.py` | 修饰符提取 |
| `core/negation_detector.py` | 否定检测 |
| `core/coref_resolver.py` | 共指消解 |
| `core/relation_mapper.py` | 关系提取 |
| `core/mapper.py` | 句型检测、韵律分析 |
| `tests/test_classical_chinese.py` | 测试套件 |

## 版本历史

| 版本 | 日期 | 改动 |
|------|------|------|
| 1.0.0 | 2026-06-12 | 初始版本 |