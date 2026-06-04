# Narrative Schema Protocol (NSP)

本文件定义 NarrativeOS 的叙事协议标准（Narrative Schema Protocol, NSP），统一所有 NLP 算子的输出格式。

## Core Principle

不要将底层 NLP 引擎（HanLP、spaCy 等）的原始输出直接暴露给上层。每个算子必须通过 **Schema Mapper** 将原始分析结果降维为标准"叙事原子（Narrative Atoms）"。

### NSP Three Rules

| Rule | Requirement |
|------|-------------|
| **Atomicity** | Every operator output MUST contain `entities` and `relations` lists, with nesting depth ≤ 3 |
| **Traceability** | Every `relation` MUST point to corresponding `evidence` (original text span) |
| **Compatibility** | If `relations` cannot be provided, the field MAY be empty, but the structure MUST remain consistent |

---

## Unified Wrapper

All operator outputs MUST follow this top-level structure:

```json
{
  "meta": {
    "source": "hanlp_v2",
    "version": "1.0",
    "timestamp": "2026-06-04T12:00:00Z",
    "text_length": 1024
  },
  "content": {
    "tokens": [],
    "entities": [],
    "relations": [],
    "structural": {}
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `meta.source` | string | ✅ | Original engine identifier (e.g., `hanlp_v2`, `spacy_v3`) |
| `meta.version` | string | ✅ | Schema version |
| `meta.timestamp` | string | ✅ | ISO 8601 analysis timestamp |
| `meta.text_length` | int | ✅ | Original text character count |
| `content.tokens` | Token[] | ✅ | Normalized token list |
| `content.entities` | Entity[] | ✅ | Unified entity list (core) |
| `content.relations` | Relation[] | ✅ | Dependency/SRL-derived logical chains (core) |
| `content.structural` | object | ❌ | Raw analysis output (debug only) |

---

## Token Schema

```json
{
  "id": 0,
  "text": "碳钢",
  "pos": "NN",
  "span": [0, 2]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | ✅ | Sequential index starting from 0 |
| `text` | string | ✅ | Token text |
| `pos` | string | ✅ | Part-of-speech (Universal POS standard) |
| `span` | [int, int] | ✅ | Character offset [start, end) in original text |

---

## Entity Schema

Unified format for heterogeneous NER outputs:

```json
{
  "id": "ent_001",
  "text": "北京立方庭",
  "category": "FACILITY",
  "span": [2, 4],
  "normalized": "北京立方庭",
  "source": "ner/ontonotes",
  "confidence": 0.97
}
```

### Entity Categories

| Category | Description | Source Labels |
|----------|-------------|---------------|
| `PERSON` | Person name | `ner/pku: NR`, `ner/msra: PERSON` |
| `ORGANIZATION` | Organization | `ner/pku: NT`, `ner/msra: ORG` |
| `LOCATION` | Location | `ner/pku: NS`, `ner/msra: LOC` |
| `FACILITY` | Facility | `ner/ontonotes: FAC` |
| `PRODUCT` | Product | `ner/pku: NZ` |
| `DATE` | Date | `ner/msra: DATE` |
| `NUMBER` | Numeric value | `pos: CD` |
| `MATERIAL` | Material | Custom mapping |
| `STANDARD` | Standard/specification | Custom mapping |
| `PARAMETER` | Parameter | Custom mapping |
| `UNKNOWN` | Unclassified | Fallback |

---

## Relationship Schema

Extract simple triples from dependency parsing (`dep`) and semantic role labeling (`srl`):

```json
{
  "id": "rel_001",
  "subject": "碳钢",
  "subject_ent_id": "ent_002",
  "predicate": "IS_A",
  "object": "钢",
  "object_ent_id": "ent_003",
  "evidence": "碳钢是钢的一种",
  "evidence_span": [10, 17],
  "confidence": 0.98,
  "source": "dep/nsubj"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Unique relation ID |
| `subject` | string | ✅ | Subject text |
| `subject_ent_id` | string | ❌ | Linked entity ID (if available) |
| `predicate` | string | ✅ | Relation type (from predefined set) |
| `object` | string | ✅ | Object text |
| `object_ent_id` | string | ❌ | Linked entity ID (if available) |
| `evidence` | string | ✅ | Original text fragment |
| `evidence_span` | [int, int] | ✅ | Character offset of evidence |
| `confidence` | float | ✅ | Confidence score [0, 1] |
| `source` | string | ✅ | Extraction source (`dep/nsubj`, `srl/A0-A1`, etc.) |

### Predefined Relation Types

| Relation | Description | Trigger Pattern |
|----------|-------------|-----------------|
| `IS_A` | Classification (X is a type of Y) | `nsubj + cop + attr` |
| `PART_OF` | Composition (X is part of Y) | `nmod:poss`, `dep` |
| `PROPERTY_OF` | Property relation (X's Y is Z) | `nsubj + nmod` |
| `HAS_PROPERTY` | Attribute possession (X has property Y) | `amod` |
| `LOCATED_AT` | Spatial (X is located at Y) | `nmod:loc` |
| `TEMPORAL_AT` | Temporal (X occurred at Y) | `nmod:tmod` |
| `CAUSES` | Causation (X causes Y) | `mark + advcl` |
| `DEPENDS_ON` | Dependency (X depends on Y) | `aux:pass` |
| `EQUIVALENT_TO` | Equivalence (X is equivalent to Y) | `appos` |
| `REFERENCE_OF` | Reference (X references standard Y) | Custom |
| `CONSTRAINT_OF` | Constraint (X constrains Y) | Custom |

---

## Schema Mapper Flow

```
HanLP Raw Output
      │
      ▼
┌─────────────────┐
│  Token Mapper    │  tok/fine → NSP Token[]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Entity Mapper   │  ner/pku + ner/msra + ner/ontonotes → NSP Entity[]
│  (merge + dedup) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Relation Mapper  │  dep + srl → NSP Relation[]
│ (rule extraction)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ NarrativeDocument│  Unified NSP output
└─────────────────┘
```

## See Also

- [Operator NLP Architecture](https://github.com/narrativeos/narrative-docs/blob/main/architecture/operator-nlp/README.md)
- [Narrative Schema (docs)](https://github.com/narrativeos/narrative-docs/blob/main/architecture/operator-nlp/narrative-schema.md)
