# 词典架构设计

## 概述

本文档描述 narrative-operator-nlp 中词典管理的架构设计，包括数据源、存储格式和更新机制。

## 架构原则

1. **分层存储**：根据数据规模选择合适的存储格式
2. **订阅更新**：对于频繁更新的外部数据源，采用订阅模式
3. **来源追踪**：所有词典条目必须标记来源
4. **版本管理**：支持词典版本控制和回滚

## 存储层设计

### 层1: YAML 种子词典 (小规模)

**适用场景：**
- 数据量 < 10,000 条目
- 更新频率低
- 需要人工审核

**存储格式：**
```
config/classical/seed/
├── persons.yaml          # 人名 (< 5,000)
├── locations.yaml        # 地名 (< 2,000)
├── titles.yaml           # 官职 (< 2,000)
├── eras.yaml             # 朝代 (< 100)
├── classics.yaml         # 典籍 (< 200)
├── institutions.yaml     # 典章制度 (< 200)
├── organizations.yaml    # 组织 (< 500)
└── events.yaml           # 事件 (< 20)
```

**来源：**
- Classical-Chinese-NER-RE-Dataset (beeevita)
- 手动整理

### 层2: SQLite 数据库 (大规模)

**适用场景：**
- 数据量 > 10,000 条目
- 更新频率高
- 需要高效查询

**存储格式：**
```
config/classical/databases/
├── cbdb_merged.sqlite    # CBDB 合并数据库
└── metadata.json         # 数据库元数据
```

**来源：**
- CBDB (China Biographical Database)
- 其他大型开源数据集

## 数据源管理

### CBDB (China Biographical Database)

| 属性 | 值 |
|------|-----|
| 来源 | https://huggingface.co/datasets/cbdb/cbdb-sqlite |
| 更新频率 | 定期更新 |
| 数据规模 | ~500K 人物, ~15K 地点, ~30K 官职 |
| 存储格式 | SQLite |
| 订阅方式 | 定期检查 latest.json |

**订阅脚本：**
```python
# scripts/sync_cbdb.py
import json
import requests
import sqlite3

def check_for_update():
    """Check if CBDB has a new version."""
    url = "https://github.com/cbdb-project/cbdb_sqlite/raw/master/latest.json"
    response = requests.get(url)
    latest = response.json()
    
    # Compare with local metadata
    local_metadata = load_metadata()
    if latest["sha256"] != local_metadata.get("sha256"):
        download_and_merge(latest)
```

## 合并策略

### 人物合并

```
                    ┌─────────────────┐
                    │  CBDB Database  │
                    │  (~485K persons)│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Merge Engine   │
                    │  - Dedup        │
                    │  - Normalize    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼               ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │ Classical   │ │ NER Dataset│ │ Manual     │
       │ Literature  │ │ (~4K)      │ │ Additions  │
       │ Names       │ │            │ │            │
       └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
              │              │               │
              ▼              ▼               ▼
                    ┌─────────────────┐
                    │  Merged SQLite  │
                    │  Database       │
                    └─────────────────┘
```

### 去重策略

1. **精确匹配**：完全相同的字符串
2. **模糊匹配**：考虑异体字、简繁体
3. **音似匹配**：考虑拼音相同但字形不同

## 访问接口

### DictionaryLoader

```python
class DictionaryLoader:
    """Unified dictionary access interface."""
    
    def __init__(self, config_dir):
        self._yaml_dir = Path(config_dir) / "classical" / "seed"
        self._db_dir = Path(config_dir) / "classical" / "databases"
        self._cache = {}
    
    def get_persons(self) -> Set[str]:
        """Get all person names from all sources."""
        # From YAML
        yaml_persons = self._load_yaml("persons.yaml")
        # From SQLite
        db_persons = self._load_sqlite("persons")
        return yaml_persons.union(db_persons)
    
    def lookup(self, keyword: str) -> Optional[DictEntry]:
        """Look up a keyword in all dictionaries."""
        # Check YAML first (faster for small dicts)
        entry = self._lookup_yaml(keyword)
        if entry:
            return entry
        # Then check SQLite
        return self._lookup_sqlite(keyword)
```

## 更新流程

```
┌─────────────────────────────────────────────────────────────┐
│                     Update Pipeline                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Check for Updates  │
                   │  (latest.json)      │
                   └──────────┬──────────┘
                              │
                       ┌──────┴──────┐
                       │ New version? │
                       └──────┬──────┘
                          Yes │
                              ▼
                   ┌─────────────────────┐
                   │  Download Database  │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Verify Checksum    │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Merge with Local   │
                   │  - Dedup            │
                   │  - Add metadata     │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Update SQLite      │
                   │  + backup old       │
                   └─────────────────────┘
```

## 文件格式比较

| 特性 | YAML | SQLite |
|------|------|--------|
| 可读性 | 高 | 低 |
| 查询效率 | 低 | 高 |
| 适合规模 | < 10K | > 10K |
| 更新方式 | 手动 | 自动 |
| 版本控制 | 友好 | 需额外工具 |
| 压缩率 | 低 | 高 |

## 推荐方案

**对于 CBDB 这类大规模数据集：**

1. **保持原始 SQLite 格式**，不转换为 YAML
2. **创建合并视图**，将多个来源的数据统一查询
3. **定期同步**，检查 HuggingFace 更新
4. **本地缓存**，减少网络请求

**实施步骤：**

1. 创建 `config/classical/databases/` 目录
2. 下载 CBDB SQLite 到该目录
3. 创建合并脚本 `scripts/sync_cbdb.py`
4. 更新 `DictionaryLoader` 支持 SQLite
5. 添加定时任务定期同步