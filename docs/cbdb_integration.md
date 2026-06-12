# CBDB (China Biographical Database) 整合完成

## 概述

CBDB (China Biographical Database) 是一个包含大量中国历史人物传记数据的开源数据库，由哈佛大学的伊佩霞 (Patricia Ebrey) 等学者创建。

## 当前状态

**已完成** - CBDB 数据库已集成到 narrative-operator-nlp 中，支持自动按需查询。

## 架构设计

### 目录结构

```
config/classical/
├── seed/              # YAML 种子词典 (小规模 < 10K)
│   ├── persons.yaml
│   ├── locations.yaml
│   ├── titles.yaml
│   └── ...
├── custom/            # 自定义词典 (手动维护)
├── user/              # 用户词典 (运行时注入)
└── databases/         # SQLite 数据库 (大规模)
    ├── cbdb.sqlite    # CBDB 原始数据库 (576MB)
    └── metadata.json  # 版本信息
```

### 查询机制

采用**两级查询**策略：

1. **一级**：YAML 种子词典（全量加载到内存）
2. **二级**：SQLite 数据库（按需查询 + 缓存）

### 查询优先级

1. ERA (朝代) - 最高优先级
2. TITLE (官职)
3. LOCATION (地名)
4. PERSON (人名) - 最低优先级（表最大）

## 使用方法

```python
from core.dictionary_loader import DictionaryLoader
from pathlib import Path

config_dir = str(Path.cwd() / 'config')
loader = DictionaryLoader(config_dir)
loader.load_all()

# 自动查询（先查 YAML，再查 SQLite）
entry = loader.lookup("诸葛亮")  # PERSON (seed)
entry = loader.lookup("唐朝")    # LOCATION (CBDB)
```

## 数据规模

| 类别 | 数量 | 来源 |
|------|------|------|
| 人物 | 658,941 | CBDB |
| 地名 | 14,472 | CBDB |
| 官职 | 29,367 | CBDB |
| 朝代 | 85 | CBDB |

## 同步更新

### 检查更新

```bash
python scripts/sync_cbdb.py
```

### 手动更新

```bash
# 1. 下载最新数据库
wget -O /tmp/cbdb_latest.zip "https://huggingface.co/datasets/cbdb/cbdb-sqlite/resolve/main/latest.zip"

# 2. 解压
unzip /tmp/cbdb_latest.zip -d /tmp/

# 3. 替换旧数据库
cp /tmp/cbdb_*.sqlite3 config/classical/databases/cbdb.sqlite

# 4. 更新元数据
# 编辑 config/classical/databases/metadata.json
```

## 注意事项

1. **许可证**: CBDB 使用 CC BY-SA 4.0 许可证
2. **数据质量**: CBDB 数据可能存在不完整或不准确的情况
3. **更新频率**: CBDB 定期更新，建议每月检查一次
4. **性能**: SQLite 查询使用索引，单次查询 < 1ms

## 参考链接

- CBDB 官方网站: https://cbdb.fas.harvard.edu/
- CBDB GitHub: https://github.com/cbdb-project
- HuggingFace 数据集: https://huggingface.co/datasets/cbdb/cbdb-sqlite
