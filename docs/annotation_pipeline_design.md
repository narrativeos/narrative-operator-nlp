# 语料标注方案（Phase 2 数据契约）

> 状态：**契约已实现，执行体在项目外**。回答四个问题：标注放哪、要不要内置
> 智能体、自动与人工怎么分、是否按书为单元。

## 1. 决策依据：项目现状

| 事实 | 出处 |
|---|---|
| 全仓**零 LLM 依赖**（无 openai/anthropic/litellm/langchain） | 全量 grep |
| 项目**已是 MCP server**，对外暴露 3 个工具 | `adapters/mcp_server.py` |
| 已有 span 级 gold 格式且被 CI 校验 | `tests/datasets/*.json` |
| 已有大语料外置先例（SQLite + metadata + 同步脚本） | CBDB、`scripts/sync_cbdb.py` |
| 已有一致性信号：`evidence` / `ner_disputed` / `filter`(F1/F2 降级) | `core/schema.py`、`core/entity_quality.py` |
| schema **没有文档级身份**（`NarrativeMeta.source` 是引擎名） | `core/schema.py` |

## 2. 契约内置，执行体外置

**内置**（属于本项目的数据契约）：`core/annotation.py` 的模型与校验、
`scripts/label_corpus.py` 的切块/预标/分流/导出。

**外置**（标注执行体）：MCP client agent（Cline / Claude Code 等）通过
`adapters/mcp_server.py` 调用本服务完成预标与复核建议。理由不是省事，而是：

- `analyze()` 必须保持可复现——这是纯确定性算子不可替代的价值；一旦让它内部
  调 LLM，CI 与"同输入同输出"同时失效。
- 标注是**离线批处理**，不在请求路径上，内置等于把离线职责搬进在线服务。
- 项目已经是 MCP server，天然站在"被 agent 调用"的位置，内置 agent 反而多一层。

**自动预标器用确定性引擎，不用 agent**：词典（`config/domain_keywords.yaml`）
+ 可选 GLiNER（见 `gliner_integration_design.md`）+ 可选 HanLP MTL。

## 3. 三级单元：书是批次，section 才是标注单元

```
book      批次单元：license、train/dev/test 划分、内容 hash
  └─ section   标注与人工审核单元：≤2000 字，段落边界切，兜底硬切
       └─ span  标签本身：[start, end) + EntityCategory
```

按书为单元是对的，但**不能直接拿一本书当标注单元**：MTL/GLiNER 有长度上限，
人工审核也需要可定位的粒度。书级负责三件事——许可是书级的、切分必须按书
（同书跨集会造成泄漏）、领域一致使词典与标签集可按书配置。

按书为批次还有一层收益：**频率类指标需要语料尺度**。`new_word_discovery.md`
的实测已证明 CNG/BM25 在单文档上无净收益；书级语料是让它们从"无效诊断"
变成"可用特征"的最小尺度。

存储用**每书一个 JSONL**，而不是照抄 CBDB 的 SQLite：标注数据必须在 diff 里
可读可审，576MB 二进制做不到这一点。目录可用 `NARRATIVE_ANNOTATION_DIR` 覆盖，
测试与试跑不会写进仓库。

## 4. 自动执行 + 人工审核：按分歧度分流，不全量人审

分流直接复用管线已产出的信号，不新造置信度：

| 层级 | 条件 | 处理 |
|---|---|---|
| `agree` | **≥2 个引擎**给出完全相同的 span+category | 自动接受，抽 5% 复检 |
| `partial` | 仅部分引擎命中 | 人工队列（主要工作量） |
| `conflict` | 同 span 类别互相矛盾，或被 F1/F2 降级 | 人工优先，且是未来训练集里信息量最高的样本 |

**AGREE 必须有引擎数量下限**，这是实现中的一个关键陷阱：若只看"跑过的引擎是否
一致"，那么只开词典一个引擎时全部 span 都算"一致"，自动接受率会显示 100%——
实测正是如此（`agree=96 / 100%`）。修正后单引擎语料诚实报告
`agree=0 partial=96`，并提示"单引擎无法构成一致性，请加 `--with-gliner` 或
`--with-mtl`"。自动接受率必须和 `engines_seen` 一起读，否则无意义。

## 5. 用法

```bash
python scripts/label_corpus.py split --book 书.txt --book-id add-mfg \
       --title 增材制造工艺 --license CC-BY-4.0 --split dev
python scripts/label_corpus.py prelabel --book-id add-mfg --with-gliner --with-mtl
python scripts/label_corpus.py validate --book-id add-mfg     # 偏移与文本是否自洽
python scripts/label_corpus.py stats   --book-id add-mfg      # 人工工作量估计
python scripts/label_corpus.py export  --book-id add-mfg --out /tmp/eval.jsonl
python scripts/bench_gliner_cpu.py --eval /tmp/eval.jsonl     # 回到 go/no-go 闸门
```

`export` 只导出 `labeled` 的 section：把没有 gold 的 pending 段喂给闸门，会让
召回率对着空答案集计算而虚高，因此该命令此时直接拒绝执行。

## 6. 明确不做

- 不在本仓内置 agent 或 LLM 客户端。
- 不把标注数据存成二进制库。
- 不用单引擎结果冒充"引擎一致"。

## 7. 边界在哪：契约内置，执行体外置

已实现的分界线：

- **仓内**：数据契约、切分、确定性预标、分流、校验、导出，以及一个**只读**的
  MCP `suggest_labels` tool。
- **仓外**：标注执行体。它是 MCP client，调用 `suggest_labels` 拿到候选与
  分流档位，自行决定采纳哪些，再落盘。

`analyze_text` 是同步、确定性的在线服务面，把 LLM 调用塞进 `core/` 会同时破坏
可复现性并让离线标注占用线上算力。因此**写盘动作不在 MCP 面**：gold 标签只能由
`scripts/label_corpus.py` 显式提交。一次查询不可能顺手改坏已标注语料——这条约束
由 `test_suggest_labels_does_not_touch_the_corpus` 守着，而不是只写在文档里。

`suggest_labels` 返回 `engines_seen` 与一句 `note`。单引擎时它明确说"没有任何
span 能算 agree"，而不是让 agent 把一堆 `partial` 误读成"引擎没有分歧"。

## 8. 预标的三个引擎

| 引擎 | 来源 | 成本 |
|---|---|---|
| `dictionary` | `config/domain_keywords.yaml` | 近零 |
| `gliner` | GLiNER sidecar，见 `docs/gliner_integration_design.md` | 需过 CPU 闸门 |
| `mtl` | 现有 HanLP MTL 的 NER 输出，经 `ner_label_mapper` 映射 | 复用已加载模型 |

三者都只产出**候选**。MTL 的 NER 标签经映射器转换，未命中的落 `UNKNOWN`；
`UNKNOWN` 表示"这个引擎分不出类别"，不算与别的引擎冲突——只有两个具体标签互斥
才标 `disputed`。把分不出类别当成反对，会制造一堆假的人工复核项。

## 9. 数据格式

```
data/annotations/manifest.json     # 书级元数据
data/annotations/<book_id>.jsonl   # 一行一个 section
```

每行 = `{section_id, text, start, end, labels[], prelabels[], status}`。选 JSONL
而非 sqlite：标注数据的价值在于**可审阅**，`git diff` 能直接看懂谁改了哪个 span，
sqlite 做不到。目录可用 `NARRATIVE_ANNOTATION_DIR` 覆盖，测试因此不写进仓库。

## 10. 与闸门的关系

标注管线把 `bench_gliner_cpu.py` 等的那块缺口补上了：导出 gold JSONL → 计算
"词典召回 / 词典+GLiNER 召回 / GLiNER-only 召回" → 决定 GLiNER 的 go/no-go。
闸门对**未评测**与**低于 1ms 计时分辨率**的比值都判 NO-GO，否则空实现能骗过延迟
闸门。GLiNER 若不过关，退路是在现有 HanLP MTL 里加自定义 NER 任务。

## 11. 待确认

1. 标注数据的授权与可分发性——决定语料能否进仓库。
2. 类别体系是否沿用 `EntityCategory` 的 MATERIAL/STANDARD/PARAMETER，还是为
   预训练另立更细的 schema。
3. 目标 CPU 机型与内存上限——`bench_gliner_cpu.py` 需要在那台机器上跑。
4. ~~MCP 面要不要新增标注 tool~~ **已实现**：只读的 `suggest_labels`；写盘仍留在 CLI。

- 不按句子切分后再按书统计——切分与统计必须同一粒度。
