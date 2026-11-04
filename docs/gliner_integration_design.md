# GLiNER 第二引擎集成设计（CPU 优先）

> 状态：**设计待评审，未实现**。选型前提是"CPU 友好优先"——本项目的部署形态是
> 单机容器 + gRPC sidecar，没有 GPU 预算，因此模型路线只在**纯 CPU 可跑、
> 可 ONNX 化、许可证无障碍**三个条件同时成立时才考虑。

## 1. 要解决的问题

`KeywordExtractor` 对 MATERIAL / STANDARD / PARAMETER 这类 NSP 自定义类别
**完全依赖人工 YAML 词典**（`config/domain_keywords.yaml`），命中即
`confidence=0.95`，词典外一律漏检。这是当前实体标注的主要召回缺口，
且不是统计方法能补的（`discoverer.py` 只产出候选词，不产出类别标签）。

## 2. 为什么是 GLiNER，而不是别的

| 候选 | CPU 友好度 | 结论 |
|---|---|---|
| **GLiNER** (`urchade/GLiNER`，Apache-2.0) | 小型 encoder，官方支持 ONNX 导出，bi-encoder 变体可缓存标签向量 | **选它** |
| PaddleNLP UIE (Apache-2.0) | 需引入 Paddle 框架，与现有 TensorFlow/PyTorch 栈共存成本高，CPU 推理慢 | 排除 |
| DeepKE (MIT) | 少样本训练配方有价值，但推理链路重 | 仅借鉴训练配方 |
| HanLP MTL 加自定义 NER 任务 | 复用现有进程，**零新增依赖** | 微调阶段的默认路线 |

关键取舍：**HanLP MTL 自定义任务是迁移成本最低的一条路**（不换引擎、只加任务），
GLiNER 的价值在于**零样本冷启动**——新领域还没有标注数据时先用它兜底。

## 3. 架构：可选 sidecar，默认关闭

```
FastAPI /analyze
   └─ HanLP MTL pipeline            （现有主链路，不动）
        └─ KeywordExtractor         （词典精确匹配，不动）
             └─ GlinerEngine?       （feature flag，进程外 gRPC sidecar）
                  └─ EntityDeduplicator → EntityMerger   （复用现有归一管线）
```

- **默认 `gliner_enabled=False`**：不装依赖、不起容器时，行为与现在完全一致。
- 走 **sidecar 而非 in-process**：ONNX runtime 的线程池不能和 HanLP 的
  TensorFlow/PyTorch 线程池抢核；隔离后可以为它单独设 `OMP_NUM_THREADS` 与
  容器 `cpus` 上限。
- 输出**必须**经现有 `EntityDeduplicator` / `EntityMerger` 归一，不允许第二
  引擎直接产出最终实体——否则 ID 体系与去重语义会分裂。

## 4. 标签映射（NSP → GLiNER label prompt）

`core/schema.py` 的 `EntityCategory` 直接作为 prompt 标签，中文标签比英文
标签在中文模型上召回更好，因此每个类别给中英双标签：

| EntityCategory | GLiNER labels |
|---|---|
| MATERIAL | `["材料", "material"]` |
| STANDARD | `["标准", "标准规范", "standard"]` |
| PARAMETER | `["参数", "性能指标", "parameter"]` |
| TITLE / ERA / INSTITUTION | 古汉语类别需自建评测集后再开，**先不接** |

映射表放 `config/gliner_labels.yaml`，与 `domain_keywords.yaml` 同级同风格。

## 5. CPU 预算与准入闸门

闸门脚本已存在：`scripts/bench_gliner_cpu.py`（未安装 `gliner` 时给出可执行的
安装提示并以退出码 2 结束，因此可以在任何机器上先跑起来看缺什么）。

| 指标 | 目标 | 脚本输出行 |
|---|---|---|
| 单请求延迟 | ≤ 现有 MTL 主链路的 +30% | `[latency] ratio=` |
| 常驻内存增量 | ≤ 300 MB | `[memory ] rss delta=` |
| 并发退化 | 2 并发下 P95 ≤ 2× 单请求 | `[concurr] ratio=` |
| 精度收益 | 词典+GLiNER 相对纯 GLiNER 召回 +5pt 以上 | `[quality] dict gain=` |

两条判定规则（都已实现，是脚本的硬行为）：

1. **未评测 = 不通过。** 缺 `--eval` 标注集、或用 `--no-baseline` 跳过主链路对比时，
   对应闸门记为未评测并强制 `NO-GO`，不会因为"没测到问题"而放行。
2. **低于 1ms 计时分辨率的比值不算通过。** 比值在噪声量级时输出
   `INCONCLUSIVE` 并判 `NO-GO`——防止一个耗时≈0 的空实现骗过延迟闸门。

模型侧优先 **ONNX INT8 量化版**；若量化后精度掉点明显，退回 fp32 而不是
换更大的模型。

## 6. 分期

1. **P0 离线评测**（不改生产代码）：`python scripts/bench_gliner_cpu.py
   --corpus <每行一篇文档> --eval <标注 jsonl>`，产出 go/no-go。
   标注文件格式：`{"text": "...", "entities": [{"text": "碳纤维", "category": "MATERIAL"}]}`
2. **P1 影子模式**：sidecar 起来但结果只记日志不进响应，观察一周真实流量分布。
3. **P2 灰度**：`gliner_enabled` 按请求开启，词典命中优先、GLiNER 仅补词典未覆盖
   的 span（避免与 0.95 置信度的词典结果抢排名）。
4. **P3 决策**：若 P2 收益不足，转 HanLP MTL 自定义 NER 任务微调（Phase 2 标注
   数据本就要建，见 `docs/new_word_discovery.md`）。

## 7. 明确不做

- 不引入 Paddle 栈。
- 不让第二引擎绕过 dedup/merge 直接写 `NarrativeDocument.entities`。
- 不在 CPU 上跑 >200M 参数的模型。
- 不用零样本结果训练零样本模型（自训练需先有标注集）。

## 8. 相关实测记录

`discoverer.py` 的 CNG/BCBM25 集成实验（`scripts/bench_discoverer_indicators.py`）
已证明：从关键短语抽取借来的频率指标在**单文档**尺度上不产生净收益。同一
方法论适用于此处——任何"看起来更先进"的模型在进入主链路前，必须先在同一份
真实语料上量出可复现的增量。