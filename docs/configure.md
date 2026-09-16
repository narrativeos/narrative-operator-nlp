# Configuration

`narrative-operator-nlp` 的配置分为三层：

1. **抽取规则配置**（`config/` 目录）— 控制实体抽取行为，无需改代码
2. **部署环境变量**（`.env`）— 控制 Docker 部署行为，详见 [deployment.md](deployment.md)
3. **引擎环境变量** — 透传给底层 HanLP 引擎，控制模型缓存与下载

## 1. 抽取规则配置（config/）

| 文件 | 用途 |
|------|------|
| `config/entity_mapping.yaml` | NER 模型标签 → NSP 标准实体类别的映射规则 |
| `config/merge_rules.yaml` | 相邻实体跨类别合并规则（`from` 类别对 → 结果类别，按 priority 排序） |
| `config/domain_keywords.yaml` | 内置通用领域关键词（MATERIAL / STANDARD / PARAMETER）；**领域特定关键词应通过 API 运行时注入，而非写入此文件** |
| `config/custom_dict.txt` | 自定义分词词典，每行一个词，`#` 开头为注释；`dict_combine` 会将模型输出中连续匹配的 token 合并为一个词 |
| `config/classical/` | 古汉语资源：CBDB 数据库（`databases/`）与种子词典（`seed/`：classics / eras / events / institutions / locations） |

修改 YAML 规则后重启服务即可生效，无需重新构建镜像。

## 2. 部署环境变量（.env）

Docker 部署的 `HTTP_PORT`、`GRPC_PORT`、`HANLP_MODEL_SET` 等变量见 [deployment.md](deployment.md) 与 [`.env.example`](../.env.example)。

## 3. 引擎环境变量（HanLP）

以下环境变量透传给底层 HanLP 引擎（`hanlp/` fork），本地开发与 Docker 部署均适用：

### 模型缓存目录 HANLP_HOME

所有模型资源缓存到 `HANLP_HOME` 目录，默认 `~/.hanlp`（Windows 为 `%appdata%\hanlp`）。可重定向到任意路径：

```bash
export HANLP_HOME=/data/hanlp
```

### 模型下载镜像 HANLP_URL

默认从 HanLP 官方 CDN 下载模型。部分地区速度较慢时，可设置镜像源（下次启动生效）：

```bash
export HANLP_URL=https://ftp.hankcs.com/hanlp/
```

Hugging Face 模型（Transformers）使用独立的镜像变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 中国大陆加速
```

### 下载进度输出 HANLP_VERBOSE

默认加载模型时打印进度信息。静默模式：

```bash
export HANLP_VERBOSE=0
```

### GPU 选择

默认自动选择占用最少的 GPU。可通过 PyTorch/TensorFlow 标准的 `CUDA_VISIBLE_DEVICES` 限制可选设备：

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

如需对单个组件做细粒度控制，使用 `hanlp.load(..., devices=...)`（见 [HanLP 文档](https://hanlp.hankcs.com/docs/api/hanlp/index.html#hanlp.load)）。

GPU 环境搭建参考：[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)、[PyTorch 安装指南](https://pytorch.org/get-started/locally/)。未安装 GPU 版 PyTorch 时，默认安装 CPU 版，功能不受影响。
