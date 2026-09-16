# Install

`narrative-operator-nlp` 的本地安装指南。Docker 部署见 [deployment.md](deployment.md)。

## 环境要求

- **Python 3.10**（本项目与底层 HanLP 引擎的统一要求）
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- GPU 可选（CPU 即可运行，GPU 可加速推理）

## 1. 创建虚拟环境

```bash
git clone https://github.com/narrativeos/narrative-operator-nlp
cd narrative-operator-nlp

uv venv --python 3.10
source .venv/bin/activate
```

## 2. 安装

```bash
uv pip install -e ".[narrative]"
```

`narrative` extras 包含三层协议适配器所需的全部依赖（pydantic / fastapi / uvicorn / grpcio / mcp）。

### 可选 extras

| Extra | 说明 |
|-------|------|
| `narrative` | 协议适配器依赖（MCP / gRPC / FastAPI），**本项目主要使用** |
| `amr` | AMR 语义解析模型依赖（penman 等） |
| `fasttext` | fastText 依赖 |
| `tf` | TensorFlow + fastText（部分旧模型需要） |
| `full` | 以上全部 |

## 3. 下载预训练模型

模型不会随包安装，由 `scripts/setup_models.py` 统一管理并缓存到 `~/.hanlp/`（可用 `HANLP_HOME` 环境变量重定向）：

```bash
# 下载全部模型（MTL + LZH + PIPELINE，约 2-3 GB）
python scripts/setup_models.py

# 只检查已下载的模型，不下载
python scripts/setup_models.py --check

# 按需下载特定模型集
python scripts/setup_models.py --model MTL       # 现代汉语（必选，~500MB）
python scripts/setup_models.py --model LZH       # 古汉语（~300MB）
python scripts/setup_models.py --model PIPELINE  # 单任务模型
```

模型清单与说明见 [implementation.md](implementation.md#模型管理)。

## 4. 验证

```python
from core.analyzer import analyze

doc = analyze("碳钢是钢的一种，具有高强度和高韧性。")
print(doc.model_dump_json(indent=2))
```

## 模型下载问题排查

### 下载失败 / 速度慢

1. 重试（文件服务器在高峰时段可能较慢）。
2. 按终端提示手动下载 `zip` 文件到指定路径。
3. 使用镜像源加速：

   ```bash
   # HanLP 模型镜像
   export HANLP_URL=https://ftp.hankcs.com/hanlp/

   # Hugging Face 镜像（中国大陆加速）
   export HF_ENDPOINT=https://hf-mirror.com
   ```

### 离线服务器

在无网络的服务器上部署时，先在本地机器完成模型下载，再拷贝以下两个目录：

1. `~/.hanlp` — HanLP 模型缓存（对应容器内的 `HANLP_HOME`）
2. `~/.cache/huggingface` — Hugging Face Transformers 缓存

拷贝后可设置 `TRANSFORMERS_OFFLINE=1` 强制使用本地缓存、不再检查更新。

### 更多配置

`HANLP_HOME`、`HANLP_URL`、`HANLP_VERBOSE`、GPU 选择等引擎级配置见 [configure.md](configure.md)。