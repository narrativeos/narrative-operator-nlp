# ───────────────────────────────────────────────────────────────
# Narrative Operator NLP — Docker Image
# ───────────────────────────────────────────────────────────────
# Mirrors: apt (Aliyun), pip (Tsinghua), PyTorch (Tsinghua)
#
# Build:
#   docker build -t narrative-operator-nlp -f deploy/Dockerfile .
#
# Run (FastAPI + Demo 页面 + API 接口说明):
#   docker run -p 8000:8000 -v hanlp-models:/root/.hanlp narrative-operator-nlp
#
# 访问交互式 Demo:  http://localhost:8000/demo
# 访问 API 文档:     http://localhost:8000/docs
# ───────────────────────────────────────────────────────────────

FROM python:3.10-slim

WORKDIR /app

# ── Apt: Aliyun mirror ──────────────────────────────────────
RUN rm -f /etc/apt/sources.list.d/* \
    && echo "deb http://mirrors.aliyun.com/debian trixie main" > /etc/apt/sources.list \
    && echo "deb http://mirrors.aliyun.com/debian-security trixie-security main" >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Pip: Tsinghua mirror ────────────────────────────────────
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# Layer 1: install heavy deps first for cache
COPY setup.py hanlp/version.py ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir tensorflow transformers tf_keras

# Layer 2: copy app and install narrative extras
COPY . .
RUN pip install --no-cache-dir -e ".[narrative]" \
    && pip install --no-cache-dir -e plugins/hanlp_common

# ── Environment ──────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV HANLP_HOME=/root/.hanlp
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV COPAW_HANLP_REQUIRE_LOCAL_MODELS=1
# HuggingFace mirror (China mainland acceleration)
ENV HF_ENDPOINT=https://hf-mirror.com
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
# TF 2.16+ compat: use legacy Keras for ConvSeg model
ENV TF_USE_LEGACY_KERAS=1

# Model cache volume (~2-3 GB for all models)
VOLUME /root/.hanlp

# Entrypoint
COPY deploy/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

EXPOSE 8000
CMD ["fastapi"]
