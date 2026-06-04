# ───────────────────────────────────────────────────────────────
# Narrative Operator NLP — Docker Image
# ───────────────────────────────────────────────────────────────
# Mirrors: apt (Aliyun), pip (Tsinghua), PyTorch (Tsinghua)
#
# Build:
#   docker build -t narrative-operator-nlp -f deploy/Dockerfile .
# Run:
#   docker run -p 8000:8000 -v hanlp-models:/root/.hanlp narrative-operator-nlp
# ───────────────────────────────────────────────────────────────

FROM python:3.10-slim

WORKDIR /app

# ── Apt: Aliyun mirror ──────────────────────────────────────
RUN echo "deb http://mirrors.aliyun.com/debian bookworm main" > /etc/apt/sources.list \
    && echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main" >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Pip: Tsinghua mirror ────────────────────────────────────
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# Layer 1: install heavy deps first for cache
COPY setup.py hanlp/version.py ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Layer 2: copy app and install narrative extras
COPY . .
RUN pip install --no-cache-dir -e ".[narrative]"

# ── Environment ──────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV HANLP_HOME=/root/.hanlp
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV COPAW_HANLP_REQUIRE_LOCAL_MODELS=1

# Model cache volume (~2-3 GB for all models)
VOLUME /root/.hanlp

# Entrypoint
COPY deploy/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

EXPOSE 8000
CMD ["fastapi"]
