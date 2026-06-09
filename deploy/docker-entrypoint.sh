#!/bin/bash
# ───────────────────────────────────────────────────────────────
# Narrative Operator NLP — Docker Entrypoint
# ───────────────────────────────────────────────────────────────
# Modes: fastapi | grpc | both | demo
# ───────────────────────────────────────────────────────────────
set -e

MODE="${1:-fastapi}"
GRPC_SOCKET="${GRPC_SOCKET:-/tmp/narrative-operator-nlp.sock}"

echo "============================================"
echo " Narrative Operator NLP"
echo " Mode:       $MODE"
echo " Python:     $(python --version)"
echo " HanLP Home: $HANLP_HOME"
echo "============================================"

# ── Model Check & Auto-Download ───────────────────────────────
echo ""
echo "[1/2] Checking pre-trained models..."
MODELS_OK=true

# Check MTL (modern Chinese)
if python scripts/setup_models.py --check --model MTL 2>/dev/null; then
    echo "  ✅ MTL (现代汉语) — found"
else
    echo "  ⬇️  MTL (现代汉语) — downloading (~500 MB)..."
    python scripts/setup_models.py --model MTL
fi

# Check LZH (classical Chinese)
if python scripts/setup_models.py --check --model LZH 2>/dev/null; then
    echo "  ✅ LZH (古汉语) — found"
else
    echo "  ⬇️  LZH (古汉语) — downloading (~300 MB)..."
    python scripts/setup_models.py --model LZH
fi

echo "  Models cached at $HANLP_HOME"

# ── Launch Service ────────────────────────────────────────────
echo ""
echo "[2/2] Starting service: $MODE"
echo ""

case "$MODE" in
    fastapi)
        echo "  Demo 页面:    http://localhost:8000/demo"
        echo "  Swagger UI:   http://localhost:8000/docs"
        echo "  API 接口说明: Demo 页面 → 📋 API 接口 标签页"
        exec python adapters/fastapi_app.py
        ;;
    grpc)
        exec python adapters/grpc_server.py --socket "$GRPC_SOCKET"
        ;;
    both)
        echo "  Demo 页面:    http://localhost:8000/demo"
        echo "  Swagger UI:   http://localhost:8000/docs"
        echo "  gRPC Socket:  $GRPC_SOCKET"
        python adapters/grpc_server.py --socket "$GRPC_SOCKET" &
        sleep 2
        python adapters/fastapi_app.py
        ;;
    mcp)
        exec python adapters/mcp_server.py
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: docker run ... narrative-operator-nlp [fastapi|grpc|both]"
        exit 1
        ;;
esac
