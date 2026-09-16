#!/bin/bash
# ───────────────────────────────────────────────────────────────
# Narrative Operator NLP — Docker Entrypoint
# ───────────────────────────────────────────────────────────────
# Modes: fastapi | grpc | both | demo
# ───────────────────────────────────────────────────────────────
set -e

MODE="${1:-fastapi}"
GRPC_SOCKET="${GRPC_SOCKET:-/tmp/narrative-operator-nlp.sock}"
MODEL_SET="${HANLP_MODEL_SET:-MTL}"

case "$MODEL_SET" in
    MTL|LZH|PIPELINE|ALL) ;;
    *)
        echo "❌ Invalid HANLP_MODEL_SET: '$MODEL_SET' (expected MTL | LZH | PIPELINE | ALL)"
        exit 1
        ;;
esac

echo "============================================"
echo " Narrative Operator NLP"
echo " Mode:       $MODE"
echo " Models:     $MODEL_SET"
echo " Python:     $(python --version)"
echo " HanLP Home: $HANLP_HOME"
echo "============================================"

# ── Model Setup & Auto-Download ───────────────────────────────
# hanlp.load() is idempotent: cached models load in seconds,
# missing models are downloaded on first run.
echo ""
echo "[1/2] Setting up pre-trained models ($MODEL_SET)..."

if ! python scripts/setup_models.py --model "$MODEL_SET"; then
    echo ""
    echo "❌ Model setup reported failures."
    if [ "$MODEL_SET" = "MTL" ] || [ "$MODEL_SET" = "ALL" ]; then
        echo "   MTL (modern Chinese) is required — aborting startup."
        exit 1
    fi
    echo "   ⚠️  Continuing anyway — features depending on the missing"
    echo "      model set will degrade until models are available."
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
