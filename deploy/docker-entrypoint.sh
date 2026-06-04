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
if python scripts/setup_models.py --check --model MTL 2>/dev/null; then
    echo "  All required models found."
else
    echo ""
    echo "  Downloading required models (~500 MB, first run only)..."
    python scripts/setup_models.py --model MTL
    echo "  Models cached at $HANLP_HOME"
fi

# ── Launch Service ────────────────────────────────────────────
echo ""
echo "[2/2] Starting service: $MODE"
echo ""

case "$MODE" in
    fastapi)
        exec python adapters/fastapi_app.py
        ;;
    grpc)
        exec python adapters/grpc_server.py --socket "$GRPC_SOCKET"
        ;;
    both)
        # gRPC in background, FastAPI in foreground
        python adapters/grpc_server.py --socket "$GRPC_SOCKET" &
        sleep 2
        python adapters/fastapi_app.py
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: docker run ... narrative-operator-nlp [fastapi|grpc|both]"
        exit 1
        ;;
esac
