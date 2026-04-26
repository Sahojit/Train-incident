#!/usr/bin/env bash
set -e

# Copy BDD100K checkpoint to the path the dashboard looks for, if not already there
MODELS_DIR="models"
BDD_CKPT="traffic-bdd100k/models/best_model.pth"
TARGET="$MODELS_DIR/best_vision_model.pt"

mkdir -p "$MODELS_DIR"

if [ -f "$BDD_CKPT" ] && [ ! -f "$TARGET" ]; then
    echo "[start] Copying BDD100K checkpoint → $TARGET"
    cp "$BDD_CKPT" "$TARGET"
fi

exec streamlit run dashboard/app.py \
    --server.port "${PORT:-8501}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
