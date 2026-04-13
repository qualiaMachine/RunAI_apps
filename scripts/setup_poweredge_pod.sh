#!/usr/bin/env bash
# ===========================================================================
# setup_poweredge_pod.sh — one-time setup for running the Streamlit app
# inside a Run:ai pod on PowerEdge.
#
# Usage (from repo root):
#   bash scripts/setup_poweredge_pod.sh
#
# After setup, start the app with:
#   bash scripts/run_app.sh
# ===========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo " KohakuRAG_UI — PowerEdge Pod Setup"
echo "============================================"
echo ""
echo "Repo root:  $REPO_ROOT"
echo "Python:     $(python --version 2>&1)"
echo ""

# ------------------------------------------------------------------
# Step 1: Install Python dependencies
# ------------------------------------------------------------------
echo "[1/4] Installing Python dependencies..."
if command -v uv &>/dev/null; then
    echo "  Using uv (fast installer)"
    uv pip install --system -r local_requirements.txt
else
    echo "  Using pip"
    python -m pip install --no-cache-dir -r local_requirements.txt
fi
echo "  Done."
echo ""

# ------------------------------------------------------------------
# Step 2: GPU check
# ------------------------------------------------------------------
echo "[2/4] Checking GPU availability..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free \
               --format=csv,noheader
else
    echo "  WARNING: nvidia-smi not found."
    echo "  Make sure your Run:ai job was submitted with --gpu N"
    echo "  (e.g. runai submit ... --gpu 1)"
fi
echo ""

# ------------------------------------------------------------------
# Step 3: Build the vector index (if needed)
# ------------------------------------------------------------------
DB_PATH="$REPO_ROOT/data/embeddings/wattbot_jinav4.db"
echo "[3/4] Checking vector database..."
if [ -f "$DB_PATH" ]; then
    echo "  Found: $DB_PATH ($(du -h "$DB_PATH" | cut -f1))"
    echo "  Skipping index build."
else
    echo "  Not found: $DB_PATH"
    echo "  Building the vector index (this may take a few minutes)..."
    cd "$REPO_ROOT/vendor/KohakuRAG"
    kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
    cd "$REPO_ROOT"
    if [ -f "$DB_PATH" ]; then
        echo "  Built successfully: $(du -h "$DB_PATH" | cut -f1)"
    else
        echo "  ERROR: Index build did not produce $DB_PATH"
        echo "  Check that data/corpus/ has source documents."
        exit 1
    fi
fi
echo ""

# ------------------------------------------------------------------
# Step 4: HuggingFace token check (for gated models)
# ------------------------------------------------------------------
echo "[4/4] Checking HuggingFace token..."
if [ -n "${HF_TOKEN:-}" ]; then
    echo "  HF_TOKEN is set."
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "  Found cached token at ~/.cache/huggingface/token"
else
    echo "  WARNING: No HF_TOKEN set and no cached token found."
    echo "  Gated models (Llama 3, Gemma 2) will fail with 401."
    echo "  To fix: export HF_TOKEN=\"hf_your_token_here\""
fi
echo ""

echo "============================================"
echo " Setup complete! Start the app with:"
echo ""
echo "   bash scripts/run_app.sh"
echo "============================================"
