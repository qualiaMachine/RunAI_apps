#!/usr/bin/env bash
# Bootstrap script for the Streamlit app on RunAI.
# Keeps the workspace command short to avoid UI field truncation.
#
# Usage (RunAI Workspace):
#   Command:   bash
#   Arguments: -c "bash /home/jovyan/work/KohakuRAG_UI/scripts/start_app.sh"
#
# The script uses the repo already cloned on the personal PVC from Step 0.
# No GitHub download needed — the code is on the shared filesystem.
set -euo pipefail

# Where Step 0 cloned the repo (personal workspace PVC)
REPO_DIR="${APP_REPO_DIR:-/home/jovyan/work/KohakuRAG_UI}"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "[start_app] ERROR: Repo not found at ${REPO_DIR}"
    echo "[start_app] Run Step 0 first to clone the repo."
    exit 1
fi

echo "[start_app] Using repo at ${REPO_DIR}"
cd "$REPO_DIR"

# Ensure uv is available
command -v uv >/dev/null 2>&1 || {
    echo "[start_app] Installing uv..."
    pip install uv 2>&1 | tail -1
}

# Data setup: the PPVC volume may be read-only. KVaultNodeStore writes
# metadata on open, so we must COPY the vector DB to a writable location.
# Corpus and PDFs can be symlinked since they're read-only.
if [[ -d /wattbot-data ]]; then
    echo "[start_app] Setting up data from /wattbot-data PPVC..."
    mkdir -p data

    # Copy vector DB to writable temp dir (PPVC may be read-only)
    DB_SRC="/wattbot-data/embeddings/wattbot_jinav4.db"
    DB_DST="/tmp/vectordb/wattbot_jinav4.db"
    if [[ -f "$DB_SRC" ]]; then
        mkdir -p /tmp/vectordb
        cp "$DB_SRC" "$DB_DST"
        export VECTOR_DB_PATH="$DB_DST"
        echo "[start_app] Copied vector DB to $DB_DST"
    else
        echo "[start_app] WARNING: Vector DB not found at $DB_SRC"
    fi

    # Symlink read-only data dirs
    rm -rf data/corpus data/pdfs
    ln -sf /wattbot-data/corpus data/corpus
    ln -sf /wattbot-data/pdfs   data/pdfs
else
    echo "[start_app] WARNING: /wattbot-data PPVC not mounted — using local data dirs."
fi

echo "[start_app] Installing dependencies..."
uv pip install --system streamlit openai httpx "numpy<2" python-dotenv
uv pip install --system vendor/KohakuVault vendor/KohakuRAG

echo "[start_app] Starting Streamlit on port 8501..."
exec python3 -m streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
