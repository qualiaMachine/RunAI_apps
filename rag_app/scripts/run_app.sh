#!/usr/bin/env bash
# ===========================================================================
# run_app.sh — Launch the Streamlit app inside a Run:ai pod.
#
# The app binds to 0.0.0.0:8501 so it is reachable from outside the pod
# (via Run:ai nodeport, port-forward, or ingress).
#
# Usage:
#   bash scripts/run_app.sh           # default port 8501
#   PORT=8080 bash scripts/run_app.sh # custom port
# ===========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8501}"

# Quick sanity checks
if ! python -c "import streamlit" 2>/dev/null; then
    echo "ERROR: streamlit not installed."
    exit 1
fi

DB_PATH="$REPO_ROOT/data/embeddings/wattbot_jinav4.db"
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Vector database not found at $DB_PATH"
    exit 1
fi

echo "Starting WattBot RAG on 0.0.0.0:$PORT"
echo ""
echo "Access the app:"
echo "  - Inside the pod:   http://localhost:$PORT"
echo ""
echo "  - Via jupyter-server-proxy (recommended):"
echo "    https://deepthought.doit.wisc.edu/doit-ai-eval/<WORKSPACE-NAME>/proxy/$PORT/"
echo "    Replace <WORKSPACE-NAME> with your Run:ai workspace name."
echo "    Requires jupyter-server-proxy in the system Python (see docs/Streamlit_App_Guide.md)."
echo ""
echo "  - Via kubectl port-forward:"
echo "    kubectl port-forward <pod-name> $PORT:$PORT"
echo "    then open http://localhost:$PORT"
echo ""

exec streamlit run app.py \
    --server.port="$PORT" \
    --server.address="0.0.0.0" \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
