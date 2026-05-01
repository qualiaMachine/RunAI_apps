# Deploy the Reranker Server

The reranker server is a custom FastAPI service that wraps a
cross-encoder model (e.g. `BAAI/bge-reranker-v2-m3` or
`OpenSciLM/OpenScholar_Reranker`). It follows the same pattern as the
embedding server — same base image, same PVC overlay, same startup
approach.

> **Why a separate inference job?** Cross-encoder models need GPU for
> inference. Running one inside the Streamlit app (CPU-only workspace)
> would be too slow. A separate inference job lets the reranker run on
> its own GPU fraction while the Streamlit app calls it over HTTP.

> **Why a custom FastAPI server (not vLLM `--task score`)?** BGE reranker
> *is* vLLM-compatible, but we use the same custom-server pattern as the
> [embedding server](deploy-embedding.md) for symmetry — same read-only
> PVC overlay, same per-request energy reporting.

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

## Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project (e.g. `jupyter-endemann01`) |
| **Inference type** | **Custom** (not "Model: from Hugging Face") |
| **Inference name** | `wattbot-reranker` |

## Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `vllm/vllm-openai:latest` |
| **Image pull** | Pull only if not present (recommended) |

## Serving endpoint

| Field | Value |
|-------|-------|
| **Protocol** | HTTP |
| **Container port** | `8082` |
| **Access** | External (Public access) |

## Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | see below |
| **Working directory** | *(leave empty)* |

**Arguments** (copy-paste this entire block):

```
-c "pip install uv && curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp && mv /tmp/RunAI_apps-main /tmp/RunAI_apps && cd /tmp/RunAI_apps && uv pip install --system fastapi uvicorn sentence-transformers && python3 rag_app/scripts/reranker_server.py"
```

Same shape as the [embedding server](deploy-embedding.md) — pull the
repo tarball into the vLLM image, install a few packages, then run our
own FastAPI wrapper. Piece by piece:

| Chunk | Why it's there |
|-------|----------------|
| `-c "..."` | Tells `bash` to run the rest as a shell command, then exit. The whole arguments value is one string. |
| `pip install uv` | Pulls in `uv` so the heavier installs below run in seconds instead of minutes. |
| `curl -sL https://github.com/.../main.tar.gz \| tar xz -C /tmp` | Pull the current `main` branch as a tarball and unpack it under `/tmp`. The `vllm/vllm-openai` image doesn't ship `git`, so we use `curl` (which is present) instead of `git clone`. |
| `mv /tmp/RunAI_apps-main /tmp/RunAI_apps` | GitHub's tarball unpacks to `<repo>-<branch>/`. Rename to a stable path. To deploy a different branch, swap `main` everywhere — note GitHub converts `/` → `-` in the tarball directory name. |
| `cd /tmp/RunAI_apps` | The reranker server script is referenced from the repo root. |
| `uv pip install --system fastapi uvicorn sentence-transformers` | Server runtime deps. `fastapi` + `uvicorn` for the HTTP wrapper, `sentence-transformers` for the cross-encoder model. `--system` installs into the container's system Python (no venv needed in an ephemeral pod). |
| `python3 rag_app/scripts/reranker_server.py` | Start the server as the long-running foreground process. The script binds `0.0.0.0:8082` (matching the **Serving endpoint** above) and creates a writable HF cache overlay so the read-only `shared-models` PVC isn't a problem. The image provides `python3` but not the `python` alias, so we call `python3` explicitly. |

**Environment variables:**

| Name | Value |
|------|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` |

> **Model options:**
> - `BAAI/bge-reranker-v2-m3` — small, fast, multilingual (~0.5 GB VRAM). Good default.
> - `BAAI/bge-reranker-large` — better quality (~1.3 GB VRAM)
> - `OpenSciLM/OpenScholar_Reranker` — science-tuned, best for research papers (~1.2 GB VRAM)
>
> You must pre-download the model to the shared PVC before deploying.
> See [Pre-downloading the model](#pre-downloading-the-model) below.

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled — set to `10%` of device (reranker models are small, ~0.5 GB VRAM) |
| **CPU request** | *(leave default)* |
| **CPU memory request** | *(leave default)* |
| **Replica autoscaling** | Min `1`, Max `1` (no autoscaling) |

## Data & storage

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |

> The reranker does not need `wattbot-data` — it only scores
> query–passage pairs, it doesn't read the vector index.

## General

| Field | Value |
|-------|-------|
| **Priority** | `very-high` (or as appropriate) |

## Expected startup time

**2-3 minutes** on first deploy:
- **Image pull**: Instant if vLLM image already cached on node
- **Dependency install** (~30s): Same packages as embedding server
- **Model loading** (~10-30s): BGE reranker is small (~0.5 GB)

---

## Pre-downloading the model

The reranker model must be on the shared PVC before deployment (the
server runs in offline mode). From your workspace terminal:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-reranker-v2-m3', cache_dir='/models/.cache/huggingface')
"
```

For the science-tuned model:
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('OpenSciLM/OpenScholar_Reranker', cache_dir='/models/.cache/huggingface')
"
```

---

## Connecting the Streamlit app

Add one more environment variable to your Streamlit workspace:

| Name | Value |
|------|-------|
| `RERANKER_SERVICE_URL` | `http://wattbot-reranker.runai-<project>.svc.cluster.local` |

> **Important:** Use the full FQDN (no port number). Inference workloads
> use Knative/envoy which listens on port 80. Short names like
> `wattbot-reranker:8082` will return envoy 404.

Once set, the "Cross-encoder reranker" toggle will appear in the
Streamlit sidebar under "Retrieval enhancements". It works with or
without Semantic Scholar search enabled:
- **Reranker only**: Reranks local retrieval matches for better ordering
- **Reranker + S2**: Jointly reranks local + Semantic Scholar results so
  they compete fairly on relevance

---

## CLI equivalent

```bash
runai submit wattbot-reranker \
  --type inference \
  --image vllm/vllm-openai:latest \
  --gpu 0.10 \
  --cpu 2 \
  --memory 4Gi \
  --pvc shared-models:/models \
  --env HF_HOME=/models/.cache/huggingface \
  --env RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
  --port 8082 \
  --command -- /bin/bash -c \
    'pip install uv && \
     curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz \
       | tar xz -C /tmp && \
     mv /tmp/RunAI_apps-main /tmp/RunAI_apps && \
     cd /tmp/RunAI_apps && \
     uv pip install --system fastapi uvicorn sentence-transformers && \
     python3 rag_app/scripts/reranker_server.py'
```

## Updated architecture

```
  ┌─────────────────────┐
  │   Streamlit App     │  (Workspace, CPU only)
  │   Port 8501         │
  └──┬──────┬───────┬───┘
     │      │       │ HTTP (internal cluster DNS)
     ▼      ▼       ▼
  ┌──────┐ ┌──────┐ ┌──────────┐
  │ vLLM │ │Embed │ │ Reranker │
  │ 8000 │ │ 8080 │ │   8082   │
  │GPU80%│ │GPU10%│ │  GPU 10% │
  └──────┘ └──────┘ └──────────┘
     Total: 1.00 GPU
```
