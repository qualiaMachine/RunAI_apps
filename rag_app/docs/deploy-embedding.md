# Deploy the Embedding Server

The embedding server is a custom FastAPI service that wraps Jina V4.
We use the same `vllm/vllm-openai` image as the vLLM server — it
already has PyTorch, CUDA, and `curl` pre-installed, so only a handful
of lightweight Python packages need to be added at startup.

> **Why a custom FastAPI server instead of vLLM's `--task embed`?**
> - **Jina V4** uses per-task LoRA adapters (retrieval, code, …) swapped
>   at inference time — vLLM's embedding path doesn't support that.
> - **Read-only PVC**: HF writes metadata on every load; the server
>   creates a writable `/tmp/hf_home` overlay that symlinks weights from
>   the PVC.
> - **Per-request `energy_wh`** in the `/embed` response (NVML).
>
> Throughput gap to vLLM is small for embedding (single forward pass, no
> KV cache).

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

## Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project (e.g. `jupyter-endemann01`) |
| **Inference type** | **Custom** (not "Model: from Hugging Face") |
| **Inference name** | `wattbot-embedding` |

## Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `vllm/vllm-openai:latest` |
| **Image pull** | Pull the image only if it's not already present on the host (recommended) |
| **Image pull secret** | *(leave empty — public Docker Hub image)* |

## Serving endpoint

| Field | Value |
|-------|-------|
| **Protocol** | HTTP |
| **Container port** | `8080` |

## Runtime settings

Inference jobs don't have access to the personal workspace
(`/home/jovyan/work/`), so the command downloads the repo as a
tarball and installs dependencies at startup.

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | `-c "pip install uv && curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp && mv /tmp/RunAI_apps-main /tmp/RunAI_apps && cd /tmp/RunAI_apps && uv pip install --system fastapi uvicorn httpx numpy sentence-transformers 'transformers>=4.42,<5' accelerate huggingface_hub peft && python3 rag_app/scripts/embedding_server.py"` |
| **Working directory** | *(leave empty)* |

Same shape as the [reranker server](deploy-reranker.md) — pull the
repo tarball into the vLLM image, install our deps, then run our own
FastAPI wrapper. Piece by piece:

| Chunk | Why it's there |
|-------|----------------|
| `-c "..."` | Tells `bash` to run the rest as a shell command, then exit. The whole arguments value is one string. |
| `pip install uv` | Pulls in `uv`, a drop-in replacement for `pip` that's 10-100x faster. Installs that take 1-3 minutes with pip finish in seconds. |
| `curl -sL https://github.com/.../main.tar.gz \| tar xz -C /tmp` | Pull the current `main` branch as a tarball and unpack it under `/tmp`. The `vllm/vllm-openai` image doesn't include `git`, so we use `curl` (which is pre-installed) instead of `git clone`. |
| `mv /tmp/RunAI_apps-main /tmp/RunAI_apps` | GitHub's tarball unpacks to `<repo>-<branch>/`. Rename to a stable path. To deploy a different branch, swap `main` in both the URL and the `mv` target — GitHub converts `/` → `-` in the tarball directory name (so `refs/heads/claude/my-feature.tar.gz` becomes `RunAI_apps-claude-my-feature/`). |
| `cd /tmp/RunAI_apps` | The embedding server script is referenced from the repo root. |
| `uv pip install --system fastapi uvicorn httpx numpy sentence-transformers 'transformers>=4.42,<5' accelerate huggingface_hub peft` | Server runtime deps. `fastapi` + `uvicorn` for the HTTP wrapper; `sentence-transformers` + `transformers` + `accelerate` to load Jina V4; `peft` for LoRA adapter support that Jina V4 needs; `huggingface_hub` for the auto-adapter-download fallback. The `transformers>=4.42,<5` pin avoids the 5.x ABI break. `--system` installs into the container's system Python (no venv needed in an ephemeral pod). |
| `python3 rag_app/scripts/embedding_server.py` | Start the server as the long-running foreground process. The script binds `0.0.0.0:8080` (matching the **Serving endpoint** above) and creates the writable HF cache overlay (see read-only PVC note below). The image provides `python3` but not the `python` alias, so we call `python3` explicitly. |

**Environment variables:**

| Name | Value |
|------|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v4` |
| `EMBEDDING_DIM` | `1024` |
| `EMBEDDING_TASK` | `retrieval` |

> **Read-only PVC handling:** The shared models PVC is often read-only
> despite being configured for read-write (a RunAI/cluster admin issue).
> The embedding server handles this automatically by creating a writable
> overlay at `/tmp/hf_home` that symlinks model weights from the PVC while
> redirecting all metadata writes to `/tmp`. No manual workaround needed —
> just set `HF_HOME` to the PVC path and the server handles the rest.
>
> **Adapters:** If the shared PVC is missing the Jina V4 `adapters/`
> directory, the server auto-downloads them to `/tmp` on startup (requires
> internet access, ~few hundred MB). This re-downloads on each cold start.
> To avoid this, add adapters to the shared PVC using the provisioning
> script (see [Managing Models](managing-models.md)).
> The startup command also installs `peft` (required for LoRA adapter
> support).

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled — set to `10%` of device |
| **CPU request** | *(leave default)* |
| **CPU memory request** | *(leave default)* |
| **Replica autoscaling** | Min `1`, Max `1` (no autoscaling) |

## Data & storage

Under **Data & storage**, add the data volumes and set container paths:

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |
| `wattbot-data` | `/wattbot-data` |

## General

| Field | Value |
|-------|-------|
| **Priority** | `very-high` (or as appropriate) |

## Expected startup time

First deploy takes **3-5 minutes**:
- **Image pull** (~2-3 min): If the vLLM image is already cached on
  the node (from the vLLM server deploy), this is instant. Otherwise ~15 GB download.
- **Dependency install** (~30-60s): `uv` installs FastAPI,
  sentence-transformers, etc. (only a few lightweight packages —
  PyTorch is already in the image).
- **Model loading** (~30s): Jina V4 weights (~3 GB) load from the
  shared PVC into GPU memory.

## CLI equivalent

```bash
runai submit wattbot-embedding \
  --type inference \
  --image vllm/vllm-openai:latest \
  --gpu 0.10 \
  --cpu 2 \
  --memory 8Gi \
  --pvc shared-models:/models \
  --env HF_HOME=/models/.cache/huggingface \
  --env EMBEDDING_MODEL=jinaai/jina-embeddings-v4 \
  --env EMBEDDING_DIM=1024 \
  --env EMBEDDING_TASK=retrieval \
  --port 8080 \
  --command -- /bin/bash -c \
    "pip install uv && \
     curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz \
       | tar xz -C /tmp && \
     mv /tmp/RunAI_apps-main /tmp/RunAI_apps && \
     cd /tmp/RunAI_apps && \
     uv pip install --system fastapi uvicorn httpx numpy \
       sentence-transformers 'transformers>=4.42,<5' accelerate && \
     python3 rag_app/scripts/embedding_server.py"
```

> **Note on the vLLM image:**
> - Uses `python3` (not `python`) — there is no `python` alias
> - Does not include `git` — use `curl` tarball instead of `git clone`
> - Has `curl`, `pip`, PyTorch, and CUDA pre-installed
