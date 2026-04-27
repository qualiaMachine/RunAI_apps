# Deploy the Embedding Server

The embedding server is a custom FastAPI service that wraps Jina V4.
We use the same `vllm/vllm-openai` image as the vLLM server — it
already has PyTorch, CUDA, and `curl` pre-installed, so only a handful
of lightweight Python packages need to be added at startup.

> **Why the vLLM image?** Using one image for both services means
> fewer image pulls and less storage on each node. The vLLM image
> (~15 GB) ships with PyTorch + CUDA, which is everything the
> embedding server needs. The NGC PyTorch image
> (`nvcr.io/nvidia/pytorch:25.02-py3`, ~20 GB) also works but is
> larger and adds no benefit here.

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

> **Using a different branch?** Replace `main`
> in the URL and the `mv` target with your branch name. GitHub converts
> `/` → `-` in tarball directory names:
> ```
> # URL:  .../refs/heads/claude/my-feature.tar.gz   (slashes OK)
> # mv:   RunAI_apps-claude-my-feature            (slashes become dashes)
> ```
>
> **Why `curl` tarball instead of `git clone`?** The vLLM image
> doesn't include `git`. Downloading a tarball via `curl` (which is
> pre-installed) avoids needing to install git at runtime.
>
> **Why `python3` not `python`?** The vLLM image provides `python3`
> but does not alias it to `python`.
>
> **Why uv?** `uv` is a drop-in replacement for `pip` that's 10-100x
> faster. Installs that take 1-3 minutes with pip finish in seconds.

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

## How it works

Jina V4 weights are already pre-cached on the shared PVC at
`/models/.cache/huggingface/`. The server loads the model into GPU
memory and exposes a FastAPI endpoint for encoding queries into vectors.

**Verify (from any other pod's terminal):**
```bash
curl http://wattbot-embedding:8080/health
# {"status": "ok"}
```

---

## Pre-deploy testing

These steps catch import errors, missing dependencies, and startup crashes
BEFORE you waste time on a 5-min container deploy cycle.

### Step 1: Test the startup command locally (any machine with Python 3.10+)

```bash
rm -rf /tmp/RunAI_apps && \
curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/<your-branch>.tar.gz \
  | tar xz -C /tmp && \
mv /tmp/RunAI_apps-<your-branch> /tmp/RunAI_apps && \
cd /tmp/RunAI_apps && \
uv pip install --system fastapi uvicorn httpx numpy \
  sentence-transformers 'transformers>=4.42,<5' accelerate && \
python3 rag_app/scripts/embedding_server.py
```

Expected output: `[embedding_server] Model loaded in Xs. Serving on 0.0.0.0:8080`
The server should start without import errors. Ctrl+C to stop.

### Step 2: Smoke-test the /health endpoint (in a second terminal)

```bash
curl http://localhost:8080/health
# Expected: {"status":"ok"}
```

### Step 3 (optional): Test an actual embedding call

```bash
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world"]}'
# Expected: {"embeddings":[[...]], "dimension":1024, "count":1, ...}
```

Only proceed to RunAI deployment once all steps pass locally.

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
