# WattBot RAG

Retrieval-augmented generation over research paper corpora. Streamlit
chat UI backed by vLLM, Jina V4 embeddings, and optional cross-encoder
reranking. 2025 WattBot Challenge winner.

> **New to the cluster?** Read the [top-level new-user
> guide](../docs/README.md) first — especially [00 Overview](../docs/00-overview.md)
> for the Workspace / Data Source / Data Volume mental model and
> [03 Storage](../docs/03-storage.md) for how data gets onto the
> cluster.

## Why this architecture?

A naive deployment would bundle everything into one container — load the
LLM, embed queries, and serve the UI from a single process. That works
for one developer, but falls apart with multiple users:

- **HuggingFace `model.generate()` is single-threaded.** Two users
  asking questions at the same time? One blocks until the other
  finishes.
- **GPU waste.** The Streamlit UI is pure Python/CPU, but a monolith
  allocates GPU to the entire container — even the parts that never
  touch it.
- **Rigid restarts.** Swapping the LLM kills the UI and loses user
  sessions.

Splitting into 3 independent services solves all of these:

| Workload | Type | What it does | GPU | Port |
|----------|------|-------------|-----|------|
| **`wattbot-chat`** | Inference | Serves the LLM (OpenScholar 8B) via vLLM's OpenAI-compatible API | 0.80 | 8000 |
| **`wattbot-embedding`** | Inference | Encodes user questions into vectors (Jina V4) for DB lookup | 0.10 | 8080 |
| **`wattbot-reranker`** | Inference | *(optional)* Cross-encoder reranking of retrieved passages | 0.10 | 8082 |
| **`wattbot-app`** | Workspace | Streamlit UI — connects to the other services via HTTP | 0 | 8501 |

### Multi-user scaling with vLLM

The LLM is the bottleneck in any RAG system. **vLLM** replaces naive
`model.generate()` with two key innovations:

- **Continuous batching** — instead of processing one request at a
  time, vLLM dynamically groups incoming requests into GPU batches.
  Multiple users get served concurrently, typically **2–4x more
  throughput** than HuggingFace on the same GPU.
- **PagedAttention** — manages the KV cache like virtual memory pages,
  allocating only what's actually needed per request. Standard
  frameworks pre-allocate worst-case memory and waste 60–80% of GPU
  RAM. PagedAttention enables **3–5x more concurrent requests** in the
  same memory.

Because each service is a separate RunAI **Inference** workload, you
can independently scale replicas: need more LLM throughput? Set
`wattbot-chat` to 2 replicas. Embedding bottleneck? Scale
`wattbot-embedding`. The Streamlit app stays at 1 replica (it's
stateless and cheap). RunAI handles load balancing across replicas
automatically via Knative.

### Service layout

The GPU services use **Inference** workloads (always-on, autoscalable).
The Streamlit UI uses a **Workspace** because Workspaces provide
browser-accessible proxy URLs, while Inference workloads on most
clusters only expose internal Knative routes.

```
  Users (browser)
       │
       ▼
┌─────────────────────┐
│   Streamlit App     │  CPU only, no GPU
│   Port 8501         │
└──┬──────┬───────┬───┘
   │      │       │ HTTP (internal cluster DNS)
   ▼      ▼       ▼
┌──────┐ ┌──────┐ ┌──────────┐
│ vLLM │ │Embed │ │ Reranker │
│ 8000 │ │ 8080 │ │   8082   │
│GPU80%│ │GPU10%│ │ GPU 10%  │
└──────┘ └──────┘ └──────────┘
                   (optional)
```

**Query flow:** User asks a question → Streamlit [`wattbot-app`] sends
it to the Embedding Server [`wattbot-embedding`] → gets a vector back →
searches the pre-built vector DB → sends question + retrieved context to
vLLM [`wattbot-chat`] → Streamlit [`wattbot-app`] displays the answer
with citations.

All three mount a shared model PVC at `/models/` (read-only) and share
one physical GPU via RunAI's fractional allocation. GPU budget: **1.0
GPU** total — 0.80 for vLLM, 0.10 for embeddings, 0.10 for reranker
(optional), 0 for Streamlit.

---

## RunAI Deployment

All steps use the **RunAI web UI only** — no CLI tools required.

### Prerequisite: shared models data volume

This guide assumes your cluster has an admin-provisioned
**`shared-models`** data volume that mounts read-only at `/models/` and
contains the model weights this app needs (Qwen LLM, Jina V4
embeddings, optionally Qwen2.5-VL for figure verification). On the
DoIT AI cluster this is provisioned and maintained by the cluster
admin — you just attach it to your workloads in the steps below. To
check what's available, see [Models currently on the admin's shared
PVC](docs/managing-models.md#models-currently-on-the-admins-shared-pvc).

> **Need write control?** If you want to add or swap model weights
> without going through an admin, you can provision your own
> project-level PVC instead — see [Provision Your Own Shared Models
> PVC](docs/setup-shared-models.md) *(advanced, optional)*.

### Deployment steps

Follow these docs in order:

1. **[Setup & Prerequisites](docs/setup-workspace.md)** — Create the Data Source for the vector index, clone the repo, build the index (one-time)
2. **[Deploy vLLM Server](docs/deploy-vllm.md)** — LLM inference with Qwen 7B
3. **[Deploy Embedding Server](docs/deploy-embedding.md)** — Jina V4 query encoding
4. **[Deploy Reranker Server](docs/deploy-reranker.md)** *(optional)* — Cross-encoder reranking for better retrieval quality
5. **[Deploy Streamlit App](docs/deploy-streamlit.md)** — Browser UI connecting to all services

Additional: **[Troubleshooting](docs/troubleshooting.md)** | **[Managing Models](docs/managing-models.md)** | **[Reference](docs/reference.md)**

### Deployment order notes

1. **Setup workspace** — clone repo, install deps, build vector index, then stop
2. **vLLM** — loads Qwen from `/models/` (~30s)
3. **Embedding server** — loads Jina V4 from `/models/` (~30s)
4. **Reranker** *(optional)* — loads cross-encoder from `/models/` (~10s)
5. **Streamlit app** — last, needs vLLM + embedding running (reranker is optional)

Restarts are fast since all model weights are on the shared PVC — no
downloads at runtime.

---

## Key Files

```
rag_app/
├── app.py                          # Streamlit chat UI
├── pages/1_Corpus.py               # Corpus exploration page
├── vendor/
│   ├── KohakuRAG/                  # RAG engine
│   └── KohakuVault/                # Rust+PyO3 SQLite vector store
├── scripts/
│   ├── embedding_server.py         # FastAPI Jina V4 server
│   ├── reranker_server.py          # FastAPI cross-encoder server
│   └── add_papers.py               # Corpus management
├── deploy/
│   └── runai_jobs.yaml             # RunAI job configs
├── data/                           # Corpus, embeddings, metadata
├── docs/                           # Per-step deployment guides (linked above)
├── requirements_local.txt          # GPU/local inference deps
└── requirements_remote.txt         # Remote client deps (minimal)
```
