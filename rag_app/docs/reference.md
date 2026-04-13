# Reference

## How Data Sharing Works (PVCs)

The cluster has three storage areas:

| Path | Type | Access | Size | Purpose |
|------|------|--------|------|---------|
| `/models/` | **Your shared models PVC** ([setup](setup-shared-models.md)) | **RW** by creator / **RO** for consumers | varies | Model weights (Qwen, Jina V4, etc.) |
| `/wattbot-data/` | **Project PVC (PPVC)** | RW (setup) / RO (inference) | 1 GB | Vector index, corpus, PDFs — shared across all jobs |
| `/home/jovyan/work/` | **Personal workspace** | Read-write | 30 GB | Git repo, Python deps, cache |

```
/models/                                    ← your shared models PVC (RW by creator, RO for others)
└── .cache/huggingface/
    ├── models--Qwen--Qwen2.5-7B-Instruct/
    ├── models--Qwen--Qwen2.5-14B-Instruct/
    ├── models--Qwen--Qwen2.5-72B-Instruct/
    ├── models--Qwen--Qwen3.5-35B-A3B/
    ├── models--jinaai--jina-embeddings-v4/
    └── ...  (~744 GB total)

/wattbot-data/                          ← PPVC, shared across jobs
├── embeddings/
│   └── wattbot_jinav4.db                   # vector index (~130 MB)
├── corpus/                                 # parsed JSON documents
└── pdfs/                                   # cached source PDFs

/home/jovyan/work/                          ← personal, read-write
├── RunAI_apps/                           # git repo (cloned in Step 0)
│   ├── app.py                              # Streamlit app
│   ├── data/
│   │   ├── metadata.csv                    # document URLs
│   │   ├── embeddings -> /wattbot-data/embeddings  # symlink
│   │   ├── corpus -> /wattbot-data/corpus          # symlink
│   │   └── pdfs -> /wattbot-data/pdfs              # symlink
│   ├── rag_app/vendor/KohakuRAG/                   # RAG library
│   └── scripts/
│       └── embedding_server.py             # query-time embedding server
└── .cache/
    └── huggingface/                        # for any new model downloads
```

In the RunAI UI, these are exposed as:

| RunAI Name | Mount Point | Access | Used by |
|------------|-------------|--------|---------|
| `shared-model-repository` (Data Source) | `/models` | Read-only | All workloads — pre-cached model weights |
| `wattbot-data` (Project PVC) | `/wattbot-data` | RW for setup, RO for inference | Vector index, corpus, PDFs |
| Personal workspace | `/home/jovyan/work` | Read-write | Setup Workspace (clone, build) |

**Key points:**
- Model weights are **already pre-cached** — no need to download Qwen
  or Jina V4
- The shared Data Volume is **read-only** — inference jobs can't
  accidentally modify or delete weights
- The **PPVC** (`wattbot-data`) holds the vector index — all jobs
  mount it, so there's a single copy of the DB (no duplication)
- Your personal workspace persists across workspace restarts
- The Workspace does NOT need to be running for Inference jobs to read
  from the PPVC or shared Data Volume

### What lives where

| Item | Location | Size | Written by |
|------|----------|------|------------|
| Qwen 7B model weights | `/models/.cache/huggingface/` | ~14 GB | Pre-cached on shared Data Volume |
| Qwen 14B, 72B, 3.5-35B, etc. | `/models/.cache/huggingface/` | ~744 GB total | Pre-cached on shared Data Volume |
| Jina V4 model weights | `/models/.cache/huggingface/` | ~3 GB | Pre-cached on shared Data Volume |
| Vector index (`wattbot_jinav4.db`) | `/wattbot-data/embeddings/` | ~130 MB | Setup Workspace (on PPVC) |
| Parsed corpus (JSON) | `/wattbot-data/corpus/` | ~50 MB | Setup Workspace (on PPVC) |
| Cached PDFs | `/wattbot-data/pdfs/` | ~200 MB | Setup Workspace (on PPVC) |
| Git repo clone | `/home/jovyan/work/RunAI_apps/` | ~50 MB | Setup Workspace |
| Python packages + cache | `/home/jovyan/work/.cache/` | Varies | Setup Workspace |

---

## Data Sources vs Data Volumes

RunAI has two related but distinct concepts for storage. Understanding
the difference is critical for managing model weights:

| Concept | What it is | Who can write | Who can read |
|---------|-----------|---------------|--------------|
| **Data Source** (PVC) | The actual storage — a Kubernetes PVC in your project's namespace | **Only workloads in the creator's project** | Creator's project |
| **Data Volume** | A shareable wrapper around a Data Source | Nobody (read-only replicas) | Any project it's shared with |

**Key insight:** RunAI creates **read-only replica PVCs** in each
consumer's namespace when you share a Data Volume. The replicas point
to the same underlying storage, but write access is stripped. Only the
original project can mount the data source with write access.

This means:
- To **read** models: mount the Data Volume (any project)
- To **add/update** models: mount the Data Source from the creator's
  project (see [Setup Shared Models](setup-shared-models.md))
- You **cannot** write to someone else's PVC, even if you're an admin
  on the cluster — you'd need to create a workload in their project

Your personal workspace at `/home/jovyan/work/` is separate writable
storage for code, indexes, and caches.

---

## Access Control

### Who can modify the shared models PVC?

| Action | Who can do it |
|--------|--------------|
| **Write** to the PVC (add/remove models) | Only workloads in the **creator's project** mounting the **data source** |
| **Read** via Data Volume | Any project the Data Volume is shared with |
| Create / delete Data Volumes | Users with the **Data Volumes Administrator** role |
| Share Data Volumes across projects | Data Volumes Administrator |

### Preventing accidents

- Model weights are expensive to re-download. The Data Volume
  mechanism ensures consumers can't accidentally delete or corrupt
  weights — they only get read-only access.
- When provisioning, use a dedicated `model-provisioner` Workspace
  (see [Setup Shared Models](setup-shared-models.md)) rather than
  downloading from inference jobs. This keeps the write path isolated
  and intentional.

- **Use a naming convention.** Models live under
  `/models/.cache/huggingface/models--<org>--<name>/`. Don't put
  arbitrary files at the PVC root — keep it organized.
- **Document what's on the PVC.** After adding or removing a model, run
  `du -sh /models/.cache/huggingface/models--*/` and note the change
  so the team knows what's available.

---

## Why This Architecture?

RunAI offers three workload types: **Workspace** (interactive dev),
**Training** (batch jobs), and **Inference** (always-on serving). We use
Inference because we want WattBot available as a persistent service — not
something that has to be manually launched each time.

### Why vLLM?

Standard HuggingFace `model.generate()` processes one request at a time —
if two users send a question simultaneously, one blocks until the other
finishes. This is fine for a single developer but breaks down for any
multi-user deployment like a RAG system.

**vLLM** solves this with two key innovations:

- **Continuous batching** — instead of waiting for one request to finish
  before starting the next, vLLM dynamically groups incoming requests
  into GPU batches. Multiple users get served concurrently on a single
  GPU, typically 2-4x more throughput than naive generation.
- **PagedAttention** — LLM inference is bottlenecked by the KV cache
  (key-value memory that grows with sequence length). Standard frameworks
  pre-allocate worst-case memory per request, wasting 60-80% of GPU RAM.
  PagedAttention manages KV cache like virtual memory pages — allocating
  only what's actually needed and sharing common prefixes across requests.
  This means vLLM can serve **3-5x more concurrent requests** in the same
  GPU memory compared to naive HuggingFace serving.

For a RAG system where multiple users may query at once, each with
different context lengths, this memory efficiency is critical.

**What vLLM replaces (and what it doesn't):** vLLM only handles the
"run the LLM on the GPU" part. It exposes an OpenAI-compatible API
(`/v1/chat/completions`) that our code calls over HTTP. Everything
else — the RAG pipeline, retrieval, context assembly, prompt
construction, embedding search — is still our custom KohakuRAG code.
We wrote `VLLMChatModel` (in `kohakurag/remote.py`) as a thin client
that sends our assembled prompts to vLLM and gets completions back.
Think of vLLM as replacing `model.generate()`, not replacing our RAG
logic.

### Why not a single monolithic Inference job?

You *could* bundle vLLM + Jina V4 + Streamlit into one container — and
it would technically work. But splitting them out has practical benefits:

- **Wasted GPU on the UI.** Streamlit is pure Python/CPU. In a monolith,
  RunAI allocates GPU to the whole container even though the UI never
  touches it. Splitting lets the Streamlit job request 0 GPU.
- **Rigid scaling.** With a monolith you can't independently restart the
  LLM (e.g. to swap from Qwen 7B to a larger model) without also
  killing the UI and losing user sessions. Separate jobs let you restart
  one without affecting the others.
- **Simpler containers.** The Streamlit app only needs `pip install
  streamlit openai httpx` — a tiny image. A monolith needs PyTorch,
  vLLM, and Jina V4 all in one image, which is harder to build and
  debug.

### Why three jobs instead of two?

A natural simplification is two jobs: **Job 1** runs vLLM (LLM only),
and **Job 2** bundles the Streamlit UI with Jina V4 embeddings together.
Fewer moving parts, but now Job 2 needs a GPU for Jina V4 (~3 GB VRAM),
so you can't use a lightweight CPU-only image — you'd need a full
PyTorch + CUDA container just for the UI pod.

By splitting into three — vLLM, embedding server, Streamlit — each job
gets exactly the resources it needs. The embedding model (Jina V4,
~3 GB VRAM) and the LLM (Qwen 7B, ~6-14 GB) have very different
resource profiles, so RunAI can allocate fractional GPU to each (`1.0`
for vLLM, `0.5` for embeddings) across the available 1.5 GPUs (~90 GB).
The Streamlit app gets `0` GPU — just CPU and RAM.

### Alternatives considered

| Option | What's in each job | Pros | Cons |
|--------|-------------------|------|------|
| Single Workspace job | Everything in one process (LLM + embeddings + UI) | Simple | Not persistent, no batching, wastes GPU on UI |
| Two jobs | **Job 1:** vLLM (LLM only) — **Job 2:** Streamlit + Jina V4 embeddings bundled together | Fewer moving parts | Job 2 needs GPU for Jina V4, can't use lightweight `python:3.11-slim` image |
| **Three jobs (chosen)** | **Job 1:** vLLM (LLM, 1.0 GPU) — **Job 2:** Jina V4 (embeddings, 0.5 GPU) — **Job 3:** Streamlit (UI, CPU-only) | Best resource efficiency, independent scaling | More services to configure |

### Index Build vs Query Serving

The system has two distinct phases that use embeddings differently:

**Phase 1: Index Build (one-time, batch)** — Embeds all documents in
the corpus into a vector database. This runs once (or when the corpus
changes) and produces `wattbot_jinav4.db`. Use a RunAI **Workspace**
for this ([Setup](setup-workspace.md)).

**Phase 2: Query Serving (always-on, 3 Inference jobs)** — Handles
live user queries. The embedding server only encodes the user's
question (a few sentences) for vector search — it does NOT re-embed
the corpus.

---

## Local Development

You can still run everything locally for development:

```bash
# Default — local GPU models
streamlit run rag_app/app.py

# Test remote mode against local services:
# Terminal 1: python rag_app/scripts/embedding_server.py
# Terminal 2: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct
# Terminal 3: RAG_MODE=remote streamlit run rag_app/app.py
```
