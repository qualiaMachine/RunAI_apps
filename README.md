# RunAI / PowerEdge Applications

GPU-accelerated applications for UW-Madison research computing, deployed on
RunAI with Dell PowerEdge infrastructure. Built for DoIT and the research
community — production-ready patterns for running open-source AI models on
local GPU clusters instead of cloud APIs.

## New User Guide

A progressive walkthrough of the RunAI cluster for researchers and lab
admins onboarding to the pilot. Read in order if you're new; skip
ahead if you already have your bearings.

| # | Doc | Read this if... |
|---|-----|-----------------|
| 00 | [Overview](docs/00-overview.md) | You're new to the NVIDIA RunAI cluster and want to know whether it fits your work |
| 01 | [Access](docs/01-access.md) | You need a login, project assignment, or storage quota |
| 02 | [First workspace](docs/02-first-workspace.md) | You want a working Jupyter notebook on the cluster with this repo cloned and a shared model loaded, in ~15 minutes |
| 03 | [Share a model as a vLLM endpoint](docs/03-share-as-endpoint.md) | You want to host a model once and have multiple users / workloads hit it via HTTP, instead of every user loading their own copy onto a GPU |
| 04 | [Storage](docs/04-storage.md) | You need to know where data lives — short-term scratch through cluster-wide shared datasets — and how to get it from "a drive in my lab" to "mountable in a workload" |
| 05 | [Examples](docs/05-examples.md) | You're ready to deploy something — pointers to the OCR pipeline, the RAG/chatbot, and the patterns to copy when building your own |
| 06 | [Expose a vLLM endpoint outside the cluster](docs/06-external-endpoint.md) | A non-RunAI client (Denodo, an institutional app, your laptop) needs to call a hosted model over `https://`, including the cross-VLAN firewall hand-off |

The OCR-specific and RAG-specific deployment guides live in the app
READMEs — [`ocr_app/README.md`](ocr_app/README.md) and
[`rag_app/README.md`](rag_app/README.md) — with per-step details
under each app's `docs/`. Those assume you've already worked through
00–04 here. All workloads are created through the RunAI web UI.

## Applications

This repo is a growing collection of example use cases — PoCs we are building out as the pilot uncovers what labs actually need. Use
them as starting templates and adapt to your needs.

| Path | What it is | When you'd use it |
|------|------------|-------------------|
| [`ocr_app/`](ocr_app/README.md) | Vision-language document extraction (Qwen3-VL-32B). Turns PDFs/scans into structured JSON. | Grant administration, archival corpora, library digitization, anything where layout matters |
| [`rag_app/`](rag_app/README.md) | Retrieval-augmented chatbot over a curated corpus (Qwen 7B/14B/72B). | Q&A over institutional knowledge bases, research literature search, "ChatGPT for our docs" |
| [`scripts/`](scripts) | Shared utilities used by both apps. | You usually don't touch this directly. |


### [Document Extraction (`ocr_app/`)](ocr_app/README.md)

Structured data extraction from institutional documents — grant award
notices, budgets, terms & conditions, archival scans, and other records.
Every page is rendered as an image and sent to a Vision Language Model
(Qwen3-VL-32B-Instruct-AWQ), so the model sees layout, tables, signatures,
watermarks, and annotations — not just raw text.

- **Chunk-based two-pass pipeline (notebooks):** overlapping page chunks
  sent in single VLM calls, then programmatic merge + continuation-flag
  stitching across chunks, then a doc-level pass-2 synthesis
- **Two variants:** grant administration schema
  (stakeholders/tables/narratives) and library/archival schema
  (bibliographic metadata, body text, marginalia, stamps)

**Status:** PoC validated on sample documents.

### [RAG Chat (`rag_app/`)](rag_app/README.md)

WattBot — retrieval-augmented generation over research paper corpora.
Chat interface for querying scientific literature with citations. 2025
WattBot Challenge winner.

- **4 services on 1 GPU:** vLLM (LLM), Jina V4 (embeddings),
  cross-encoder (reranker), Streamlit (UI) via fractional GPU allocation
- Multiple knowledge bases, hybrid search (vector + BM25)
- Supports Qwen, Llama, OpenScholar models

**Status:** Deployed on RunAI, documented end-to-end.

## Infrastructure

Each app includes:
- `app.py` and supporting scripts (Streamlit UI, FastAPI servers, batch
  CLI)
- `docs/` — step-by-step RunAI deployment guides (storage setup,
  model provisioning, workspace config, troubleshooting)
- Requirements files split by role (UI/client vs. GPU server)

All apps use the same approach:

```
  +------------------+
  |   App (CPU)      |  Workspace -- code pulled from GitHub at startup
  |   Streamlit /    |  and installed with uv/pip
  |   FastAPI / CLI  |
  +--------+---------+
           | HTTP (cluster-internal DNS)
           v
  +------------------+
  |  vLLM / Ollama   |  Inference workload -- GPU, fractional allocation
  |  Model serving   |  Weights loaded from shared PVC
  +------------------+
```

- **No Docker builds** — stock images (`vllm/vllm-openai`,
  `nvcr.io/nvidia/pytorch`) with deps installed at startup
- **Fractional GPU** — multiple services share one physical GPU
- **Shared model PVC** — download once, mount read-only everywhere
- **Knative DNS** — services addressed via FQDN
  (`workload.runai-project.svc.cluster.local`)

### Shared utilities (`scripts/`)

| Script | Purpose |
|--------|---------|
| `hardware_metrics.py` | GPU/energy profiling — VRAM, power draw, energy per request |
| `provision_shared_models.py` | Download HuggingFace models to the shared PVC |



## Author

- **Chris Endemann** — Research Cyberinfrastructure Consultant, RCI/DoIT, UW-Madison
