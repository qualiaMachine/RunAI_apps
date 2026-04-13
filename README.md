# RunAI / PowerEdge Applications

GPU-accelerated applications for UW-Madison research computing, deployed on
RunAI with Dell PowerEdge infrastructure. Built for DoIT and the research
community — production-ready patterns for running open-source AI models on
local GPU clusters instead of cloud APIs.

## Applications

### [Document Extraction (`ocr_app/`)](ocr_app/README.md)

Structured data extraction from institutional documents — grant awards,
budgets, terms & conditions, archival scans, and other records. Processes
document archives (TIFF and PDF) into structured JSON for downstream
systematic analysis.

- **Hybrid pipeline:** digital PDFs get instant text extraction; scanned
  pages fall back to VLM OCR (Qwen3-VL-32B-Instruct)
- **Batch mode:** async concurrent processing with resume support for
  large-scale runs
- **Interactive mode:** Streamlit UI for PoC demos and format exploration
- Outputs structured JSON (award details, budget line items, regulatory
  citations, key-value pairs, tables)

**Status:** PoC — testing on sample documents from DoIT imaging service.

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

### What's here

Each app includes:
- `deploy/runai_jobs.yaml` — RunAI job configs with copy-paste arguments
- `docs/` — Step-by-step deployment guides (data volume setup, model
  provisioning, workspace config, troubleshooting)
- Requirements files for CPU and GPU components

### Deployment pattern

All apps use the same approach:

```
  +------------------+
  |   App (CPU)      |  Workspace -- code pulled from GitHub at startup
  |   Streamlit /    |  via curl|tar + uv pip install
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
| `provision_shared_models.py` | Download HuggingFace models to shared PVC |
| `setup_poweredge_pod.sh` | PowerEdge pod initialization |

## Getting Started

1. **Pick an app** and read its README
2. **Set up a shared models PVC** — see
   [rag_app/docs/setup-shared-models.md](rag_app/docs/setup-shared-models.md)
   (same PVC works for all apps)
3. **Follow the app's deployment guide** in its `docs/` directory
4. **Deploy via RunAI UI** using the configs in the app's `deploy/` directory

## Author

- **Chris Endemann** — Research Cloud Consultant, RCI/DoIT, UW-Madison
