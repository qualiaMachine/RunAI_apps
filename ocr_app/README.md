# Document Extraction

Structured data extraction from grant award notices, budgets, terms &
conditions, archival scans, and other institutional documents. Produces
JSON for downstream systematic analysis.

All pages (digital PDFs, scans, TIFFs) are rendered as images and sent
to a Vision Language Model (Qwen3-VL-32B-Instruct-AWQ) for structured
extraction. This ensures the VLM sees layout, tables, signatures,
watermarks, and annotations — not just raw text. Traditional OCR
pipelines (Tesseract + regex) are brittle on layout changes and need
per-document-type rules; the VLM approach replaces both.

> **New to the cluster?** Read the [top-level new-user
> guide](../docs/README.md) first — especially [00 Overview](../docs/00-overview.md)
> for the Workspace / Data Source / Data Volume mental model and
> [03 Storage](../docs/03-storage.md) for how data gets onto the
> cluster.

## Two pipelines

Two code paths share the same VLM and the same output schemas, tuned to
different use cases:

| Pipeline | Best for | Entry point |
|----------|----------|-------------|
| **Chunk-based, two-pass** (notebooks) | High-fidelity extraction with cross-page table/narrative stitching, doc-level synthesis | `notebooks/*.ipynb` |
| **Per-page** (batch + Streamlit) | One-shot demos and high-throughput bulk runs where per-page JSON is enough | `app.py`, `scripts/batch_extract.py`, `scripts/ocr_server.py` |

### Chunk-based pipeline (notebooks)

**Pass 1 (chunk extraction):** The document is split into overlapping
page chunks (default: 20 pages per chunk, 1-page overlap). Each chunk
is sent to the VLM as a single call with every page as an image, plus a
boundary hint so the model knows which side of the chunk may continue
into a neighbor. Fewer chunks is better for merge quality, so chunks
are kept as large as context and timeouts allow. The chunk response is
a single doc-synthesis JSON whose span-type fields (`tables`,
`narrative_responses`) carry `continues_from_previous_chunk` /
`continues_to_next_chunk` flags. Per-chunk JSONs are merged by
`scripts/merge.py`, which dedups items with stable fingerprints and
stitches fragments across chunk boundaries.

**Pass 2 (document-level synthesis):** The merged JSON is fed back to
the VLM as text (no images) to fill doc-level metadata —
`one_sentence_summary`, issue notes, cross-chunk observations. Pass 2
never rewrites per-chunk extractions.

Cross-page table/narrative linking is done programmatically from the
continuation flags, not by the LLM.

The extraction prompt lives in `scripts/doc_prompt.py` and is saved in
the output JSON for reproducibility.

Two parallel notebooks ship with the repo:

| Use case | Notebook |
|----------|----------|
| Grant admin (award notices, budgets, terms, proposals) | `notebooks/test_extraction_pipeline.ipynb` |
| Library / archival (books, manuscripts, sheet music, maps, multilingual) | `notebooks/library_extraction_pipeline.ipynb` |

Both use the same chunking + merging + pass-2 architecture; only the
per-chunk prompts and merged schema differ
(stakeholders/tables/narratives vs. bibliographic/body_text/marginalia).

### Per-page pipeline (Streamlit + batch)

`app.py` (Streamlit) and `scripts/batch_extract.py` process one page at
a time through `scripts/ocr_server.py`'s `/extract/pdf` and
`/extract/image` endpoints. Output is a flat per-page JSON list. Faster
for high-throughput runs, simpler to host, but it does not stitch tables
or narratives across page boundaries.

---

## RunAI Deployment

Production deployment uses 2–4 RunAI workloads, all driven through the
RunAI web UI (no CLI tools required):

| Workload | Type | What it does | GPU | Port |
|----------|------|-------------|-----|------|
| **`ocr-setup`** | Workspace | Notebook environment — iterate on prompts, run the chunk-based pipeline end-to-end | 0 (remote) or 0.25 (local) | 8888 |
| **`qwen3--vl--32b--instruct-awq`** | Inference | Shared Qwen3-VL-32B-Instruct-AWQ endpoint that both the notebook and the batch/Streamlit path call | 0.75 | 80 (Knative) |
| **`ocr-extract`** | Inference | *(optional)* CPU-only FastAPI server for per-page extraction via HTTP | 0 | 8090 |
| **`ocr-app`** | Workspace | *(optional)* Streamlit UI over `ocr-extract` for PoC demos | 0 | 8501 |
| **`ocr-batch`** | Workspace | *(optional)* CPU workspace that runs `batch_extract.py` against the vLLM endpoint | 0 | — |

### Where the model runs

The workspace defaults to **remote mode** — it calls the shared vLLM
endpoint (`qwen3--vl--32b--instruct-awq.runai-<project>.svc.cluster.local`)
over HTTP. No GPU is requested on the workspace itself, so it starts in
seconds and multiple users share one vLLM instance via continuous
batching.

For offline experimentation you can flip `VLM_MODE = "local"` in the
notebook — it loads the model directly with `transformers`. Local mode
needs a GPU fraction on the workspace (25% for AWQ, 75% for bf16).

```
  Remote mode (default):              Local mode (offline):

  +---------------+                   +-----------------------+
  | ocr-setup     |   HTTP            | ocr-setup             |
  | (CPU only)    |------->           | model loaded in proc  |
  +---------------+        |          | (GPU fraction)        |
  +---------------+        |          +-----------------------+
  | ocr-batch     |--------+
  | (CPU only)    |        |
  +---------------+        v
                   +-----------------+
                   | vLLM shared     |
                   | Qwen3-VL-32B    |
                   | AWQ (GPU 0.75)  |
                   +-----------------+
```

### Deployment steps

Follow these docs in order:

0. **[Setup Storage](docs/setup-storage.md)** — Data Sources for input/output (NFS or PVC depending on whether your data lives on a network share) and confirm the Qwen3-VL-32B model is on the cluster-wide `shared-models` Data Volume.
1. **[Setup & Test Workspace](docs/setup-workspace.md)** — Experiment with the chunk-based pipeline in a notebook, iterate on prompts/formats, optionally test Streamlit locally.
2. **[Deploy Streamlit App](docs/deploy-streamlit.md)** *(optional)* — Deploy the per-page Streamlit UI as its own workload for a persistent demo.
3. **[Deploy vLLM Server](docs/deploy-vllm.md)** *(optional)* — Stand up a dedicated Qwen3-VL-32B-Instruct-AWQ endpoint in your own project (the default setup assumes a shared endpoint already exists).
4. **[Batch Processing](docs/batch-processing.md)** *(optional)* — Per-page `batch_extract.py` workspace with `--resume` for bulk runs.

Additional: **[Troubleshooting](docs/troubleshooting.md)**.

### PoC path (5 sample docs)

0. Confirm the Qwen3-VL-32B model is on `shared-models` (Step 0); inputs go on the workspace's inline volume via Jupyter drag-drop
1. Setup workspace (Step 1) — upload docs, run the notebook, optionally launch Streamlit from inside the workspace
2. *(optional)* Deploy Streamlit as its own workload (Step 2)

### Production path (10K+ docs/month)

0. Setup storage (Step 0) — Data Source for `/data/documents` (NFS or PVC) + `ocr-extracted` PVC Data Source + model on `shared-models`
1. Setup workspace (Step 1) — verify the chunk pipeline on a handful of real docs
2. Make sure `qwen3--vl--32b--instruct-awq` is up (Step 3 if you need your own)
3. Run `ocr-batch` (Step 4) with `--resume` for incremental intake

---

## Output Formats

The server and batch script expose these formats via `--format` / the
Streamlit sidebar. Notebook pipelines use the richer doc-synthesis
schema in `scripts/doc_prompt.py`.

| Format | Use case | Output |
|--------|----------|--------|
| `award` | Grant award notices, NOAs, subaward agreements | JSON: PI, award #, amounts, dates, F&A rate |
| `budget` | Budget pages, financial summaries | JSON: categories, line items, costs |
| `terms` | Award terms, policies, compliance docs | JSON: sections, regulatory citations |
| `library` | Books, manuscripts, sheet music, maps, archival scans | JSON: bibliographic metadata, body text, marginalia, stamps |
| `table` | Any tabular data | Markdown tables |
| `key_values` | Forms, labeled fields | Flat JSON key-value pairs |
| `markdown` | General text + tables | Markdown |
| `json` | General text, logical structure | JSON |
| `text` | Plain text | Raw text |

## Key Files

```
ocr_app/
├── app.py                          # Streamlit UI (interactive PoC)
├── scripts/
│   ├── ocr_server.py               # FastAPI extraction server (per-page)
│   ├── batch_extract.py            # Per-page batch CLI
│   ├── chunk_extract.py            # Chunk planning + message builders (notebook pipeline)
│   ├── doc_prompt.py               # Shared doc-synthesis prompt
│   ├── merge.py                    # Dedup + continuation-flag stitching across chunks
│   └── qa_audit.py                 # Coverage report: missing pages, thin content, truncation
├── notebooks/
│   ├── test_extraction_pipeline.ipynb      # Grant-admin chunked pipeline
│   └── library_extraction_pipeline.ipynb   # Library/archival chunked pipeline
├── tests/
│   └── test_merge.py               # Unit tests for merge/dedup/stitching
├── docs/                           # Per-step deployment guides (linked above)
│   ├── setup-storage.md            #   Data Sources (input/output) + model on shared-models
│   ├── deploy-vllm.md              #   vLLM server (GPU)
│   ├── deploy-streamlit.md         #   Streamlit UI + extraction server
│   ├── setup-workspace.md          #   Setup & test workspace
│   ├── batch-processing.md         #   Per-page batch runs
│   └── troubleshooting.md          #   Common issues
├── requirements_server.txt         # Server deps (no GPU)
├── requirements_ui.txt             # Streamlit UI deps
└── .env.example                    # Environment variable template
```
