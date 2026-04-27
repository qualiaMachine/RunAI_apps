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

## Pipeline

The documented and tested path is the **chunk-based, two-pass notebook
pipeline**. Entry points: `notebooks/test_extraction_pipeline.ipynb`
(grant administration) and `notebooks/library_extraction_pipeline.ipynb`
(library / archival).

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

> A simpler per-page path (`app.py`, `scripts/batch_extract.py`,
> `scripts/ocr_server.py`) exists in the tree for local experimentation
> but is not currently deployment-tested on RunAI.

---

## RunAI Deployment

All steps use the **RunAI web UI only** — no CLI tools required.

| Workload | Type | What it does | GPU | Port |
|----------|------|-------------|-----|------|
| **`ocr-setup`** | Workspace | Notebook environment — iterate on prompts, run the chunk-based pipeline end-to-end | 0 (remote) or 0.25 (local) | 8888 |
| **`qwen3--vl--32b--instruct-awq`** | Inference | Shared Qwen3-VL-32B-Instruct-AWQ endpoint that the notebook calls | 0.75 | 80 (Knative) |

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
                           |          +-----------------------+
                           v
                   +-----------------+
                   | vLLM shared     |
                   | Qwen3-VL-32B    |
                   | AWQ (GPU 0.75)  |
                   +-----------------+
```

### Deployment steps

Follow these docs in order:

0. **[Setup Storage](docs/setup-storage.md)** — Confirm the Qwen3-VL-32B model is on the cluster-wide `shared-models` Data Volume; optionally create a Data Source for input documents if you need more than the workspace inline volume can hold.
1. **[Setup & Test Workspace](docs/setup-workspace.md)** — Experiment with the chunk-based pipeline in a notebook, iterate on prompts/formats.
2. **[Deploy vLLM Server](docs/deploy-vllm.md)** *(optional)* — Stand up a dedicated Qwen3-VL-32B-Instruct-AWQ endpoint in your own project (the default setup assumes a shared endpoint already exists).

Additional: **[Troubleshooting](docs/troubleshooting.md)**

### PoC path (5 sample docs)

0. Confirm the Qwen3-VL-32B model is on `shared-models` (Step 0); skip input/output Data Sources — use Path A in setup-storage
1. Setup workspace (Step 1) — drag PDFs into Jupyter, run the notebook

---

## Output Schemas

The notebook pipeline produces a single doc-synthesis JSON per document
(see `scripts/doc_prompt.py` for the grant-admin schema; the library
notebook uses an inline bibliographic schema). The schema covers
stakeholders, addresses, tables with continuation flags, narrative
responses, and pass-2 fields like `one_sentence_summary` and
`potential_issues`.

## Key Files

```
ocr_app/
├── notebooks/
│   ├── test_extraction_pipeline.ipynb      # Grant-admin chunked pipeline
│   └── library_extraction_pipeline.ipynb   # Library/archival chunked pipeline
├── scripts/
│   ├── chunk_extract.py            # Chunk planning + message builders
│   ├── doc_prompt.py               # Shared doc-synthesis prompt
│   ├── merge.py                    # Dedup + continuation-flag stitching across chunks
│   ├── qa_audit.py                 # Coverage report: missing pages, thin content, truncation
│   ├── ocr_server.py               # FastAPI per-page server (not deployment-tested)
│   ├── batch_extract.py            # Per-page batch CLI (not deployment-tested)
│   └── ...
├── app.py                          # Streamlit per-page UI (not deployment-tested)
├── tests/
│   └── test_merge.py               # Unit tests for merge/dedup/stitching
├── docs/                           # Per-step deployment guides (linked above)
│   ├── setup-storage.md            #   Input Data Source + model on shared-models
│   ├── setup-workspace.md          #   Setup & test workspace
│   ├── deploy-vllm.md              #   vLLM server (GPU)
│   └── troubleshooting.md          #   Common issues
├── requirements_server.txt         # Server deps (no GPU)
├── requirements_ui.txt             # Streamlit UI deps
└── .env.example                    # Environment variable template
```
