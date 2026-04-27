# Document Extraction

Structured data extraction from grant award notices, budgets, terms &
conditions, archival scans, and other institutional documents. Produces
JSON for downstream systematic analysis.

All pages (digital PDFs, scans, TIFFs) are rendered as images and sent
to a Vision Language Model (Qwen3-VL-32B-Instruct-AWQ) for structured
extraction. This ensures the VLM sees layout, tables, signatures,
watermarks, and annotations — not just raw text.

## Pipeline

The documented and tested path is the **chunk-based, two-pass notebook
pipeline**. Entry points: `notebooks/test_extraction_pipeline.ipynb`
(grant administration) and `notebooks/library_extraction_pipeline.ipynb`
(library / archival).

**Pass 1 (chunk extraction):** The document is split into overlapping
page chunks (default: 20 pages per chunk, 1-page overlap). Each chunk
is sent to the VLM as a single call with every page as an image, plus a
boundary hint so the model knows which side of the chunk may continue
into a neighbor. The chunk response is a single doc-synthesis JSON
whose span-type fields (`tables`, `narrative_responses`) carry
`continues_from_previous_chunk` / `continues_to_next_chunk` flags. Per-chunk
JSONs are merged by `scripts/merge.py`, which dedups items with stable
fingerprints and stitches fragments across chunk boundaries.

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

## RunAI Deployment

Full deployment guide: **[docs/README.md](docs/README.md)**

Follow these docs in order:

0. [Setup Storage](docs/setup-storage.md) — Confirm the Qwen3-VL-32B model is on the cluster-wide `shared-models` Data Volume; optionally set up an input documents Data Source if your data is on a network share or won't fit in the workspace inline volume
1. [Setup & Test Workspace](docs/setup-workspace.md) — experiment with the pipeline in a notebook, iterate on prompts/formats
2. [Deploy vLLM Server](docs/deploy-vllm.md) *(optional)* — stand up your own Qwen3-VL-32B-Instruct-AWQ inference endpoint if no shared one exists

Additional: [Troubleshooting](docs/troubleshooting.md)

### PoC (5 sample docs)

0. Confirm the Qwen3-VL-32B model is on `shared-models` (Step 0); inputs go on the workspace's inline volume via Jupyter drag-drop
1. Setup workspace (Step 1) — upload docs, run the notebook end-to-end

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
├── docs/                           # RunAI deployment guides
│   ├── README.md                   #   Overview + deployment order
│   ├── setup-storage.md            #   Input Data Source + model on shared-models
│   ├── setup-workspace.md          #   Setup & test workspace
│   ├── deploy-vllm.md              #   vLLM server (GPU)
│   └── troubleshooting.md          #   Common issues
├── requirements_server.txt         # Server deps (no GPU)
├── requirements_ui.txt             # Streamlit UI deps
└── .env.example                    # Environment variable template
```
