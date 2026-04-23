# Document Extraction

Structured data extraction from grant award notices, budgets, terms &
conditions, archival scans, and other institutional documents. Produces
JSON for downstream systematic analysis.

All pages (digital PDFs, scans, TIFFs) are rendered as images and sent
to a Vision Language Model (Qwen3-VL-32B-Instruct-AWQ) for structured
extraction. This ensures the VLM sees layout, tables, signatures,
watermarks, and annotations — not just raw text.

## Chunk-based, two-pass pipeline (notebooks)

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

## RunAI Deployment

Full deployment guide: **[docs/README.md](docs/README.md)**

Follow these docs in order:

0. [Setup Data Volumes](docs/setup-data-volumes.md) — download model to shared PVC, create output volume
1. [Setup & Test Workspace](docs/setup-workspace.md) — run the notebook pipeline, iterate on prompts/schemas
2. [Deploy vLLM Server](docs/deploy-vllm.md) — persistent Qwen3-VL-32B-Instruct-AWQ inference endpoint

Additional: [Troubleshooting](docs/troubleshooting.md)

## Key Files

```
ocr_app/
├── scripts/
│   ├── chunk_extract.py            # Chunk planning + message builders
│   ├── doc_prompt.py               # Shared doc-synthesis prompt
│   ├── merge.py                    # Dedup + continuation-flag stitching across chunks
│   └── qa_audit.py                 # Coverage report: missing pages, thin content, truncation
├── notebooks/
│   ├── test_extraction_pipeline.ipynb      # Grant-admin chunked pipeline
│   └── library_extraction_pipeline.ipynb   # Library/archival chunked pipeline
├── tests/
│   └── test_merge.py               # Unit tests for merge/dedup/stitching
├── docs/                           # RunAI deployment guides
│   ├── README.md                   #   Overview + deployment order
│   ├── setup-data-volumes.md       #   PVC + model download
│   ├── deploy-vllm.md              #   vLLM server (GPU)
│   ├── setup-workspace.md          #   Setup & test workspace
│   └── troubleshooting.md          #   Common issues
└── .env.example                    # Environment variable template
```
