# Deploying OCR Document Extraction on RunAI

Production deployment for extracting structured JSON from institutional
documents (grant awards, budgets, terms & conditions, archival scans).
All pages are rendered as images and sent to a Vision Language Model
(Qwen3-VL-32B-Instruct-AWQ) for structured extraction.

## Why this architecture?

Traditional OCR pipelines (Tesseract + regex) are brittle — they break on
layout changes and need per-document-type rules. This pipeline uses a VLM
to understand document structure visually:

- Every page is rendered as an image so the VLM sees layout, tables,
  signatures, watermarks, and annotations
- PDF hyperlinks are pulled from the PDF metadata and passed alongside
  the images as extra context
- The notebook pipeline splits long documents into **overlapping chunks**
  (default: 20 pages per chunk, 1-page overlap) and sends each chunk in
  one VLM call — fewer chunks is better for merge quality, so we keep
  chunks as large as context and timeouts allow
- Per-chunk **continuation flags** on tables and narratives mark items
  that run off the first/last page of the chunk
- Cross-chunk stitching is done **programmatically** by
  `scripts/merge.py` (fingerprint dedup + flag-driven stitch), not by the
  LLM
- **Pass 2** feeds the merged JSON back to the VLM as text (no images)
  to fill document-level fields (`one_sentence_summary`,
  `potential_issues`). Pass 2 never rewrites per-chunk content
- Two notebook variants ship with the repo:
  - `test_extraction_pipeline.ipynb` — grant administration schema
    (stakeholders, addresses, tables, narratives with citations)
  - `library_extraction_pipeline.ipynb` — library/archival schema
    (bibliographic metadata, body_text, marginalia, stamps, musical notation)
- The per-chunk prompt lives in `scripts/doc_prompt.py` and is saved in
  the output for reproducibility

A simpler **per-page** path is also available through `ocr_server.py`
(the FastAPI server behind the Streamlit app) and `batch_extract.py`.
That path processes pages individually — faster and easier to host, but
it does not stitch items across page boundaries.

| Workload | Type | What it does | GPU | Port |
|----------|------|-------------|-----|------|
| **`ocr-setup`** | Workspace | Notebook environment — iterate on prompts, run the chunk-based pipeline end-to-end | 0 (remote) or 0.25 (local) | 8888 |
| **`qwen3--vl--32b--instruct-awq`** | Inference | Shared Qwen3-VL-32B-Instruct-AWQ endpoint that both the notebook and the batch/Streamlit path call | 0.35 | 80 (Knative) |
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
                   | AWQ (GPU 0.35)  |
                   +-----------------+
```

---

## Deployment Guide

Follow these docs in order:

0. **[Setup Model Volumes](setup-data-volumes.md)** — [Admin only] Download the model(s) of interest to the shared PVC. Ask your admin if you need an additional model.
1. **[Setup & Test Workspace](setup-workspace.md)** — Experiment with the chunk-based pipeline in a notebook, iterate on prompts/formats, optionally test Streamlit locally.

Less tested but optional future paths:

2. **[Deploy Streamlit App](deploy-streamlit.md)** *(optional)* — Deploy the per-page Streamlit UI as its own workload for a persistent demo.
3. **[Deploy vLLM Server](deploy-vllm.md)** *(optional)* — Stand up a dedicated Qwen3-VL-32B-Instruct-AWQ endpoint in your own project (the default setup assumes a shared endpoint already exists).
4. **[Batch Processing](batch-processing.md)** *(optional)* — Per-page `batch_extract.py` workspace with `--resume` for bulk runs.

### PoC path (5 sample docs)

0. Download model to shared PVC (Step 0)
1. Setup workspace (Step 1) — upload docs, run the notebook, optionally launch Streamlit from inside the workspace
2. *(optional)* Deploy Streamlit as its own workload (Step 2)

### Production path (10K+ docs/month)

0. Setup data volumes (Step 0) — PVCs for input/output
1. Setup workspace (Step 1) — verify the chunk pipeline on a handful of real docs
3. Make sure `qwen3--vl--32b--instruct-awq` is up (Step 3 if you need your own)
4. Run `ocr-batch` (Step 4) with `--resume` for incremental intake

---

## Additional Docs

- **[Troubleshooting](troubleshooting.md)** — Common errors and fixes
