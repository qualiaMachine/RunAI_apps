# Deploying OCR Document Extraction on RunAI

Production deployment for extracting structured JSON from institutional
documents (grant awards, budgets, terms & conditions, archival scans).
All pages are rendered as images and sent to a Vision Language Model
(Qwen3-VL-32B-Instruct-AWQ) for structured extraction.

## Why this architecture?

Traditional OCR pipelines (Tesseract + regex) are brittle — they break on
layout changes and need per-document-type rules. This pipeline uses a VLM
to understand document structure visually:

- Each page is rendered as an image and processed via a **3-page sliding
  window** — the VLM sees adjacent pages as context, extracts the center
  page only
- The VLM sees layout, tables, signatures, watermarks, and annotations
- PDF hyperlinks are extracted from metadata and passed as additional context
- Per-page **continuation flags** detect content spanning page boundaries
  (split tables, mid-sentence breaks) — accurate because the VLM can see
  neighboring pages
- Cross-page linking is done **programmatically** from continuation flags
- Produces structured JSON (grant admin or library/archival schema)
- **Pass 2** feeds all per-page JSONs back to the VLM (text-only, no images)
  for document-level metadata: title, type, creator, summary, cross-page notes.
  Never modifies per-page results
- Extraction prompt is saved in output for reproducibility

| Workload | Type | What it does | GPU | Port |
|----------|------|-------------|-----|------|
| **`ocr-setup`** | Workspace | Notebook environment — load model locally, iterate on prompts | 0.25 | 8888 |
| **`qwen3--vl--32b--instruct-awq`** | Inference | *(production)* Serves Qwen3-VL-32B-Instruct-AWQ via vLLM | 0.25 | 8000 |
| **`ocr-app`** | Workspace | *(optional)* Streamlit UI for PoC demos | 0 | 8501 |

### Two modes of running the VLM

**Notebook mode (setup workspace):** The `ocr-setup` workspace loads the
model directly with transformers — no vLLM server needed. This is best
for iterating on prompts, testing on sample docs, and batch runs.
The workspace needs its own GPU (25% with AWQ).

**Production mode (vLLM server):** For large-scale batch processing,
deploy the vLLM inference job and point batch scripts at it over HTTP.
The batch workspace is CPU-only; all GPU work is in vLLM.

```
  Notebook mode:                    Production mode:

  +----------------+                +---------------------+
  | ocr-setup      |                |   vLLM Server       |
  | Model loaded   |                |   Qwen3-VL-32B-AWQ  |
  | locally (GPU)  |                |   Port 8000 (GPU)   |
  +----------------+                +---------^-----------+
                                              | HTTP
                                +-------------+-------------+
                                |             |             |
                         +------+------+ +----+----+ +------+------+
                         | ocr-batch   | | extract | | Streamlit   |
                         | (CPU only)  | | server  | | UI          |
                         +-------------+ +---------+ +-------------+
```

---

## Deployment Guide

Follow these docs in order:
0. **[Setup Model Volumes](setup-data-volumes.md)** — [Admin only] Download model(s) of interest to shared PVC. Ask admin if you need additional models.
1. **[Setup & Test Workspace](setup-workspace.md)** — Experiment with pipeline in notebook, iterate on prompts/formats, test Streamlit locally

Less tested but optional future paths
2. **[Deploy Streamlit App](deploy-streamlit.md)** *(optional)* — Deploy as its own workload for persistent demo UI
3. **[Deploy vLLM Server](deploy-vllm.md)** — *(optional)* Persistent Qwen3-VL-32B-Instruct inference endpoint

### PoC path (5 sample docs)

0. Download model to shared PVC (Step 0)
1. Setup workspace (Step 1) — upload docs, run test notebook, launch Streamlit from workspace
2. Optionally deploy Streamlit as its own workload (Step 2)

### Production path (10K+ docs/month)

0. Setup data volumes (Step 0) — PVCs for input/output
1. Setup workspace (Step 1) — verify pipeline with notebook
3. Deploy `qwen3--vl--32b--instruct-awq` as persistent endpoint (Step 3)
4. Deploy `ocr-batch` (Step 4) — batch workspace with `--resume`

---

## Additional Docs

- **[Troubleshooting](troubleshooting.md)** — Common errors and fixes
