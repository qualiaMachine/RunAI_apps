# Deploying OCR Document Extraction on RunAI

Production deployment for extracting structured JSON from institutional
documents (grant awards, budgets, terms & conditions, archival scans).
All pages are rendered as images and sent to a Vision Language Model
(Qwen3-VL-32B-Instruct) for structured extraction.

## Why this architecture?

Traditional OCR pipelines (Tesseract + regex) are brittle — they break on
layout changes and need per-document-type rules. This pipeline uses a VLM
to understand document structure visually:

- Every page (digital or scanned) is rendered as an image
- The VLM sees layout, tables, signatures, watermarks, and annotations
- PDF hyperlinks are extracted from metadata and passed as additional context
- Produces structured JSON matching the grant admin schema

| Workload | Type | What it does | GPU | Port |
|----------|------|-------------|-----|------|
| **`ocr-setup`** | Workspace | One-time setup — test pipeline on sample docs | 0.85 | 8888 |
| **`qwen3--vl--32b--instruct`** | Inference | *(optional)* Serves Qwen3-VL-32B-Instruct for text parsing + VLM OCR | 0.85 | 8000 |
| **`ocr-app`** | Workspace | *(optional)* Streamlit UI for PoC demos | 0 | 8501 |

### Service layout

```
                    +---------------------+
                    |   vLLM Server       |
                    |   Qwen3-VL-32B     |
                    |   Port 8000 (GPU)   |
                    +---------^-----------+
                              | HTTP (cluster DNS)
              +---------------+----------------+
              |               |                |
   +----------+---+  +-------+--------+  +----+-----------+
   | Batch Script |  | Extract Server |  | Streamlit UI   |
   | (workspace)  |  | (optional API) |  | (optional PoC) |
   | CPU only     |  | CPU only       |  | CPU only       |
   +--------------+  +----------------+  +----------------+
```

All paths talk to the same vLLM server. The batch script is the primary
tool for processing large collections. The extraction server and Streamlit
UI are optional — useful for interactive demos.

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
3. Deploy `qwen3--vl--32b--instruct` as persistent endpoint (Step 3)
4. Deploy `ocr-batch` (Step 4) — batch workspace with `--resume`

---

## Additional Docs

- **[Troubleshooting](troubleshooting.md)** — Common errors and fixes
