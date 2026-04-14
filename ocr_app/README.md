# Document Extraction

Structured data extraction from grant award notices, budgets, terms &
conditions, archival scans, and other institutional documents. Produces
JSON for downstream systematic analysis.

All pages (digital PDFs, scans, TIFFs) are rendered as images and sent
to a Vision Language Model (Qwen3-VL-32B-Instruct-AWQ) for structured
extraction. This ensures the VLM sees layout, tables, signatures,
watermarks, and annotations — not just raw text.

## Architecture

```
  +---------------------+     +----------------------+
  |  Extraction Server  |---->|   VLM                |
  |  FastAPI            |     |   Qwen3-VL-32B       |
  |  Port 8090          |     |   Port 8000 (GPU)    |
  |                     |     |                      |
  |  PDF: render pages  |     |   All pages: image   |
  |  TIFF: send image   |     |   -> structured JSON |
  +---------------------+     +----------------------+
```

Every page is rendered as an image and sent to the VLM for extraction.
PDF hyperlinks are extracted from the metadata layer and passed as
additional context. All GPU work happens in vLLM.

## RunAI Deployment

Full deployment guide: **[docs/README.md](docs/README.md)**

Follow these docs in order:

0. [Setup Data Volumes](docs/setup-data-volumes.md) — download model to shared PVC, create output volume
1. [Setup & Test Workspace](docs/setup-workspace.md) — experiment with pipeline in notebook, iterate on prompts/formats
2. [Deploy Streamlit App](docs/deploy-streamlit.md) *(optional)* — polished demo UI, test from workspace first
3. [Deploy vLLM Server](docs/deploy-vllm.md) — persistent Qwen3-VL-32B-Instruct-AWQ inference endpoint
4. [Batch Processing](docs/batch-processing.md) — production workspace for large-scale runs

Additional: [Troubleshooting](docs/troubleshooting.md)

### PoC (5 sample docs)

0. Download model to shared PVC (Step 0)
1. Setup workspace (Step 1) — upload docs, run test notebook, launch Streamlit from workspace
2. Optionally deploy Streamlit as its own workload (Step 2)

### Production (10K+ docs/month)

0. Setup data volumes (Step 0)
1. Setup workspace (Step 1) — verify pipeline with notebook
3. Deploy vLLM as persistent endpoint (Step 3)
4. Batch processing workspace (Step 4) — `--resume` for incremental runs

## Output Formats

| Format | Use case | Output |
|--------|----------|--------|
| `award` | Grant award notices, NOAs, subaward agreements | JSON: PI, award #, amounts, dates, F&A rate |
| `budget` | Budget pages, financial summaries | JSON: categories, line items, costs |
| `terms` | Award terms, policies, compliance docs | JSON: sections, regulatory citations |
| `table` | Any tabular data | Markdown tables |
| `key_values` | Forms, labeled fields | Flat JSON key-value pairs |
| `text` | Plain text | Raw text |

## Key Files

```
ocr_app/
├── app.py                          # Streamlit UI (interactive PoC)
├── scripts/
│   ├── ocr_server.py               # FastAPI extraction server
│   └── batch_extract.py            # Batch processing script
├── notebooks/
│   └── test_extraction_pipeline.ipynb  # Step-by-step test notebook
├── deploy/
│   └── runai_jobs.yaml             # RunAI job configs
├── docs/                           # RunAI deployment guides
│   ├── README.md                   #   Overview + deployment order
│   ├── setup-data-volumes.md       #   PVC + model download
│   ├── deploy-vllm.md             #   vLLM server (GPU)
│   ├── deploy-streamlit.md         #   Streamlit UI + extraction server
│   ├── setup-workspace.md          #   Setup & test workspace
│   ├── batch-processing.md         #   Production batch runs
│   └── troubleshooting.md          #   Common issues
├── requirements_server.txt         # Server deps (no GPU)
├── requirements_ui.txt             # Streamlit UI deps
└── .env.example                    # Environment variable template
```
