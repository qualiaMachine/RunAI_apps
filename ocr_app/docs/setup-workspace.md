# Setup & Test Workspace (`ocr-setup`)

> **Step 1** in the [deployment guide](README.md).

## What this workspace does

`ocr-setup` is your **experimentation workspace** — this is where you
iterate on the extraction pipeline before deploying anything else:

1. Connect to the shared vLLM endpoint (no local model loading)
2. Upload sample documents
3. Walk through a test notebook cell by cell — render pages, send to
   VLM via 3-page sliding window, inspect JSON output
4. Run batch extraction with two-pass pipeline (per-page + doc-level synthesis)
5. Test the Streamlit app from this workspace (optional)

By default the workspace runs in **remote mode** — it calls the shared
vLLM endpoint at `qwen3--vl--32b--instruct-awq.runai-shared-models` via
HTTP. No GPU needed on the workspace itself. Multiple users share the
same vLLM instance via continuous batching.

If the shared endpoint is down or you want to experiment offline, set
`VLM_MODE = "local"` in the notebook to load the model directly — this
requires GPU on the workspace (25% fraction with AWQ, 75% with bf16).

## Pick the right notebook for your use case

Two parallel notebooks ship with the repo — same pipeline architecture,
different schemas tuned to each document type:

| If you're processing... | Use this notebook |
|-------------------------|-------------------|
| **Grant administration docs** — award notices, budgets, RSP terms, research proposals, forms with stakeholders/addresses | `repo/ocr_app/notebooks/test_extraction_pipeline.ipynb` |
| **Library / archival materials** — books, manuscripts, sheet music, newspapers, maps, multilingual scans, materials needing bibliographic metadata | `repo/ocr_app/notebooks/library_extraction_pipeline.ipynb` |

The two notebooks share:
- Remote vs local VLM mode toggle
- 3-page sliding window extraction
- Two-pass pipeline (per-page + document-level synthesis)
- Code-driven cross-page content merging
- Audit file (`*_extracted.jsonl`) + consolidated final file (`*_final.json`)

They differ in:
- Per-page prompt and JSON schema (grant admin vs library/archival)
- Assembly function (stakeholders/addresses/narratives vs bibliographic/body_text/marginalia/stamps)
- Synthesis prompt (award metadata vs catalog metadata)

---

## Create the workspace

In the RunAI UI: **Workloads** > **New Workload** > **Workspace**

## Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project |
| **Workspace name** | `ocr-setup` |

## Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **Image pull** | Pull the image only if it's not already present on the host |

## Tools

Add Jupyter for browser access:

| Field | Value |
|-------|-------|
| **Tool type** | Jupyter |
| **Port** | `8888` |

*(Optional)* Add a Custom URL tool for Streamlit testing:

| Field | Value |
|-------|-------|
| **Tool type** | Custom URL |
| **Name** | `streamlit` |
| **Container port** | `8501` |

> Only needed if you want to test the Streamlit UI from this workspace.
> Can be added later by editing the workspace.

## Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | See below |
| **Working directory** | *(leave empty)* |

### Arguments (copy-paste)

```
-c "pip install --no-cache-dir httpx pymupdf Pillow streamlit python-dotenv matplotlib; curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp; mv /tmp/RunAI_apps-main /tmp/RunAI_apps 2>/dev/null; ln -sf /tmp/RunAI_apps /ocr/repo; jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME} --ServerApp.token='' --ServerApp.allow_origin='*' --notebook-dir=/ocr"
```

> **Remote mode** (default) installs a minimal set of packages — no
> `transformers` or `qwen-vl-utils` needed since the model runs on the
> vLLM server. If you plan to use `VLM_MODE = "local"`, the notebook's
> install cell will add `transformers qwen-vl-utils autoawq` on demand.
>
> **`--ServerApp.base_url`** is required so Jupyter's URL matches RunAI's
> proxy path. **`--notebook-dir=/ocr`** opens Jupyter in the persistent
> volume where your docs live.

**Environment variables:**

| Name | Value |
|------|-------|
| `STREAMLIT_BASE_PATH` | `/${RUNAI_PROJECT}/${RUNAI_JOB_NAME}/url-1` |

> `STREAMLIT_BASE_PATH` is only needed if you test Streamlit from this
> workspace. It requires a Custom URL tool on port 8501 (see Tools above).

## Compute resources

### Default: Remote mode (recommended)

| Field | Value |
|-------|-------|
| **GPU devices** | `0` |
| **CPU** | *(leave default)* |
| **Memory** | *(leave default)* |

No GPU needed — the model runs on the shared vLLM endpoint. Workspace
starts in seconds instead of waiting for model loading.

### Optional: Local mode

If you want to run the model directly in the notebook (set
`VLM_MODE = "local"`):

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled — `25%` for AWQ, `75%` for bf16 |

## Data & storage

**Persistent volume** for your docs, notebooks, and output:

Click **+ Volume**:

| Field | Value |
|-------|-------|
| **Storage class** | `local-path` |
| **Access mode** | *(leave default)* |
| **Claim size** | `1` GB (increase if processing many docs) |
| **Volume mode** | Filesystem |
| **Container path** | `/ocr` |
| **Volume persistency** | Persistent |

This gives you `/ocr` as a persistent directory — upload sample docs
here, save notebooks here, store extraction output here. Survives
workspace restarts.

> **Shared models PVC (`shared-models` → `/models`)** is only needed if
> you plan to use `VLM_MODE = "local"`. In remote mode the workspace
> never touches the model cache.

Click **Create Workspace**.

---

## Access the workspace

Once the job reaches `Running` status (under a minute in remote mode —
no model loading):

1. Click the workspace name in the RunAI UI
2. Click the **Jupyter** tool link — opens Jupyter Lab in your browser

## Using the workspace

1. **Upload sample docs** to `/ocr/` using Jupyter's file upload button
2. **Open the notebook for your use case** (see table above):
   - Grant admin: `repo/ocr_app/notebooks/test_extraction_pipeline.ipynb`
   - Library/archival: `repo/ocr_app/notebooks/library_extraction_pipeline.ipynb`
3. **Work through it cell by cell** — both notebooks follow the same flow:

| Step | What it does |
|------|-------------|
| 1 | Checks VLM endpoint connection (remote mode) or shared models PVC (local mode) |
| 2 | *(Local mode only)* Loads model directly with transformers |
| 3 | Installs Python packages (httpx/pymupdf/Pillow always; transformers/qwen-vl-utils only for local) |
| 4 | Defines helper functions: `run_vlm()`, `extract_page()` with sliding window support |
| 5 | Defines the extraction prompt + assembly function for this pipeline's schema |
| 6 | Lists uploaded docs, you pick one |
| 7 | Renders all pages as images for VLM |
| 8 | Runs extraction on a single page (with sliding window context from adjacent pages) |
| 9 | Displays the JSON output |
| 10 | Defines pass 2 synthesis prompt + helper functions |
| 11 | **Pass 1 + Pass 2:** Batch loop — extracts all pages (sliding window), saves `_extracted.jsonl`, runs synthesis, saves `_final.json` |
| 12 | *(Optional)* Compare VLM vs reference extractions |
| 13 | *(Optional)* Launches extraction server + Streamlit app for interactive testing |
| 14 | Cleanup — stops all processes |

> **Streamlit test (step 12)** requires:
> 1. A **Custom URL tool on port 8501** in the workspace config
> 2. Env var `STREAMLIT_BASE_PATH` set to the tool's URL path (check
>    the RunAI UI — click the tool link, usually `/<project>/<job-name>/url-1`)

---

## Switching between remote and local mode

The default (`VLM_MODE = "remote"`) works out of the box — no config
needed. If you want to switch to local mode:

1. **Stop the workspace**
2. **Edit the workspace** in the RunAI UI:
   - Compute: change GPU to `1` device, `25%` fraction
   - Data & storage: add `shared-models` → `/models` data volume
   - Environment variables: add `HF_HOME=/models/.cache/huggingface`,
     `HF_HUB_CACHE=/models/.cache/huggingface`, `HF_HUB_OFFLINE=1`
3. **Restart the workspace**
4. In the notebook cell 3, change `VLM_MODE = "remote"` to
   `VLM_MODE = "local"` and re-run cells 1-4

Or just keep remote mode — it's faster and cheaper for most use cases.
