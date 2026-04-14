# Setup & Test Workspace (`ocr-setup`)

> **Step 1** in the [deployment guide](README.md).

## What this workspace does

`ocr-setup` is your **experimentation workspace** — this is where you
iterate on the extraction pipeline before deploying anything else:

1. Load the VLM directly with transformers (no vLLM server needed)
2. Upload sample documents
3. Walk through the test notebook cell by cell — render pages, send to
   VLM via 3-page sliding window, inspect JSON output
4. Experiment with different prompts (grant admin or library/archival)
5. Run batch extraction with two-pass pipeline (per-page + doc-level synthesis)
6. Test the Streamlit app from this workspace (optional)

The notebook loads the model directly on the workspace GPU — no separate
vLLM inference deployment needed at this stage. You're working directly
with the pipeline code so you can see and tweak everything.

A **test notebook** is included at
`/tmp/RunAI_apps/ocr_app/notebooks/test_extraction_pipeline.ipynb` —
this is the recommended starting point.

Once you're satisfied with the output, move on to:
- **Step 2** (Streamlit app) if you want a polished demo UI
- **Step 3** (deploy vLLM) for a persistent inference endpoint
- **Step 4** (batch processing) for production runs

---

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

> Only needed if you want to test the Streamlit UI from this workspace
> (section 10 of the notebook). Can be added later by editing the workspace.

## Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | See below |
| **Working directory** | *(leave empty)* |

### Arguments (copy-paste)

```
-c "pip install --no-cache-dir transformers huggingface_hub accelerate httpx pymupdf Pillow fastapi uvicorn python-multipart streamlit python-dotenv qwen-vl-utils matplotlib bitsandbytes; curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp; mv /tmp/RunAI_apps-main /tmp/RunAI_apps 2>/dev/null; ln -sf /tmp/RunAI_apps /ocr/repo; jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME} --ServerApp.token='' --ServerApp.allow_origin='*' --notebook-dir=/ocr"
```

> **`--ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME}`** is
> required so Jupyter's URL matches RunAI's proxy path. Without this
> you get 404 errors. `--notebook-dir=/ocr` opens Jupyter in the
> persistent volume where your docs and notebooks live.

**Environment variables:**

| Name | Value |
|------|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `HF_HUB_CACHE` | `/models/.cache/huggingface` |
| `HF_HUB_OFFLINE` | `1` |
| `STREAMLIT_BASE_PATH` | `/${RUNAI_PROJECT}/${RUNAI_JOB_NAME}/url-1` |

> `HF_*` vars point the model cache to the shared PVC. `STREAMLIT_BASE_PATH`
> is needed if you test Streamlit from this workspace (section 10 of the
> notebook). It requires a Custom URL tool on port 8501 — see below.

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled — set to `25%` of device (AWQ 4-bit model needs ~20 GB VRAM) |

> **Why GPU?** The setup workspace runs the full pipeline locally —
> including the VLM for model inference. The AWQ 4-bit quantized model
> (`QuantTrio/Qwen3-VL-32B-Instruct-AWQ`) uses ~20 GB. For the full
> bf16 model, increase to 75-85%.

## Data & storage

Two storage items:

**1. Data Volume** — shared models PVC so vLLM can load model weights:

Click **+ Data Volume**:

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |

**2. Volume** — persistent local storage for your docs, notebooks, and output:

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

---

## Access the workspace

Once the job reaches `Running` status, click the workspace name in the
RunAI UI → click the **Jupyter** tool link. This opens Jupyter Lab in
your browser.

## Using the workspace

1. **Upload sample docs** to `/ocr/` using Jupyter's file upload button
2. **Open the test notebook** at `repo/ocr_app/notebooks/test_extraction_pipeline.ipynb`
3. **Work through it cell by cell** — everything runs from the notebook:

| Step | What it does |
|------|-------------|
| 1 | Checks GPU and shared models PVC |
| 2 | Loads Qwen3-VL-32B-AWQ (4-bit, ~20 GB) with transformers |
| 3 | Lists uploaded docs, you pick one |
| 4 | Renders all pages as images for VLM |
| 5 | Runs extraction on a single page (with sliding window context from adjacent pages) |
| 6 | Displays the JSON output |
| 7 | Alternative prompts — grant admin (default) or library/archival |
| 8 | **Pass 1:** Batch extracts all pages via 3-page sliding window, saves per-page JSON with continuation flags |
| 8b | **Pass 2:** Document-level synthesis — adds title, type, creator, summary, cross-page notes |
| 9 | Compare VLM vs Gemini extractions |
| 10 | Launches extraction server + Streamlit app for interactive testing |
| 11 | Cleanup — stops all processes |

> **Streamlit test (step 10)** requires:
> 1. A **Custom URL tool on port 8501** in the workspace config
> 2. Env var `STREAMLIT_BASE_PATH` set to the tool's URL path (check
>    the RunAI UI — click the tool link, usually `/<project>/<job-name>/url-1`)
