# Deploy Streamlit App (`ocr-app`)

> **Step 2** *(optional)* in the [deployment guide](README.md). Comes
> after [Setup & Test Workspace](setup-workspace.md) (Step 1). You can
> also test Streamlit directly from the setup workspace first.

Browser-based UI for uploading documents and previewing extracted results.
Drag-and-drop your files, pick an output format, and see structured JSON
immediately. Good starting point for the PoC — no CLI needed.

> **How it works:** The Streamlit app talks to the extraction server
> (`ocr-extract`), which does text extraction locally and routes LLM/VLM
> requests to the vLLM server. You need both `ocr-extract` and `ocr-app`
> running.

---

## Step 2a: Deploy the extraction server (`ocr-extract`)

The Streamlit UI talks to this FastAPI server, which handles file
processing and routes to vLLM. CPU-only — no GPU needed.

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

### Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project |
| **Inference type** | **Custom** |
| **Inference name** | `ocr-extract` |

### Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `vllm/vllm-openai:latest` |
| **Image pull** | Pull the image only if it's not already present on the host |

> **Why the vLLM image for a CPU-only server?** It has Python 3 and pip
> pre-installed. We just need a Python environment — the GPU/CUDA parts
> of the image go unused.

### Serving endpoint

| Field | Value |
|-------|-------|
| **Protocol** | HTTP |
| **Container port** | `8090` |

### Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | See below |
| **Working directory** | *(leave empty)* |

#### Arguments (copy-paste)

```
-c "pip install uv && curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp && mv /tmp/RunAI_apps-main /tmp/RunAI_apps && cd /tmp/RunAI_apps && uv pip install --system fastapi uvicorn python-multipart httpx pymupdf Pillow && python3 ocr_app/scripts/ocr_server.py"
```

**Environment variables:**

| Name | Value |
|------|-------|
| `LLM_BASE_URL` | `http://qwen3--vl--32b--instruct-awq.runai-<project>.svc.cluster.local/v1` |
| `VLM_MODEL` | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` |
| `OCR_PORT` | `8090` |

### Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `0` (none — CPU only) |

### Data & storage

No data volumes needed — files are uploaded via HTTP from the Streamlit
app, not read from a PVC.

---

## Step 2b: Deploy the Streamlit UI (`ocr-app`)

In the RunAI UI: **Workloads** > **New Workload** > **Workspace**

### Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project |
| **Workspace name** | `ocr-app` |

### Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **Image pull** | Pull the image only if it's not already present on the host |

### Tools

| Field | Value |
|-------|-------|
| **Tool type** | Custom URL |
| **Tool name** | `streamlit` |
| **Port** | `8501` |

### Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | See below |
| **Working directory** | *(leave empty)* |

#### Arguments (copy-paste)

```
-c "pip install uv && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp && mv /tmp/RunAI_apps-main /tmp/RunAI_apps && cd /tmp/RunAI_apps && uv pip install --system streamlit httpx Pillow python-dotenv && python -m streamlit run ocr_app/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --server.baseUrlPath=$STREAMLIT_BASE_PATH"
```

**Environment variables:**

| Name | Value |
|------|-------|
| `OCR_SERVICE_URL` | `http://ocr-extract.runai-<project>.svc.cluster.local` |
| `STREAMLIT_BASE_PATH` | `/<project>/ocr-app/url-1` |

> **Replace `<project>`** with your actual project name. The `url-1`
> suffix is assigned by RunAI for the first Custom URL tool (after
> Jupyter). Check the tool link in the RunAI UI to confirm the exact
> path. Example: `STREAMLIT_BASE_PATH=/jupyter-endemann01/ocr-app/url-1`

### Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `0` (none) |

### Data & storage

No data volumes needed.

---

## Access the app

Once both `ocr-extract` and `ocr-app` are running, open the Streamlit
URL:

Click the workspace name in the RunAI UI and click the **streamlit**
tool link. The URL will be something like:

```
https://<cluster-host>/<project>/ocr-app/url-1/
```

> **Note:** The exact path suffix (`url-1`, `url-0`, etc.) is assigned by
> RunAI based on tool order. Always use the link from the RunAI UI rather
> than constructing the URL manually.

---

## Using the app for PoC

1. **Upload your sample docs** — drag and drop PDFs/TIFFs into the file
   uploader
2. **Pick an output format** from the sidebar:
   - `award` for grant award notices
   - `budget` for budget pages
   - `terms` for terms & conditions
   - `key_values` for general forms
   - `text` to see raw extraction
3. **Click "Extract"** — results appear per page with timing info
4. **Download results** — click the download button to save the JSON

The sidebar shows whether the server is connected and which model is
running. Each page result shows whether it used text extraction (digital)
or VLM OCR (scanned).

---

## Expected startup time

- **`ocr-extract`** — ~2-3 min (image pull + pip install + server start)
- **`ocr-app`** — ~2-3 min (image pull + pip install + Streamlit start)

Both are CPU-only, so startup is limited by image pull and pip install,
not model loading.
