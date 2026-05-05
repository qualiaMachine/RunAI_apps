# Setup & Test Workspace (`ocr-setup`)

> **Step 1** in the [deployment guide](../README.md).

## What this workspace does

`ocr-setup` is your **experimentation workspace** — this is where you
iterate on the extraction pipeline:

1. Connect to the shared vLLM endpoint (no local model loading)
2. Upload sample documents
3. Walk through a test notebook cell by cell — render pages, send
   overlapping page chunks to the VLM, inspect the merged JSON
4. Run the full chunk-based pipeline on every doc in `/ocr/` and a
   doc-level pass-2 synthesis on the merged result

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
- Chunk-based extraction (default 20 pages per chunk, 1-page overlap)
  via `scripts/chunk_extract.py`
- Deterministic chunk merging with continuation-flag stitching via
  `scripts/merge.py`
- Two-pass pipeline (per-chunk extraction + doc-level synthesis)
- Per-doc output folder `<stem>_chunks/` with each chunk response and a
  consolidated `<stem>_extracted.json`

They differ in:
- Per-chunk prompt and JSON schema (grant admin uses
  `scripts/doc_prompt.py`; library/archival uses an inline
  bibliographic prompt)
- Merge/assembly behavior tuned to each schema
- Pass-2 synthesis prompt (award metadata vs catalog metadata)

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

## Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | See below |
| **Working directory** | *(leave empty)* |

### Arguments (copy-paste)

```
-c "pip install --no-cache-dir httpx pymupdf Pillow python-dotenv matplotlib; curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp; mv /tmp/RunAI_apps-main /tmp/RunAI_apps 2>/dev/null; ln -sf /tmp/RunAI_apps /ocr/repo; jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME} --ServerApp.token='' --ServerApp.allow_origin='*' --notebook-dir=/ocr"
```

Same structural pattern as
[02 First workspace](../../docs/02-first-workspace.md) — pip install,
pull repo tarball, symlink into the persistent volume, then start
JupyterLab tuned for the RunAI proxy. Once you have one workspace
dialed in, save it as a Workload Template (RunAI UI > **Workload
manager** > **Templates** > **+ NEW TEMPLATE**) so future workspaces
start with this pre-filled.

What that string actually does, piece by piece:

| Chunk | Why it's there |
|-------|----------------|
| `-c "..."` | Tells `bash` to run the rest as a shell command, then exit. The whole arguments value is one string. |
| `pip install --no-cache-dir httpx pymupdf Pillow python-dotenv matplotlib` | Remote-mode dependencies: `httpx` for the vLLM HTTP client, `pymupdf` to render PDF pages to images, `Pillow` for image manipulation, `python-dotenv` for env loading, `matplotlib` for the notebook's inline plots. `--no-cache-dir` skips writing wheel caches inside the pod. No `transformers` / `qwen-vl-utils` here — the model lives on the vLLM server. If you switch to `VLM_MODE = "local"`, the notebook's install cell adds `transformers qwen-vl-utils autoawq` on demand. |
| `curl -sL https://github.com/.../main.tar.gz \| tar xz -C /tmp` | Pull the current `main` branch as a tarball and unpack it under `/tmp`. Faster and lighter than `git clone` (no `.git` history) and doesn't require git on the image. |
| `mv /tmp/RunAI_apps-main /tmp/RunAI_apps 2>/dev/null` | GitHub's tarball unpacks to `<repo>-<branch>/`. Rename to a stable path. The redirect swallows the harmless "directory already exists" error on subsequent restarts. |
| `ln -sf /tmp/RunAI_apps /ocr/repo` | Drop a symlink into the persistent `/ocr` volume so `repo/` shows up in Jupyter's file browser next to your sample documents and notebook outputs. The actual code lives in ephemeral `/tmp` and refreshes from GitHub on every restart — no stale local copy to worry about. |
| `;` | Run the next command after the previous one finishes, regardless of exit status. |
| `jupyter-lab` | Start JupyterLab as the long-running foreground process. |
| `--ip=0.0.0.0` | Bind to all interfaces so RunAI's proxy can reach the server from outside the pod. The default (`localhost`) only accepts connections from inside the container. |
| `--port=8888` | Match the port advertised in the **Tools** section above so RunAI's proxy lines up. |
| `--no-browser` | Don't try to open a local browser — there's no display inside the pod. |
| `--allow-root` | The container runs as root by default; Jupyter refuses to start as root unless you say it's fine. |
| `--ServerApp.base_url=/${RUNAI_PROJECT}/${RUNAI_JOB_NAME}` | RunAI proxies your notebook at a path like `/<project>/<workload-name>/...`. Jupyter has to know its own base path or static asset URLs and websocket reconnects break. The two env vars are auto-set by RunAI inside the pod. |
| `--ServerApp.token=''` | Disable Jupyter's own login token — RunAI's portal already authenticated you, and a token here would just block the proxy. |
| `--ServerApp.allow_origin='*'` | Allow cross-origin requests. RunAI's proxy and Jupyter end up on different origins; without this, the browser blocks the websocket. |
| `--notebook-dir=/ocr` | Open Jupyter's file browser at `/ocr` so you land directly on the persistent volume (where your sample documents and extraction output live). |

### Environment variables

In the **default remote mode** the workspace doesn't load any model
weights itself — it only talks HTTP to the shared vLLM endpoint — so
the HuggingFace cache vars from `02 First workspace` are not required.
Leave the env-var section empty.

If you plan to flip the notebook to **local mode** (`VLM_MODE =
"local"`), set the same three vars `02` uses so `transformers` reads
the cached weights from the shared PVC instead of trying to download:

| Name | Value | Why |
|------|-------|-----|
| `HF_HOME` | `/models/.cache/huggingface` | HuggingFace cache root. Default is `~/.cache/huggingface` inside the pod's ephemeral disk; pointing it at the mounted `shared-models` volume is what makes `transformers.from_pretrained(...)` find the pre-cached weights. |
| `HF_HUB_CACHE` | `/models/.cache/huggingface` | More specific override for the hub-cache path used by `huggingface_hub`. Different transformers versions respect different vars; setting both `HF_HOME` and `HF_HUB_CACHE` is belt-and-suspenders so every code path lands at the same directory. |
| `HF_HUB_OFFLINE` | `1` | Forbid network downloads. If the model isn't in the cache, you get a fast, loud error instead of a silent multi-GB download to ephemeral disk that vanishes on restart. |

These only matter when the workspace itself loads the VLM. The
[Switching between remote and local mode](#switching-between-remote-and-local-mode)
section below covers the rest of the local-mode flip (GPU, the
`shared-models` Data Volume mount).

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
   - For a stripped-down walkthrough (remote vLLM only, no Gemini
     comparison, no Streamlit launcher) use the matching
     `*_demo.ipynb` next to each production notebook.
3. **Work through it cell by cell.** Both notebooks share this flow
   (library notebook omits the Gemini comparison section):

| Section | What it does |
|---------|--------------|
| 1. Setup | Checks VLM endpoint connection (remote mode) or shared models PVC (local mode); installs runtime deps |
| 2. Load model *(local only)* | Loads the VLM in-process with transformers |
| Helpers | Defines `run_vlm()`, `_encode_image_b64()`, `extract_links()` |
| Prompts | Loads the extraction prompt — `DOC_SYNTHESIS_PROMPT` from `scripts/doc_prompt.py` for grant admin; inline library prompt for archival |
| 3. Load a sample document | Lists uploaded docs in `/ocr/`, you pick one |
| 4. Render pages | Renders every PDF page at 2x as a PIL image |
| Single-page test | *(Optional, commented by default)* Runs one chunk containing a single page to sanity-check the prompt |
| 8. Batch process all PDFs | **Pass 1**: plans chunks via `chunk_page_ranges(MAX_PAGES_PER_CHUNK, CHUNK_OVERLAP)`, extracts each chunk with boundary hints and optional pinned first-page context, writes per-chunk JSON to `<stem>_chunks/`, then merges with `merge_chunks` into `<stem>_extracted.json`. **Pass 2** (`RUN_PASS2 = True`): sends the merged JSON back to the VLM as text to fill `one_sentence_summary` and append issue notes |
| 9. Zip the output folder | Packages `/ocr/vlm_out/` into a single archive for download |
| 10. Compare VLM vs Gemini | *(Grant-admin notebook only)* Side-by-side comparison against reference extractions |

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
