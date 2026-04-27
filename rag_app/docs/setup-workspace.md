# Setup Workspace (`wattbot-setup`)

> **Step 1 of 5** in the [deployment guide](README.md). Assumes the
> cluster's `shared-models` data volume is available — see the
> [prerequisite note in the guide](README.md#prerequisite-shared-models-data-volume).

## What this workspace does

`wattbot-setup` is a **one-time workspace** that prepares everything the
three production services (vLLM, embedding server, Streamlit app) need
before they can run:

1. **Clones the repo** and installs Python dependencies
2. **Builds the vector index** (~130 MB SQLite DB) from source PDFs
3. **Verifies the full RAG pipeline** end-to-end (embedding → retrieval → LLM → answer)

Once setup is complete, you **stop the workspace** — it's not needed at
runtime. You only restart it to rebuild the index (e.g. after adding new
documents) or to debug issues.

## What this workspace does NOT do

- **Does not download large model weights.** All model weights (Qwen LLM,
  Jina V4, Qwen2.5-VL, etc.) must already exist on the shared models PVC
  at `/models/`. The workspace sets `HF_HUB_OFFLINE=1` to enforce this —
  if a model is missing, you get a clear error instead of a surprise
  multi-GB download. The VLM loader creates a writable cache overlay at
  `/tmp` so HuggingFace can write metadata without PVC write access or
  network access.
- **Does not serve anything in production.** The three production services
  are deployed separately as Inference workloads (steps 2-4). The VLM
  (Qwen2.5-VL-72B) is loaded only during figure extraction, not hosted
  as a long-running endpoint.

To add or update models on the shared PVC, use the `update-shared-models`
workspace instead — see
[Managing Models](managing-models.md#adding-or-updating-models-on-the-admins-shared-pvc).

---

## Create a Project PVC (PPVC) for Shared Data

The vector index (`wattbot_jinav4.db`, ~130 MB) is built once and read by
every inference job. Instead of copying it into each workspace, create a
**Project PVC (PPVC)** — Run:ai's mechanism for sharing storage across
workloads in the same project.

### What goes on the PPVC

| Directory | Contents | Written by | Read by |
|-----------|----------|------------|---------|
| `embeddings/` | `wattbot_jinav4.db` (~130 MB) | wattbot-setup (Step 0) | wattbot-app, notebooks, benchmarks |
| `corpus/` | Parsed JSON documents | wattbot-setup (Step 0) | Rebuild only |
| `pdfs/` | Downloaded source PDFs | wattbot-setup (Step 0) | Rebuild only |

### Create the Data Volume

In the RunAI UI (v2.23+):

1. Go to **Data & Storage** > **Data Volumes** > **New Data Volume**
2. Configure the data origin:
   - **Scope:** Select your project scope
     (e.g. `runai/doit-ai-cluster/default/<your-project>`)
   - **PVC name:** `wattbot-data` *(enter a new name — this creates
     a new PVC for the data volume)*
3. Set the data volume identity:
   - **Data volume name:** `wattbot-data`
   - **Description:** "Shared vector index, corpus, and PDFs for WattBot RAG"
4. Set scopes:
   - Share with the project so all workloads in the project can mount it
5. Create the Data Volume

> **Note:** The Data Volume wraps an underlying PVC. You don't need to
> create the PVC separately — the Data Volume wizard creates it for you.
> If your cluster requires a specific storage class or access mode,
> check with your admin.

### Mount path convention

All workloads mount this PPVC at **`/wattbot-data`**:

```
/wattbot-data/                    ← PPVC mount point
├── embeddings/
│   └── wattbot_jinav4.db            # vector index
├── corpus/                           # parsed JSON docs
└── pdfs/                             # cached source PDFs
```

When attaching the Data Volume to a workload in the RunAI UI, set:
- **Data volume:** `wattbot-data`
- **Mount path:** `/wattbot-data`
- **Access:** Read-write for `wattbot-setup`, read-only for inference jobs

---

## Step 0: Prepare the Workspace (one-time setup)

> **Prerequisite:** Model weights (Qwen 7B, Jina V4, etc.) must already
> be on the shared models PVC at `/models/.cache/huggingface/`. The
> admin-provisioned `shared-models` volume on the DoIT AI cluster
> already includes these. If something's missing, see
> [Managing Models](managing-models.md#adding-or-updating-models-on-the-admins-shared-pvc)
> to add it via the `update-shared-models` workspace, or
> [Provision Your Own Shared Models PVC](setup-shared-models.md) if you
> want write control.

### Cluster storage layout

| Path | Type | Access | Size | Purpose |
|------|------|--------|------|---------|
| `/models/` | Cluster `shared-models` data volume (or [your own PVC](setup-shared-models.md)) | **Read-only** | varies | Model weights (Qwen, Jina V4, etc.) |
| `/wattbot-data/` | **Project PVC** | **RW** (setup) / **RO** (inference) | 1 GB | Vector index, corpus, PDFs — shared across all jobs |
| `/home/jovyan/work/` | Personal workspace | Read-write | 30 GB | Git repo, Python deps, cache |

### 0a. Create a Workspace

In the RunAI UI:

1. Go to **Workloads** > **New Workload** > **Workspace**
2. Set:
   - **Name:** `wattbot-setup`
   - **Image:** `nvcr.io/nvidia/pytorch:25.02-py3`
   - **GPU:** `1.0` (PyTorch + JinaV4 model need most of a GPU's memory)
   - **Data Volumes:**
     - The cluster's `shared-models` data volume → mount at `/models` (read-only) — or your own PVC name if you provisioned one
     - `wattbot-data` → mount at `/wattbot-data` (**read-write**)
   - **Environment variables:**

     | Key | Value | Purpose |
     |-----|-------|---------|
     | `HF_HOME` | `/models/.cache/huggingface` | Read models from shared PVC |
     | `HF_HUB_CACHE` | `/models/.cache/huggingface` | Same — HF cache root |
     | `TRANSFORMERS_CACHE` | `/models/.cache/huggingface` | Same — transformers compat |
     | `HF_HUB_OFFLINE` | `1` | **Prevent any model downloads** |

     > **Why `HF_HUB_OFFLINE=1`?** This ensures the workspace never
     > silently downloads models. If a model is missing from the shared
     > PVC, you'll get a clear error instead of a multi-GB surprise
     > download to local storage. All models should be provisioned via
     > the `update-shared-models` workspace — see
     > [Managing Models](managing-models.md#adding-or-updating-models-on-the-admins-shared-pvc).

3. Create the Workspace and wait for it to start
4. Click **Connect** > open the **terminal** (JupyterLab or shell)

### 0b. Verify GPU and check shared models

```bash
# Verify GPU is available
nvidia-smi --query-gpu=index,name,memory.total,memory.free \
           --format=csv,noheader

# Confirm model weights are on the shared PVC
ls /models/.cache/huggingface/ | grep models--
# Should list: models--jinaai--jina-embeddings-v4,
#              models--Qwen--Qwen2.5-7B-Instruct,
#              models--Qwen--Qwen2.5-VL-72B-Instruct (if using VLM verification), etc.

# Verify Jina V4 has adapters (required for embedding)
ls /models/.cache/huggingface/models--jinaai--jina-embeddings-v4/snapshots/*/adapters/
# Should list: adapter_config.json, adapter_model.safetensors

# Verify VLM weights (only needed if using vlm_verify=True)
ls /models/.cache/huggingface/models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/*/config.json 2>/dev/null \
    && echo "VLM weights: OK" || echo "VLM weights: MISSING (optional — needed for figure verification)"

# Verify HF_HUB_OFFLINE is set (prevents accidental downloads)
[ "$HF_HUB_OFFLINE" = "1" ] && echo "OK: offline mode" || echo "WARNING: HF_HUB_OFFLINE not set!"
```

> **If models are missing:** Do NOT download them here. Instead, use the
> `update-shared-models` workspace in the `shared-models` project — see
> [Managing Models](managing-models.md#adding-or-updating-models-on-the-admins-shared-pvc).

### 0c. Set up cache directories

Model weights are read from the shared PVC at `/models/.cache/huggingface/`.
`HF_HUB_OFFLINE=1` (set in step 0a) prevents any model downloads — if a
model is missing, you'll get an error instead of a silent download.

Cache directories for pip, uv, and temp files go on your personal workspace:

```bash
# Create cache directories on writable storage
mkdir -p /home/jovyan/work/.cache/pip
mkdir -p /home/jovyan/work/.cache/uv
mkdir -p /home/jovyan/work/tmp

# Set environment variables (add to ~/.bashrc for persistence)
export TMPDIR=/home/jovyan/work/tmp
export UV_CACHE_DIR=/home/jovyan/work/.cache/uv
export PIP_CACHE_DIR=/home/jovyan/work/.cache/pip

# These should already be set from workspace env vars (step 0a),
# but verify:
echo "HF_HOME=$HF_HOME"           # should be /models/.cache/huggingface
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"  # should be 1
```

> **Note:** Small metadata files (tokenizer caches, config parsing) may
> be written to `/tmp` or the container's local filesystem — that's fine.
> The key rule is: **no model weight downloads**. All weights come from
> the shared PVC.

### 0d. Clone the repo and install dependencies

```bash
cd /home/jovyan/work

# If re-running setup, delete the old clone first (PPVC data is safe):
# rm -rf RunAI_apps

git clone https://github.com/qualiaMachine/RunAI_apps.git
cd RunAI_apps

# Switch to a specific branch if needed (default is master)
# git checkout <branch-name>

# Install uv (fast Python package installer, ~10-100x faster than pip)
pip install uv

# Create and activate a virtual environment
# (uses whatever Python the NGC image ships — currently 3.12)
uv venv
source .venv/bin/activate
python --version   # verify venv is active

# Install vendored packages (order matters: KohakuVault before KohakuRAG)
uv pip install -e rag_app/vendor/KohakuVault
uv pip install -e rag_app/vendor/KohakuRAG

# Install remaining dependencies
uv pip install -r local_requirements.txt

# Smoke test — verify imports work
python -c "import kohakuvault, kohakurag; print('Imports OK')"

# Register a named Jupyter kernel so you can select this venv in notebooks
python -m ipykernel install --user \
  --name wattbot \
  --display-name "wattbot"
```

> **Note:** Always `source .venv/bin/activate` before running any
> subsequent steps (index build, pipeline test, etc.). This keeps
> dependencies isolated from the container's system Python. In
> JupyterLab, select the **"wattbot"** kernel to use this environment
> in notebooks.

### 0e. Build the vector index (writes to PPVC)

The index build writes directly to the PPVC so all workloads can access
it without copies. We symlink `data/embeddings/` into the PPVC mount so
the build scripts' relative paths still work.

The build has three phases: (1) text embeddings from parsed PDFs,
(2) figure extraction with optional VLM verification, and
(3) multimodal image embeddings from extracted figures. JinaV4 embeds
images directly into the same vector space as text.

#### VLM figure verification (optional but recommended)

During figure extraction (`wattbot_store_images.py`), you can enable
VLM-based verification to filter out non-figures (logos, equations,
decorative elements) and generate rich descriptions. This uses a local
Qwen2.5-VL-72B model loaded from the shared PVC — no API endpoint needed.

**To enable:** set `vlm_verify = True` and `vlm_provider = "hf_local"`
in the index config, or pass them as overrides. The VLM model is loaded
once, processes all figures, and is unloaded when the script finishes.

> **Note:** VLM verification requires `Qwen/Qwen2.5-VL-72B-Instruct` on
> the shared models PVC. The admin-provisioned `shared-models` volume
> already includes it. If yours doesn't, add it via
> [Managing Models](managing-models.md#adding-or-updating-models-on-the-admins-shared-pvc)
> or, if you provisioned your own PVC, follow the
> [download step](setup-shared-models.md#step-3-download-models). The
> VLM loader automatically creates a writable cache overlay at `/tmp`
> for HF metadata — `HF_HUB_OFFLINE=1` stays set.

```bash
cd /home/jovyan/work/RunAI_apps

# Create directories on the PPVC
mkdir -p /wattbot-data/embeddings
mkdir -p /wattbot-data/corpus
mkdir -p /wattbot-data/pdfs

# Symlink repo data dirs to the PPVC (so kogine writes to shared storage)
rm -rf data/embeddings data/corpus data/pdfs
ln -s /wattbot-data/embeddings data/embeddings
ln -s /wattbot-data/corpus     data/corpus
ln -s /wattbot-data/pdfs       data/pdfs

# Check if index already exists on the PPVC
if [ -f /wattbot-data/embeddings/wattbot_jinav4.db ]; then
    echo "Index already exists: $(du -h /wattbot-data/embeddings/wattbot_jinav4.db | cut -f1)"
else
    echo "Building vector index (takes a few minutes)..."
    cd rag_app/vendor/KohakuRAG

    # Phase 1: Text index (fetch PDFs, parse to JSON, embed sentences)
    kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py

    # Phase 2: Figure extraction with VLM verification
    # The VLM loads Qwen2.5-VL-72B from /models/, verifies each crop is a
    # real figure, and generates rich descriptions for better retrieval.
    # To skip VLM verification, remove the vlm_verify/vlm_provider overrides.
    kogine run scripts/wattbot_store_images.py --config configs/jinav4/store_images_vlm.py

    # Phase 3: Rebuild text index (picks up new figure nodes) + image index
    kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
    kogine run scripts/wattbot_build_image_index.py --config configs/jinav4/image_index.py

    cd ../..
fi

# Verify the index was created
ls -lh /wattbot-data/embeddings/wattbot_jinav4.db
# Should be ~100-130 MB
```

### 0f. Test the full pipeline in a notebook

Before splitting into 3 separate jobs, verify the entire RAG pipeline
works end-to-end in the workspace. The embedding model is already loaded
from the index build, so this is a quick check. Open a **JupyterLab
notebook** and select the **wattbot** kernel we registered in Step 0d
(or run as a Python script with the venv activated) and test:

```python
import os, sys

REPO = "/home/jovyan/work/RunAI_apps"
os.chdir(REPO)
sys.path.insert(0, f"{REPO}/rag_app/vendor/KohakuRAG/src")
# HF_HOME and HF_HUB_OFFLINE are already set via workspace env vars (step 0a).
# Models load from /models/.cache/huggingface/ — no downloads allowed.

from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaV4EmbeddingModel
from kohakurag.llm import HuggingFaceLocalChatModel

# 1. Load embedding model (already cached from index build)
embedder = JinaV4EmbeddingModel()
print("Embedding model loaded")

# 2. Load vector index from the PPVC
# NOTE: do NOT pass dimensions= here. If the path is wrong, we want a loud
# error instead of silently creating a new empty DB.
DB = "/wattbot-data/embeddings/wattbot_jinav4.db"
store = KVaultNodeStore(DB, table_prefix="wattbot_jv4")
print(f"Vector index loaded: {len(store._vectors)} chunks")

# 3. Load LLM from shared cache (7B, not 72B!)
chat = HuggingFaceLocalChatModel(
    model="Qwen/Qwen2.5-7B-Instruct",
)
print("LLM loaded")

# 4. Run full pipeline
pipeline = RAGPipeline(embedder=embedder, store=store, chat_model=chat)
answer = await pipeline.answer("How much energy to train an LLM (ballpark)?")
print(f"\nAnswer: {answer['response']}")
print(f"\nTop snippets:")
for s in answer["snippets"][:3]:
    print(f"  - {s.document_title} ({s.node_id})")
```

If this works, you know the models, index, and code are all wired up
correctly. Any issues here are much easier to debug than across 3
separate inference jobs.

### 0g. Test the Streamlit app

Before tearing down the Workspace, launch the full Streamlit app to verify
everything works end-to-end — models, vector DB, and the UI itself. This
catches issues that are much easier to debug here than across 3 separate
Inference jobs.

```bash
cd /home/jovyan/work/RunAI_apps
source .venv/bin/activate

streamlit run rag_app/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
```

Access the app through the Workspace's URL in the RunAI UI (click
**Connect** > look for the proxied port). The app runs in **local mode**
by default — it loads the LLM and embedding model directly on the
Workspace GPU and uses the vector DB at `data/embeddings/` (symlinked to
the PPVC).

Try a question like *"How much energy to train an LLM?"* and verify you
get an answer with citations. Once you're satisfied, `Ctrl+C` to stop
the app.

> **Tip:** This uses the same `app.py` that runs in production — the only
> difference is that in production it runs in `remote` mode (talking to
> vLLM and the embedding server over HTTP), while here it loads models
> directly. If the app works here, the only thing that can go wrong in
> production is network connectivity between the 3 jobs.

### 0h. Check HuggingFace token (for gated models)

If you plan to use gated models (Llama 3, Gemma 2, etc.), you need an
HF token. This is **not needed** for Qwen or Jina V4:

```bash
# Check if token is set
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is set"
elif [ -f ~/.cache/huggingface/token ]; then
    echo "Found cached token"
else
    echo "WARNING: No HF token found."
    echo "To fix: export HF_TOKEN='hf_your_token_here'"
fi
```

### 0i. Updating the Corpus (adding new papers)

To add new papers and rebuild the index, restart the `wattbot-setup`
workspace and run the following:

```bash
cd /home/jovyan/work

# Remove old clone (symlinks in data/ point to PPVC, so PDFs are safe)
rm -rf RunAI_apps
git clone https://github.com/qualiaMachine/RunAI_apps.git
cd RunAI_apps

# Switch to a specific branch if needed (default is master)
# git checkout <branch-name>

source .venv/bin/activate 2>/dev/null || true

# Reinstall vendored packages (venv may already exist from prior setup)
uv pip install -e rag_app/vendor/KohakuVault
uv pip install -e rag_app/vendor/KohakuRAG
uv pip install -r local_requirements.txt

# Clear old corpus and index so the indexer re-downloads all PDFs
rm -f /wattbot-data/corpus/*.json
rm -f /wattbot-data/embeddings/wattbot_jinav4.db

# Ensure symlinks point to PPVC (so builds write to shared storage)
rm -rf data/embeddings data/corpus data/pdfs
ln -s /wattbot-data/embeddings data/embeddings
ln -s /wattbot-data/corpus     data/corpus
ln -s /wattbot-data/pdfs       data/pdfs

# Build text index (downloads PDFs, parses to JSON, creates embeddings)
cd rag_app/vendor/KohakuRAG
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py

# Extract figures with VLM verification (use store_images.py config for no VLM)
kogine run scripts/wattbot_store_images.py --config configs/jinav4/store_images_vlm.py

# Rebuild text index (picks up figure nodes) + build image index
kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
kogine run scripts/wattbot_build_image_index.py --config configs/jinav4/image_index.py
cd ../..

# Verify
echo "PDFs:    $(ls /wattbot-data/pdfs/*.pdf 2>/dev/null | wc -l)"
echo "JSONs:   $(ls /wattbot-data/corpus/*.json 2>/dev/null | wc -l)"
echo "DB size: $(du -h /wattbot-data/embeddings/wattbot_jinav4.db | cut -f1)"
```

The index is written directly to the PPVC via symlinks — no manual
copy needed. After the build completes, **restart `wattbot-app`** so it
picks up the new database.

**How to add papers to the corpus:**

1. **Using the helper script** (recommended):
   - Edit `rag_app/scripts/add_papers.py` — add entries to the `NEW_PAPERS` list
   - Run `python rag_app/scripts/add_papers.py` to preview, then `--apply` to write
   - Commit and push, then rebuild on `wattbot-setup` as above

2. **Manually editing `data/metadata.csv`:**
   - Add a row: `id,type,title,year,citation,url`
   - The `url` must be a direct PDF link (e.g. `https://arxiv.org/pdf/XXXX.XXXXX`)
   - The indexer auto-downloads the PDF, parses it to JSON, and embeds it

> **Tip:** You don't need to delete PDFs that already exist in
> `/wattbot-data/pdfs/` — the indexer skips re-downloading cached PDFs.
> Only delete `corpus/*.json` and the `.db` file to force a full rebuild.

### 0j. Stop the Workspace

Once the pipeline test passes, you can **stop the Workspace** from the RunAI UI
to free its GPU. The vector index persists on the PPVC (`wattbot-data`) —
Inference jobs mount it directly and don't depend on the Workspace running.

**When to re-run this step:**
- When you add/remove/update documents in `data/corpus/`
- When you change embedding settings (dimension, model)

**NOT needed when:**
- Changing the LLM model (Qwen → Llama, etc.)
- Changing retrieval settings (top_k)
- Restarting Inference jobs
