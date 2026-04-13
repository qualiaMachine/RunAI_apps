# Deploy the Streamlit App

The Streamlit UI connects to the vLLM and embedding services via internal
cluster DNS. Unlike the GPU services, the Streamlit app is deployed as a
**Workspace** — not an Inference workload — because Workspaces provide
browser-accessible URLs via the RunAI proxy, while Inference workloads
on most clusters only expose internal Knative routes that aren't
reachable from a browser.

> **Why Workspace instead of Inference?** Inference workloads use Knative
> serving, which requires wildcard DNS and ingress configuration that many
> clusters don't have for external browser access. Workspaces get a
> reliable proxy URL (`/proxy/<port>/`) that works out of the box. Since
> the Streamlit app needs no GPU and no autoscaling, a Workspace is the
> right fit.

In the RunAI UI: **Workloads** > **New Workload** > **Workspace**

## Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project (e.g. `jupyter-endemann01`) |
| **Workspace name** | `wattbot-app` |

## Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **Image pull** | Pull the image only if it's not already present on the host (recommended) |
| **Image pull secret** | *(leave empty — public NGC image)* |

> **Why NGC PyTorch?** This image has `git` and a `python` alias, making
> the startup command simpler. It needs a one-time PEP 668 workaround
> (see Runtime settings below). The `vllm/vllm-openai` image also works
> but requires `curl` tarball download instead of `git clone` and uses
> `python3` not `python`.

## Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | `-c "pip install uv && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && git clone -b main --depth 1 https://github.com/qualiaMachine/RunAI_apps.git /tmp/RunAI_apps && cd /tmp/RunAI_apps && mkdir -p /tmp/vectordb && cp /wattbot-data/embeddings/wattbot_jinav4.db /tmp/vectordb/ && ln -sf /wattbot-data/corpus rag_app/data/corpus && uv pip install --system streamlit openai httpx 'numpy<2' python-dotenv && uv pip install --system rag_app/vendor/KohakuVault rag_app/vendor/KohakuRAG && python -m streamlit run rag_app/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --server.baseUrlPath=\$STREAMLIT_BASE_PATH"` |
| **Working directory** | *(leave empty)* |

> **What the command does:**
> 1. Installs `uv` (fast Python package installer)
> 2. Removes the PEP 668 `EXTERNALLY-MANAGED` marker (safe in ephemeral containers)
> 3. Clones the repo to ephemeral `/tmp` (not on PVC)
> 4. **Copies** the vector DB to a writable `/tmp/vectordb/` directory (the PPVC
>    is read-only, and KVaultNodeStore writes metadata on open, so symlinks
>    to the PPVC won't work — the app auto-discovers the DB at `/tmp/vectordb/`)
> 5. Symlinks corpus data dir to the shared PPVC
> 6. Installs Python deps + vendored KohakuVault/KohakuRAG
> 7. Starts Streamlit with proxy-compatible settings
>
> **Why copy instead of symlink for the DB?** KVaultNodeStore writes metadata
> and runs `auto_pack` when opening the database. If the PPVC is mounted
> read-only (or the volume doesn't support writes from this pod), the
> store silently creates a new empty DB instead — resulting in **zero
> vectors** and no local retrieval. The app auto-discovers the copied DB
> at `/tmp/vectordb/wattbot_jinav4.db`.
>
> **Using a different branch?** Replace `main` in the `git clone` command
> with the desired branch name.

**Environment variables:**

| Key | Value |
|-----|-------|
| `RAG_MODE` | `remote` |
| `VLLM_BASE_URL` | `http://wattbot-chat.runai-<project>.svc.cluster.local/v1` |
| `EMBEDDING_SERVICE_URL` | `http://wattbot-embedding.runai-<project>.svc.cluster.local` |
| `RERANKER_SERVICE_URL` | `http://wattbot-reranker.runai-<project>.svc.cluster.local` |
| `STREAMLIT_BASE_PATH` | `/<project>/<workspace-name>/proxy/8501` |
| `VECTOR_DB_PATH` | `/tmp/vectordb/wattbot_jinav4.db` *(optional — the app auto-discovers this path, but setting it explicitly is recommended)* |

> **Knative DNS for Inference workloads:** RunAI Inference workloads use
> Knative serving, which routes through **port 80** (not the container
> port) and requires the **fully-qualified service name** as the hostname.
> Short names like `wattbot-chat:8000` will not work — envoy returns 404
> because it can't match the route without the namespace in the Host header.
>
> The format is: `<workload-name>.runai-<project>.svc.cluster.local`
>
> Example (project `jupyter-endemann01`, workloads `wattbot-chat` and `wattbot-embedding`):
> - `VLLM_BASE_URL=http://wattbot-chat.runai-jupyter-endemann01.svc.cluster.local/v1`
> - `EMBEDDING_SERVICE_URL=http://wattbot-embedding.runai-jupyter-endemann01.svc.cluster.local`
>
> **No port number** — Knative maps port 80 → container port automatically.
>
> **Model auto-detection:** The app automatically discovers the model name
> from the vLLM `/v1/models` endpoint — no `VLLM_MODEL` env var needed.

> **`STREAMLIT_BASE_PATH`** tells Streamlit the proxy subpath so
> WebSocket connections route correctly. Replace `<project>` with your
> RunAI project (e.g. `jupyter-endemann01`) and `<workspace-name>` with
> the workspace name you chose above (e.g. `wattbot-app`).
>
> Example: `/jupyter-endemann01/wattbot-app/proxy/8501`

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `0` (no GPU — Streamlit is CPU-only) |
| **CPU request** | *(leave default)* |
| **CPU memory request** | *(leave default)* |

## Data & storage

| Data volume name | Container path |
|------------------|----------------|
| `wattbot-data` | `/wattbot-data` |

> **Note:** The Streamlit app does not need the `shared-models` volume —
> it doesn't load any ML models directly. It connects to vLLM and the
> embedding server via HTTP.

## Connection (Tool)

The Workspace needs a **connection method** so the RunAI proxy routes
traffic to port 8501. Without this, the proxy URL returns 404.

| Field | Value |
|-------|-------|
| **Tool type** | Custom URL |
| **Name** | `streamlit` (or any name) |
| **Container port** | `8501` |

> **Blank page on first load?** After the workspace starts, the proxy URL
> may initially show a blank white page titled "Streamlit". This means the
> HTML loaded but the WebSocket hasn't connected yet. Wait 10-20 seconds
> and refresh — the app should render fully. This only happens on the very
> first load after startup.

## General

| Field | Value |
|-------|-------|
| **Priority** | `very-high` (or as appropriate) |

## Expected startup time

First deploy takes **2-4 minutes**:
- **Image pull** (~0s): If the vLLM image is already cached on the node
  (from previous deploys), this is instant.
- **Dependency install** (~30-60s): `uv` installs Streamlit, OpenAI
  client, and a few small packages. KohakuVault/KohakuRAG are installed
  from the vendored source.
- **Streamlit startup** (~5s): The app starts and begins listening on
  port 8501.

## Access the app

Once the Workspace is running, access it via the RunAI proxy URL:

```
https://<cluster-host>/<project>/<workspace-name>/proxy/8501/
```

For example:
```
https://deepthought.doit.wisc.edu/jupyter-endemann01/wattbot-app/proxy/8501/
```

You can also click **Connect** on the Workspace in the RunAI UI to
get the base URL, then append `/proxy/8501/`.
