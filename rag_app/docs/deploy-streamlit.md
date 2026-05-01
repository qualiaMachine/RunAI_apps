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

Same shape as the `02 First workspace` args, with two
Streamlit-specific wrinkles: a writable copy of the vector DB and a
proxy-aware Streamlit launch. Piece by piece:

| Chunk | Why it's there |
|-------|----------------|
| `-c "..."` | Tells `bash` to run the rest as a shell command, then exit. The whole arguments value is one string. |
| `pip install uv` | Pulls in `uv`, a drop-in replacement for `pip` that's 10-100x faster. The dependency installs below use `uv pip install` so cold-starts finish in seconds instead of minutes. |
| `rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED` | The NGC PyTorch image marks system Python as PEP 668 "externally managed", which makes `pip install` refuse without `--break-system-packages`. Removing the marker is safe in an ephemeral container — there's no host package manager to confuse. |
| `git clone -b main --depth 1 https://github.com/.../RunAI_apps.git /tmp/RunAI_apps` | Pull only the latest commit of `main` into ephemeral `/tmp`. NGC PyTorch ships with `git`, so we use `git clone` here instead of the `curl` tarball pattern. To deploy a different branch, swap `main` for the branch name. |
| `cd /tmp/RunAI_apps` | Subsequent `uv pip install` paths are relative to the repo root. |
| `mkdir -p /tmp/vectordb && cp /wattbot-data/embeddings/wattbot_jinav4.db /tmp/vectordb/` | **Copy** the vector DB onto ephemeral disk instead of reading it directly from the PPVC. `KVaultNodeStore` writes metadata and runs `auto_pack` when it opens the DB. If the PPVC is read-only the store silently creates a new empty DB instead — you get zero vectors and no retrieval. The app auto-discovers the copied DB at `/tmp/vectordb/wattbot_jinav4.db`. |
| `ln -sf /wattbot-data/corpus rag_app/data/corpus` | Point the corpus dir inside the cloned repo at the PPVC mount so the app reads parsed JSON docs from shared storage. Symlink is fine here (read-only is OK for the corpus). |
| `uv pip install --system streamlit openai httpx 'numpy<2' python-dotenv` | App runtime deps. `--system` installs into the container's system Python (no venv needed for an ephemeral pod). `numpy<2` pins below the 2.x ABI break that several embedded libs aren't compatible with yet. |
| `uv pip install --system rag_app/vendor/KohakuVault rag_app/vendor/KohakuRAG` | Install the two vendored libraries from their source paths. Order doesn't matter when both are passed in one `uv pip install` call. |
| `python -m streamlit run rag_app/app.py` | Start Streamlit as the long-running foreground process. |
| `--server.port=8501` | Match the **Connection (Tool)** container port further down. |
| `--server.address=0.0.0.0` | Bind to all interfaces so the RunAI proxy can reach the server from outside the pod. |
| `--server.headless=true` | Don't try to open a local browser — there's no display inside the pod. |
| `--server.enableCORS=false --server.enableXsrfProtection=false` | RunAI's proxy and Streamlit end up on different origins; without these, the browser blocks the websocket and form posts. |
| `--server.baseUrlPath=\$STREAMLIT_BASE_PATH` | Tells Streamlit its own URL prefix so static asset and websocket URLs round-trip through the proxy. The variable comes from the env-var table below — escaped with `\$` so the **outer** shell defers expansion until inside the pod. |

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
