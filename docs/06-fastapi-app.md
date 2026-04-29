# 06 — Build a FastAPI App on the Cluster

> **Optional follow-on** to the [New User Guide](../README.md#new-user-guide).
> Assumes you've worked through [02 First Workspace](02-first-workspace.md)
> and [03 Share a Model as a vLLM Endpoint](03-share-as-endpoint.md), and
> have a Qwen2.5-7B (or similar) Inference workload running in your project.

So far the guide has covered notebooks (02) and model-serving endpoints
(03). This doc covers the layer most apps actually want to expose to
*users*: a small custom HTTP API written in FastAPI, deployed as its own
RunAI workload, that calls one or more shared vLLM endpoints over
internal cluster DNS and serves the result to a frontend.

Same pattern the WattBot Streamlit app uses — see
[`rag_app/docs/deploy-streamlit.md`](../rag_app/docs/deploy-streamlit.md) —
just with FastAPI in place of Streamlit, so you can plug it into a
website, a chatbot, or another service instead of a built-in UI.

By the end of this doc you'll have:
- A `fastapi-example` Workspace running the
  [`scripts/fastapi_example.py`](../scripts/fastapi_example.py) reference
  app on port 8000
- A browser-reachable URL (over campus VPN) you can `curl` and that an
  internal site can call from JavaScript
- A clear picture of what changes if you later need the same endpoint
  reachable from off-VPN (short version: talk to your cluster admin
  about the firewall — it's not self-serve)

## Workspace or Inference?

Two RunAI workload types can host a FastAPI app, and they have very
different reachability properties:

| | **Workspace** *(this doc)* | **Inference** |
|---|---|---|
| URL shape | `https://<cluster-host>/<project>/<name>/proxy/<port>/` | `https://<knative-route>` (cluster-dependent) |
| Reachable from a browser | Yes, via the RunAI portal proxy (VPN required) | Only if the cluster admin has wired Knative ingress through the firewall |
| Autoscaling | No — single pod, manual Stop/Start | Yes — `min=0` scales to zero when idle |
| GPU needed | No (CPU-only is fine for an API gateway) | Optional |
| Use it for | Internal tools, VPN-only sites, anything you want to test today | Public-facing endpoints, autoscaling APIs, anything called from outside the campus VPN |

**Start with a Workspace.** It's the same shape as the Streamlit app
(`rag_app/docs/deploy-streamlit.md`) and works out of the box on the
current pilot cluster. Externalizing the same app as an Inference
workload is a follow-on step covered at the bottom of this doc.

## What you'll deploy

[`scripts/fastapi_example.py`](../scripts/fastapi_example.py) — a
~80-line reference app with three endpoints:

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/health` | Always returns 200; proves the proxy URL is wired up |
| GET | `/info` | Echoes the configured `VLLM_BASE_URL`, `VLLM_MODEL`, etc. — useful for verifying env vars made it into the pod |
| POST | `/chat` | Forwards `{question, system?, max_tokens?}` to the configured vLLM workload as a chat completion and returns the answer |

It's deliberately minimal — no retrieval, no auth, no rate limiting.
Replace `/chat` with whatever your real app needs.

## Step A. Make sure you have a vLLM endpoint to call

If you don't already have one running, follow
[03 Share a Model as a vLLM Endpoint](03-share-as-endpoint.md) Steps A
and B (deploy `qwen-Qwen2.5--7B--Instruct`, leave `first-workspace`
running with 0 GPU). Note the endpoint URL — it'll be:

```
http://qwen-Qwen2.5--7B--Instruct.runai-<your-project>.svc.cluster.local/v1
```

The FastAPI app will call this from inside the cluster, so `http://`
(not `https://`) and the internal `.svc.cluster.local` hostname are
correct.

## Step B. Deploy the FastAPI Workspace

In the RunAI UI: **Workloads** > **+ NEW WORKLOAD** > **Workspace**

### Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project |
| **Workspace name** | `fastapi-example` |

### Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **Image pull** | Pull the image only if it's not already present on the host (recommended) |

> Same base image the Streamlit app uses. PyTorch is overkill for a
> CPU-only FastAPI gateway, but reusing it means the node has the layer
> cached from other workloads — image pull is instant.

### Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | *(see below — one long string)* |

```
-c "pip install uv && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp && mv /tmp/RunAI_apps-main /tmp/RunAI_apps && cd /tmp/RunAI_apps && uv pip install --system fastapi uvicorn openai pydantic && uvicorn scripts.fastapi_example:app --host 0.0.0.0 --port 8000 --root-path \$FASTAPI_BASE_PATH"
```

What this does, piece by piece:

| Chunk | Why |
|-------|-----|
| `pip install uv` | Fast Python installer; the Streamlit doc explains why |
| `rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED` | PEP 668 workaround so `uv pip install --system` works in the NGC image |
| `curl ... \| tar xz` + `mv` | Pull the repo at `main` into `/tmp/RunAI_apps`. Replace `main` with your branch name if you're iterating on a fork |
| `uv pip install ... fastapi uvicorn openai pydantic` | Just the four packages the example needs — PyTorch and CUDA are already in the image |
| `uvicorn scripts.fastapi_example:app --host 0.0.0.0` | Bind on all interfaces so the RunAI proxy can reach the app |
| `--root-path \$FASTAPI_BASE_PATH` | Tells FastAPI/Uvicorn that requests arrive prefixed with the proxy subpath, so OpenAPI docs and any `request.url_for(...)` calls resolve correctly. The `\$` escapes the `$` so the shell expands it inside the container, not in the RunAI form |

### Environment variables

| Name | Value | Why |
|------|-------|-----|
| `VLLM_BASE_URL` | `http://qwen-Qwen2.5--7B--Instruct.runai-<your-project>.svc.cluster.local/v1` | The vLLM workload from doc 03. Replace `<your-project>` with your RunAI project name |
| `VLLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Whatever model ID the vLLM workload was started with |
| `FASTAPI_BASE_PATH` | `/<project>/fastapi-example/proxy/8000` | The proxy subpath the RunAI portal serves this app at. Replace `<project>` with your project name. Same shape as `STREAMLIT_BASE_PATH` in the Streamlit doc |
| `ALLOWED_ORIGIN` | `*` *(for testing — see [Step E](#step-e-thinking-about-going-public) before tightening for production)* | CORS allow-list |

### Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `0` (CPU-only — the model lives in the vLLM workload) |
| **CPU request** | *(default)* |
| **CPU memory request** | *(default)* |

### Data & storage

None. The example has no persistent state — your real app likely will,
in which case attach a project PVC the same way `04-storage.md`
describes.

### Connection (Tool)

This is the part that makes the proxy URL work. Without it, the URL
returns 404.

| Field | Value |
|-------|-------|
| **Tool type** | Custom URL |
| **Name** | `fastapi` |
| **Container port** | `8000` |

### Click **CREATE WORKSPACE**.

Boots in 30–60s on a warm node — the four pip installs are the only
real work.

## Step C. Hit it from your laptop

Once the workload is **Running**, click **Connect** > **fastapi**, or
build the URL by hand:

```
https://<cluster-host>/<project>/fastapi-example/proxy/8000/
```

Then, on a machine connected to the campus VPN:

```bash
curl https://<cluster-host>/<project>/fastapi-example/proxy/8000/health
# {"status":"ok"}

curl https://<cluster-host>/<project>/fastapi-example/proxy/8000/info
# {"service":"fastapi-example","vllm_base_url":"http://qwen-...","vllm_configured":true,...}

curl -X POST https://<cluster-host>/<project>/fastapi-example/proxy/8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "In one sentence, what is retrieval-augmented generation?"}'
# {"answer":"...","model":"Qwen/Qwen2.5-7B-Instruct"}
```

If `/health` works but `/chat` returns 503: check `/info` first.
`vllm_configured: false` means the env vars didn't reach the pod;
`vllm_configured: true` but a 5xx on `/chat` means the vLLM workload
itself isn't ready (check its **Pods** tab).

OpenAPI docs are at `/docs` on the same proxy URL.

## Step D. Call it from a website (still VPN-only)

Any browser on the campus VPN can hit the proxy URL, so a static site
or single-page app served from anywhere can use it as long as **its
viewers are also on the VPN**. The example has CORS open
(`ALLOWED_ORIGIN=*`) for ease of testing — fine for a private internal
wiki, not OK for anything public.

A quick sanity check from the browser console of any HTTPS page on
the VPN:

```js
const base = "https://<cluster-host>/<project>/fastapi-example/proxy/8000";
fetch(`${base}/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "What is RAG?" }),
}).then(r => r.json()).then(console.log);
```

For an actual internal lab wiki, point your frontend at that base URL,
add a tiny chat component, and you have an in-house "ChatGPT for our
docs." Plug retrieval in by importing `KohakuRAG` (see
[`rag_app/`](../rag_app/README.md)) inside `/chat` and embedding+searching
your corpus before forwarding to the LLM.

## Step E. Thinking about going public

A Workspace proxy URL is **only** reachable through the RunAI portal,
which lives behind the campus VPN. To expose the same FastAPI to
off-VPN users, two things have to change:

1. **Workload type → Inference.** Inference workloads on this cluster
   get a Knative HTTPS route that *can* be reached from outside the
   data center. The cluster admin has validated this path (the OCR
   `qwen3--vl--32b--instruct-awq` shared endpoint and the DAPIR PoC
   both use it), but it's a per-workload firewall conversation, not a
   self-serve checkbox. Email Chris/Mike with the workload name and
   the source IPs/networks that need access.
2. **Auth.** The `Auth: Internal` setting on Inference workloads
   blocks external ingress entirely, which is the right default. Once
   you do open it up, the FastAPI itself needs to enforce auth — at
   minimum a static API key checked in a `Depends(...)`, ideally OAuth
   / institutional SSO if real users will hit it. The example app
   ships with **no auth**.

Other things worth getting right before exposing publicly:

- **Tighten CORS.** Replace `ALLOWED_ORIGIN=*` with the exact origin
  of your frontend (e.g. `https://wiki.your-lab.wisc.edu`).
- **Rate-limit the LLM-touching endpoints.** A single bot can drain a
  shared GPU faster than it can drain your patience. `slowapi` is the
  usual FastAPI add-on.
- **Don't echo internal hostnames.** The reference `/info` endpoint
  returns the cluster-internal vLLM URL, which is harmless on VPN but
  leaks topology if exposed externally. Drop or gate that endpoint
  before going public.
- **Watch what the model can be talked into saying.** A public LLM
  endpoint over UW data is a content-moderation surface; treat it
  like one.

The Inference deployment recipe itself (image, args, ports) is the
same as Workspace, with the workload type swapped and an authenticated
ingress configured by the cluster admin. Once a use case actually goes
through that flow we'll add a deployable walkthrough back into this
doc.

## Step F. Tear down

Stop and delete `fastapi-example` from **Workloads**, then tear down
the Qwen workload from doc 03 if you spun one up just for this. Per
the same reasoning as in 03, we don't leave personal endpoints running
on the 2-GPU pilot.

## Where to take it next

- Add retrieval: import `kohakurag` from `rag_app/vendor/`, point it at
  your corpus on a project PVC, call it inside `/chat` before the LLM
  request. The Streamlit app is a working reference.
- Add auth: a 10-line `Depends(api_key_header)` is enough to turn
  `*`-CORS into "named callers only."
- Add structured outputs: vLLM supports JSON-schema-constrained
  decoding via the OpenAI client's `response_format` argument — handy
  if downstream code is parsing the answer.
- Promote to Inference + external ingress once the use case is real
  (see [Step E](#step-e-thinking-about-going-public)).
