# Image Generation (`image_app/`)

Text-to-image generation with **Qwen/Qwen-Image** (20B MMDiT, Apache
2.0), served as a RunAI inference workload with a minimal browser UI.
Aimed at illustrative presentation graphics — cartoon diagrams, posters,
labeled illustrations — where Qwen-Image's strong in-image text
rendering shines.

Same deployment pattern as the RAG app's embedding server: stock
`vllm/vllm-openai` image, repo pulled as a tarball at startup, a few
lightweight deps installed with `uv`, weights loaded from the shared
models PVC. vLLM itself cannot serve diffusion models, so the server is
a small FastAPI wrapper around the Diffusers pipeline
([`scripts/qwen_image_server.py`](scripts/qwen_image_server.py)).

> **Content note:** Qwen-Image is trained with filtered data but, like
> any open-weight image model, has no hard guarantee against
> inappropriate output. Keep the endpoint cluster-internal (the default
> below) and covered by the [Usage Policy](../docs/usage-policy.md). If
> you later expose it beyond trusted users, add an output-side NSFW
> classifier to the server.

## Prerequisite: model on the shared PVC

From the `update-shared-models` workspace (see
[Managing Models](../rag_app/docs/managing-models.md)):

```bash
python /models/provision_shared_models.py download Qwen/Qwen-Image
```

The full repo (transformer + Qwen2.5-VL text encoder + VAE) is
**~55–60 GB** on disk. Check `df -h /models` first.

## Deploy the inference workload

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

### Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project (e.g. `jupyter-endemann01`) |
| **Inference type** | **Custom** (not "Model: from Hugging Face") |
| **Inference name** | `qwen-image` |

### Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `vllm/vllm-openai:latest` |
| **Image pull** | Pull the image only if it's not already present on the host (recommended) |
| **Image pull secret** | *(leave empty — public Docker Hub image)* |

### Serving endpoint

| Field | Value |
|-------|-------|
| **Protocol** | HTTP |
| **Container port** | `8080` |

### Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | `-c "pip install uv && curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp && mv /tmp/RunAI_apps-main /tmp/RunAI_apps && cd /tmp/RunAI_apps && uv pip install --system fastapi uvicorn pillow 'diffusers>=0.35' 'transformers>=4.53,<5' accelerate safetensors && python3 image_app/scripts/qwen_image_server.py"` |
| **Working directory** | *(leave empty)* |

Same shape as the
[embedding server](../rag_app/docs/deploy-embedding.md): pull the repo
tarball into the vLLM image (it has `curl` but not `git`), install our
deps, run the server. `diffusers>=0.35` is the first release with
Qwen-Image pipeline support; the `transformers<5` pin avoids the 5.x
ABI break; PyTorch and CUDA come with the image.

**Environment variables:**

| Name | Value | Why |
|------|-------|-----|
| `HF_HUB_OFFLINE` | `1` | Fail loudly instead of downloading ~60 GB to ephemeral disk if the model is missing from the PVC |
| `OFFLOAD` | *(unset)* | Set to `1` only on GPUs with < 80 GB VRAM — enables CPU offload (slower, much smaller footprint) |

The server resolves the model's snapshot directory on the PVC itself
and loads from that local path, so the read-only mount needs no
writable HF cache overlay.

### Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | **Disabled — full GPU.** bf16 needs ~60 GB VRAM (20B transformer + 7B text encoder + activations at 1328px+) |
| **CPU memory request** | `32Gi` (higher if using `OFFLOAD=1`) |
| **Replica autoscaling** | Min `1`, Max `1` |

### Data & storage

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |

### Expected startup time

First deploy takes **5–10 minutes**: image pull (skipped if the vLLM
image is cached on the node), ~1 min dependency install, then ~2–4 min
loading ~60 GB of weights from the PVC into GPU memory.

## Usage

From a browser inside the cluster (or via the endpoint URL RunAI
shows), open the workload URL — the root page is a simple prompt form.

From code or `curl`:

```bash
curl -s http://qwen-image.<runai-project>.svc.cluster.local/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A friendly cartoon diagram of a GPU cluster: three server racks with smiling faces, arrows labeled JOBS flowing from a laptop, flat vector style", "aspect_ratio": "16:9"}' \
  -o diagram.png
```

Request fields: `prompt`, `negative_prompt`, `aspect_ratio` (`1:1`,
`16:9`, `9:16`, `4:3`, `3:4` — the model's supported resolutions),
`num_inference_steps` (default 50), `true_cfg_scale` (default 4.0),
`seed` (reproducibility). Response is a PNG; the
`X-Generation-Seconds` header reports render time.

Generation is serialized on the single GPU — concurrent requests
queue. Expect roughly 1–3 minutes per image at 50 steps.

### Prompting tips for diagrams

- Spell out any text you want rendered, in quotes, and keep labels
  short — text fidelity is the model's strength but degrades with long
  strings.
- Name a style: "flat vector illustration", "hand-drawn whiteboard
  sketch", "isometric cartoon".
- For structurally exact diagrams (flowcharts where arrows must connect
  the right boxes), use Mermaid/draw.io/SVG instead — diffusion models
  draw plausible-looking, not logically-correct, structure.
