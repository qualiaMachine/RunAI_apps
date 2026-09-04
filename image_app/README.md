# Image Generation (`image_app/`)

Headless text-to-image endpoints for illustrative presentation graphics
— cartoon diagrams, posters, labeled illustrations. JSON in, PNG out,
served the same way as the RAG app's embedding server: stock
`vllm/vllm-openai` image (used only as a CUDA + PyTorch base — vLLM
itself cannot serve diffusion-family models), repo pulled as a tarball
at startup, lightweight deps installed with `uv`, weights loaded from
the shared models PVC.

Two servers:

| Server | Model | Params / disk | License | Status |
|--------|-------|---------------|---------|--------|
| [`scripts/hidream_server.py`](scripts/hidream_server.py) | HiDream-ai/HiDream-O1-Image-Dev-2604 | 9B / ~36 GB (F32 on disk, loaded bf16) | MIT | Current deployment |
| [`scripts/qwen_image_server.py`](scripts/qwen_image_server.py) | Qwen/Qwen-Image | 20B / ~55–60 GB | Apache 2.0 | Alternative (stronger safety filtering; see [Appendix](#appendix-qwen-image-alternative)) |

HiDream-O1-Dev-2604 ranks near the top of the Artificial Analysis
open-weights text-to-image leaderboard. It has no official headless
server — upstream ships a Flask web UI — so `hidream_server.py` wraps
their inference pipeline directly: at startup it fetches a **pinned
commit** of [their repo](https://github.com/HiDream-ai/HiDream-O1-Image)
(`dev` branch), applies their documented no-flash-attn fallback (the
stock image has no flash-attn; SDPA attention is used instead), loads
the model bf16 from the PVC, and exposes `POST /generate`.

> **Content note:** open-weight image models have no hard guarantee
> against inappropriate output, and HiDream does not document its
> training-data filtering. Keep the endpoint covered by the
> [Usage Policy](../docs/usage-policy.md); if exposure ever widens
> beyond trusted users, add an output-side NSFW classifier.

## Prerequisite: model on the shared PVC

From the `update-shared-models` workspace (see
[Managing Models](../rag_app/docs/managing-models.md)):

```bash
python /models/provision_shared_models.py download HiDream-ai/HiDream-O1-Image-Dev-2604
```

~36 GB on disk. `verify` reports missing top-level config/weights for
this repo layout — false alarm; trust the download completing.

## Deploy the inference workload

In the RunAI UI: **Workloads** > **New Workload** > **Inference**
(or save these settings as an inference template).

| Field | Value |
|-------|-------|
| **Inference type** | **Custom** (not "Model: from Hugging Face") |
| **Name** | `hidream-image` |
| **Image URL** | `vllm/vllm-openai:latest` |
| **Serving endpoint** | HTTP, container port `8000` |
| **Command** | `bash` |
| **Arguments** | *(below)* |
| **GPU** | 1 device, fraction 75% (bump to 100% if OOM at 2048×2048) |
| **CPU / Memory** | 8 cores / `32Gi` |
| **Replicas** | Min 1 / Max 1 |
| **Data & storage** | Models PVC/data volume → `/models` **plus a writable scratch PVC** (e.g. a user-workspace claim) → `/scratch` — see below |

**Arguments** (single line; `/scratch` = your scratch mount's container
path):

```
-c "export TMPDIR=/scratch && pip install --target /scratch/boot uv && /scratch/boot/bin/uv pip install --target /scratch/deps fastapi uvicorn transformers==4.57.1 diffusers accelerate einops scipy numpy pillow tqdm torchvision && curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /scratch && cd /scratch/RunAI_apps-main && PYTHONPATH=/scratch/deps python3 image_app/scripts/hidream_server.py"
```

Why this differs from the embedding server's simpler
`uv pip install --system` pattern: **inference containers on this
cluster reject writes outside mounted volumes** — a bare `pip install`
dies in seconds with no visible log. So every write is redirected to
the scratch mount: packages via `pip/uv --target` + `PYTHONPATH`, temp
files via `TMPDIR` (which the server script honors for its upstream
code fetch). Two consequences:

- The isolated `--target` env can't see the image's torch, so uv
  installs a fresh torch + CUDA stack (~2.5 GB download, **~8 GB on
  the scratch volume** — size the claim accordingly). `torchvision`
  must be in the install list: without it Python falls through to the
  image's copy, which was built against a different torch and fails
  with `operator torchvision::nms does not exist`.
- `transformers==4.57.1` is upstream's pin (a version with native
  qwen3_vl support).

> **Debugging tip:** a failing startup crash-loops too fast for the
> RunAI log viewer. Temporarily append `|| sleep 3600` inside the
> quoted arguments — on failure the container stays up and the full
> error is readable in the Logs tab. Remove it once the workload is
> healthy so real failures restart cleanly.

**Environment variables** (all optional — script defaults shown):

| Name | Default | Why change it |
|------|---------|---------------|
| `HIDREAM_MODEL_ID` | `HiDream-ai/HiDream-O1-Image-Dev-2604` | Newer checkpoint on the PVC |
| `HIDREAM_MODEL_TYPE` | `dev` | `full` for the undistilled 50-step model (different checkpoint) |
| `HIDREAM_REF` | *(pinned commit)* | Track a different upstream ref |
| `PORT` | `8000` | Must match the serving endpoint port |

Startup takes ~3–5 min: dependency install, upstream code fetch,
loading ~18 GB of bf16 weights. The log sequence to expect:
`Model snapshot: ...` → `Fetching HiDream pipeline code` →
`Patched pipeline for SDPA attention` → `Loading ... in bf16` →
`Model ready.` → uvicorn on `0.0.0.0:8000`.

## Usage

```bash
curl -s https://<endpoint-url>/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A friendly cartoon diagram of a GPU cluster: three server racks with smiling faces, arrows labeled JOBS flowing from a laptop, flat vector style", "width": 2048, "height": 1152, "seed": 7}' \
  -o diagram.png
```

Request fields: `prompt` (required), `width`/`height` (512–2048,
multiples of 64; the pipeline snaps to its closest supported
resolution), `seed`. Response is a PNG; the `X-Generation-Seconds`
header reports render time. `GET /health` for probes.

Generation is serialized on the GPU — concurrent requests queue. The
`dev` model runs 28 steps per image.

### Prompting tips for diagrams

- Spell out any text you want rendered, in quotes; keep labels short.
- Name a style: "flat vector illustration", "whiteboard sketch",
  "isometric cartoon".
- HiDream-O1 responds well to long, detailed spatial descriptions
  (see the examples on its model card).
- For structurally exact diagrams (flowcharts where arrows must
  connect the right boxes), use Mermaid/draw.io/SVG instead —
  diffusion models draw plausible-looking, not logically-correct,
  structure.

### Maintenance notes

- `HIDREAM_REF` pins upstream's code; the server verifies its
  no-flash-attn patch applied and exits with a clear error if upstream
  moved the code. Bump the ref deliberately, not automatically.
- Upstream's pipeline also supports image editing and multi-reference
  subject generation; the server exposes text-to-image only. Extend
  `GenerateRequest` if those are ever needed.

---

## Appendix: Qwen-Image alternative

`scripts/qwen_image_server.py` is the same idea for Qwen/Qwen-Image
(Apache 2.0, stronger safety filtering, mid-pack leaderboard quality,
best-in-class text rendering). Download `Qwen/Qwen-Image` to the PVC
(~55–60 GB), then deploy with serving endpoint port `8080` and:

```
-c "pip install uv && curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp && mv /tmp/RunAI_apps-main /tmp/RunAI_apps && cd /tmp/RunAI_apps && uv pip install --system fastapi uvicorn pillow 'diffusers>=0.35' 'transformers>=4.53,<5' accelerate safetensors && python3 image_app/scripts/qwen_image_server.py"
```

Full GPU (bf16 needs ~60 GB VRAM), or env `OFFLOAD=1` for smaller
GPUs. `POST /generate` takes JSON (`prompt`, `aspect_ratio` of
`1:1|16:9|9:16|4:3|3:4`, `num_inference_steps`, `seed`) and returns a
PNG; it also serves a minimal HTML prompt form at `GET /`.
