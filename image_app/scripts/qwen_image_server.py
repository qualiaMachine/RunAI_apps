#!/usr/bin/env python3
"""Qwen-Image text-to-image server.

FastAPI wrapper around the Qwen/Qwen-Image diffusion pipeline, following
the same deployment pattern as rag_app/scripts/embedding_server.py:
stock vllm/vllm-openai image, lightweight deps installed at startup,
model weights loaded straight from the shared models PVC.

Endpoints:
    GET  /          minimal browser UI (prompt box -> image)
    GET  /health    liveness/readiness probe
    POST /generate  JSON in, PNG out

Environment variables:
    SHARED_MODELS_PATH  HF cache dir on the PVC (default /models/.cache/huggingface)
    QWEN_IMAGE_MODEL    HuggingFace model ID (default Qwen/Qwen-Image)
    PORT                bind port (default 8080)
    OFFLOAD             "1" = model CPU offload, for GPUs with < 80 GB VRAM
                        (slower per image, much smaller VRAM footprint)

The PVC is mounted read-only in inference workloads. We sidestep all
HuggingFace Hub metadata writes by resolving the model's snapshot
directory on the PVC ourselves and handing that local path to
diffusers — no writable HF cache overlay needed.
"""

import io
import os
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

PVC_CACHE = os.environ.get("SHARED_MODELS_PATH", "/models/.cache/huggingface")
MODEL_ID = os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image")
PORT = int(os.environ.get("PORT", "8080"))
OFFLOAD = os.environ.get("OFFLOAD", "0") == "1"

# Officially supported resolutions (model card). Keys are aspect ratios.
ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
}


def resolve_snapshot(model_id: str) -> str:
    """Find the model's snapshot directory on the (read-only) PVC cache."""
    model_dir = os.path.join(PVC_CACHE, f"models--{model_id.replace('/', '--')}")
    snap_root = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snap_root):
        raise RuntimeError(
            f"{model_id} not found at {model_dir}. Download it to the shared "
            f"PVC first: python /models/provision_shared_models.py download {model_id}"
        )
    # Prefer the commit that refs/main points at; else newest snapshot dir.
    ref_main = os.path.join(model_dir, "refs", "main")
    if os.path.isfile(ref_main):
        with open(ref_main) as f:
            commit = f.read().strip()
        candidate = os.path.join(snap_root, commit)
        if os.path.isdir(candidate):
            return candidate
    snapshots = sorted(os.listdir(snap_root))
    if not snapshots:
        raise RuntimeError(f"No snapshots under {snap_root}")
    return os.path.join(snap_root, snapshots[-1])


print(f"Resolving {MODEL_ID} on PVC cache {PVC_CACHE} ...")
snapshot_path = resolve_snapshot(MODEL_ID)
print(f"Loading pipeline from {snapshot_path} (offload={OFFLOAD}) ...")

from diffusers import DiffusionPipeline  # noqa: E402  (import after torch)

pipe = DiffusionPipeline.from_pretrained(snapshot_path, torch_dtype=torch.bfloat16)
if OFFLOAD:
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to("cuda")
print("Pipeline ready.")

# One GPU, one generation at a time. Concurrent requests queue here.
gpu_lock = threading.Lock()

app = FastAPI(title="Qwen-Image server")


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = " "
    aspect_ratio: str = Field("1:1", description=f"One of {list(ASPECT_RATIOS)}")
    num_inference_steps: int = Field(50, ge=1, le=100)
    true_cfg_scale: float = Field(4.0, ge=1.0, le=10.0)
    seed: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "offload": OFFLOAD}


@app.post("/generate")
def generate(req: GenerateRequest):
    if req.aspect_ratio not in ASPECT_RATIOS:
        raise HTTPException(400, f"aspect_ratio must be one of {list(ASPECT_RATIOS)}")
    width, height = ASPECT_RATIOS[req.aspect_ratio]

    generator = None
    if req.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(req.seed)

    with gpu_lock:
        t0 = time.time()
        image = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=req.num_inference_steps,
            true_cfg_scale=req.true_cfg_scale,
            generator=generator,
        ).images[0]
        elapsed = time.time() - t0

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"X-Generation-Seconds": f"{elapsed:.1f}"},
    )


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Qwen-Image</title>
<style>
  body { font-family: sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
  textarea { width: 100%; height: 6rem; }
  label { display: block; margin-top: .75rem; }
  img { max-width: 100%; margin-top: 1rem; }
  #status { margin-top: 1rem; color: #555; }
  button { margin-top: 1rem; padding: .5rem 1.5rem; }
</style>
</head>
<body>
<h1>Qwen-Image</h1>
<p>Text-to-image generation. One request renders at a time; expect
roughly one to three minutes per image.</p>
<textarea id="prompt" placeholder="A friendly cartoon diagram of ..."></textarea>
<label>Aspect ratio
  <select id="ratio">
    <option>1:1</option><option>16:9</option><option>9:16</option>
    <option>4:3</option><option>3:4</option>
  </select>
</label>
<label>Steps <input id="steps" type="number" value="50" min="1" max="100"></label>
<label>Seed (blank = random) <input id="seed" type="number"></label>
<button onclick="go()">Generate</button>
<div id="status"></div>
<img id="out" hidden>
<script>
async function go() {
  const status = document.getElementById('status');
  const img = document.getElementById('out');
  img.hidden = true;
  status.textContent = 'Generating...';
  const body = {
    prompt: document.getElementById('prompt').value,
    aspect_ratio: document.getElementById('ratio').value,
    num_inference_steps: parseInt(document.getElementById('steps').value),
  };
  const seed = document.getElementById('seed').value;
  if (seed !== '') body.seed = parseInt(seed);
  try {
    const resp = await fetch('generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const blob = await resp.blob();
    img.src = URL.createObjectURL(blob);
    img.hidden = false;
    status.textContent = 'Done in ' + (resp.headers.get('X-Generation-Seconds') || '?') + 's';
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  }
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
