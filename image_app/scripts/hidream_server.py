#!/usr/bin/env python3
"""HiDream-O1-Image text-to-image server.

Headless FastAPI endpoint around HiDream's official inference pipeline
(github.com/HiDream-ai/HiDream-O1-Image, dev branch), following the same
deployment pattern as rag_app/scripts/embedding_server.py: stock
vllm/vllm-openai image, lightweight deps installed at startup, weights
loaded from the shared models PVC.

At startup the server fetches a pinned commit of HiDream's repo, patches
it for machines without flash-attn (their documented fallback), loads the
model in bf16 from the PVC, then serves:

    GET  /health    liveness/readiness probe
    POST /generate  JSON in -> PNG out

Environment variables:
    SHARED_MODELS_PATH  HF cache dir on the PVC (default /models/.cache/huggingface)
    HIDREAM_MODEL_ID    model on the PVC (default HiDream-ai/HiDream-O1-Image-Dev-2604)
    HIDREAM_MODEL_TYPE  "dev" (distilled, 28 steps) or "full" (50 steps)
    HIDREAM_REF         upstream git ref to fetch (default: pinned commit)
    PORT                bind port (default 8000)
"""

import io
import os
import shutil
import sys
import tarfile
import threading
import time
import urllib.request

PVC_CACHE = os.environ.get("SHARED_MODELS_PATH", "/models/.cache/huggingface")
MODEL_ID = os.environ.get("HIDREAM_MODEL_ID", "HiDream-ai/HiDream-O1-Image-Dev-2604")
MODEL_TYPE = os.environ.get("HIDREAM_MODEL_TYPE", "dev")
UPSTREAM_REF = os.environ.get("HIDREAM_REF", "3237a638a5c2c7be106b0175958f4c0db8c2dfbf")
PORT = int(os.environ.get("PORT", "8000"))
UPSTREAM_DIR = "/tmp/hidream-upstream"


def fail(msg: str):
    print(f"FATAL: {msg}", flush=True)
    sys.exit(1)


def resolve_snapshot(model_id: str) -> str:
    """Find the model's snapshot directory on the (read-only) PVC cache."""
    model_dir = os.path.join(PVC_CACHE, f"models--{model_id.replace('/', '--')}")
    snap_root = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snap_root):
        fail(
            f"{model_id} not found at {model_dir}. Download it to the shared PVC "
            f"first: python /models/provision_shared_models.py download {model_id}"
        )
    ref_main = os.path.join(model_dir, "refs", "main")
    if os.path.isfile(ref_main):
        with open(ref_main) as f:
            commit = f.read().strip()
        candidate = os.path.join(snap_root, commit)
        if os.path.isdir(candidate):
            return candidate
    snapshots = sorted(os.listdir(snap_root))
    if not snapshots:
        fail(f"No snapshots under {snap_root}")
    return os.path.join(snap_root, snapshots[-1])


def fetch_upstream(ref: str) -> str:
    """Download HiDream's pipeline code at a pinned ref and patch it for
    machines without flash-attn (their README-documented fallback)."""
    url = f"https://github.com/HiDream-ai/HiDream-O1-Image/archive/{ref}.tar.gz"
    tarball = "/tmp/hidream-upstream.tar.gz"
    print(f"Fetching HiDream pipeline code: {url}", flush=True)
    urllib.request.urlretrieve(url, tarball)

    shutil.rmtree(UPSTREAM_DIR, ignore_errors=True)
    with tarfile.open(tarball) as tf:
        top = tf.getnames()[0].split("/")[0]
        tf.extractall("/tmp")
    shutil.move(os.path.join("/tmp", top), UPSTREAM_DIR)

    # No flash-attn in the stock image. Two-part fallback per upstream README:
    # FA_VERSION != 2/3 turns their hard `import flash_attn` into a soft one,
    # and the pipeline must stop requesting the flash kernel.
    os.environ["FA_VERSION"] = "0"
    pipeline_py = os.path.join(UPSTREAM_DIR, "models", "pipeline.py")
    with open(pipeline_py) as f:
        src = f.read()
    patched = src.replace('"use_flash_attn": True', '"use_flash_attn": False')
    if patched == src:
        fail(
            "use_flash_attn patch found nothing to replace in models/pipeline.py "
            "— upstream code changed? Check HIDREAM_REF."
        )
    with open(pipeline_py, "w") as f:
        f.write(patched)
    print("Patched pipeline for SDPA attention (no flash-attn).", flush=True)
    return UPSTREAM_DIR


snapshot_path = resolve_snapshot(MODEL_ID)
print(f"Model snapshot: {snapshot_path}", flush=True)
sys.path.insert(0, fetch_upstream(UPSTREAM_REF))

import torch  # noqa: E402

if not torch.cuda.is_available():
    fail("CUDA is not available in this container.")

from transformers import AutoProcessor  # noqa: E402
from models.pipeline import DEFAULT_TIMESTEPS, generate_image  # noqa: E402
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # noqa: E402

print(f"Loading {MODEL_ID} in bf16 ...", flush=True)
processor = AutoProcessor.from_pretrained(snapshot_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    snapshot_path, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()

# Special-token shortcuts the pipeline expects on the tokenizer.
_tokenizer = getattr(processor, "tokenizer", processor)
_tokenizer.boi_token = "<|boi_token|>"
_tokenizer.bor_token = "<|bor_token|>"
_tokenizer.eor_token = "<|eor_token|>"
_tokenizer.bot_token = "<|bot_token|>"
_tokenizer.tms_token = "<|tms_token|>"
print("Model ready.", flush=True)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

app = FastAPI(title="HiDream-O1-Image server")

# One GPU, one generation at a time. Concurrent requests queue here.
gpu_lock = threading.Lock()

if MODEL_TYPE == "full":
    GEN_KWARGS = dict(
        num_inference_steps=50, guidance_scale=5.0, shift=3.0,
        timesteps_list=None, scheduler_name="default",
    )
else:  # dev (distilled): 28-step flash scheduler, mirrors upstream app.py
    GEN_KWARGS = dict(
        num_inference_steps=28, guidance_scale=0.0, shift=1.0,
        timesteps_list=DEFAULT_TIMESTEPS, scheduler_name="flash",
        noise_scale_start=8.0, noise_scale_end=8.0, noise_clip_std=0.0,
    )


class GenerateRequest(BaseModel):
    prompt: str
    width: int = Field(2048, ge=512, le=2048, multiple_of=64)
    height: int = Field(2048, ge=512, le=2048, multiple_of=64)
    seed: int = 42


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "model_type": MODEL_TYPE}


@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    with gpu_lock:
        t0 = time.time()
        image = generate_image(
            model=model,
            processor=processor,
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            seed=req.seed,
            **GEN_KWARGS,
        )
        elapsed = time.time() - t0
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"X-Generation-Seconds": f"{elapsed:.1f}"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
