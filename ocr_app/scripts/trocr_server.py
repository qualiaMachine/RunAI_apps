#!/usr/bin/env python3
"""TrOCR line-OCR server.

Headless FastAPI endpoint around a TrOCR VisionEncoderDecoder checkpoint
(default: dh-unibe/trocr-kurrent-XVI-XVII for German Kurrent handwriting),
loading weights from the shared models PVC. Deployed on the stock
vllm/vllm-openai image, which already ships every dependency this needs —
no pip installs at startup.

TrOCR is LINE-level: send images of single text lines (e.g. Kraken
segmentation output), not full pages.

Endpoints:
    GET  /health     liveness probe
    POST /ocr        {"image_b64": "..."} -> {"text": "..."}
    POST /ocr_batch  {"images_b64": ["...", ...]} -> {"texts": [...]}
                     (one page's lines in one call; batched generate)

Environment variables:
    SHARED_MODELS_PATH  HF cache dir on the PVC (default /models/.cache/huggingface)
    TROCR_MODEL_ID      model on the PVC (default dh-unibe/trocr-kurrent-XVI-XVII)
    PORT                bind port (default 8000)
    MAX_NEW_TOKENS      generation cap per line (default 256)
"""

import base64
import io
import os
import sys
import threading
import time

PVC_CACHE = os.environ.get("SHARED_MODELS_PATH", "/models/.cache/huggingface")
MODEL_ID = os.environ.get("TROCR_MODEL_ID", "dh-unibe/trocr-kurrent-XVI-XVII")
PORT = int(os.environ.get("PORT", "8000"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
MAX_BATCH = 64


def fail(msg):
    print(f"FATAL: {msg}", flush=True)
    sys.exit(1)


def resolve_snapshot(model_id):
    model_dir = os.path.join(PVC_CACHE, f"models--{model_id.replace('/', '--')}")
    snap_root = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snap_root):
        fail(
            f"{model_id} not found at {model_dir}. Download it first: "
            f"python /models/provision_shared_models.py download {model_id}"
        )
    ref_main = os.path.join(model_dir, "refs", "main")
    if os.path.isfile(ref_main):
        with open(ref_main) as f:
            candidate = os.path.join(snap_root, f.read().strip())
        if os.path.isdir(candidate):
            return candidate
    snaps = sorted(os.listdir(snap_root))
    if not snaps:
        fail(f"No snapshots under {snap_root}")
    return os.path.join(snap_root, snaps[-1])


import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: E402

snapshot = resolve_snapshot(MODEL_ID)
print(f"Loading {MODEL_ID} from {snapshot} ...", flush=True)
processor = TrOCRProcessor.from_pretrained(snapshot)
model = VisionEncoderDecoderModel.from_pretrained(snapshot)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    model = model.half()
model = model.to(device).eval()
print(f"Model ready on {device}.", flush=True)

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

app = FastAPI(title="TrOCR line-OCR server")
gpu_lock = threading.Lock()


def decode_image(b64):
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        return img.convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image data: {e}")


def recognize(images):
    with gpu_lock, torch.inference_mode():
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        if device == "cuda":
            pixel_values = pixel_values.half()
        ids = model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS)
        return processor.batch_decode(ids, skip_special_tokens=True)


class OcrRequest(BaseModel):
    image_b64: str


class OcrBatchRequest(BaseModel):
    images_b64: list = Field(..., min_length=1, max_length=MAX_BATCH)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}


@app.post("/ocr")
def ocr(req: OcrRequest):
    t0 = time.time()
    text = recognize([decode_image(req.image_b64)])[0]
    return {"text": text, "seconds": round(time.time() - t0, 2)}


@app.post("/ocr_batch")
def ocr_batch(req: OcrBatchRequest):
    t0 = time.time()
    images = [decode_image(b) for b in req.images_b64]
    texts = recognize(images)
    return {"texts": texts, "seconds": round(time.time() - t0, 2)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
