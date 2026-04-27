#!/usr/bin/env python3
"""FastAPI cross-encoder reranker server.

Serves reranking over HTTP so the Streamlit app (running in a separate
RunAI inference job) can call it without loading the model locally.

Launch:
    python scripts/reranker_server.py
    # or with uvicorn directly:
    uvicorn scripts.reranker_server:app --host 0.0.0.0 --port 8082

Environment variables:
    RERANKER_MODEL  - HuggingFace model ID (default: BAAI/bge-reranker-v2-m3)
    RERANKER_PORT   - Server port (default: 8082)
    RERANKER_HOST   - Server host (default: 0.0.0.0)
    RERANKER_DEVICE - PyTorch device (default: auto)
    RERANKER_BATCH  - Batch size for inference (default: 32)
"""

import os
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Writable HF cache overlay (same pattern as embedding_server.py)
# ---------------------------------------------------------------------------
_PVC_HF_CACHE = "/models/.cache/huggingface"
_WRITABLE_HF_HOME = "/tmp/hf_home"


def _setup_hf_cache_overlay():
    """Create a writable HF cache at /tmp/hf_home that symlinks to read-only PVC model files."""
    if not os.path.isdir(_PVC_HF_CACHE):
        return

    os.makedirs(_WRITABLE_HF_HOME, exist_ok=True)

    for entry in os.listdir(_PVC_HF_CACHE):
        src = os.path.join(_PVC_HF_CACHE, entry)
        if entry.startswith("models--") and os.path.isdir(src):
            model_dir = os.path.join(_WRITABLE_HF_HOME, entry)
            os.makedirs(os.path.join(model_dir, "snapshots"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, "refs"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, ".no_exist"), exist_ok=True)

            snap_src = os.path.join(src, "snapshots")
            if os.path.isdir(snap_src):
                for snap in os.listdir(snap_src):
                    snap_dst = os.path.join(model_dir, "snapshots", snap)
                    if not os.path.exists(snap_dst):
                        os.symlink(os.path.join(snap_src, snap), snap_dst)

            refs_src = os.path.join(src, "refs")
            if os.path.isdir(refs_src):
                for ref in os.listdir(refs_src):
                    ref_dst = os.path.join(model_dir, "refs", ref)
                    if not os.path.exists(ref_dst):
                        shutil.copy2(os.path.join(refs_src, ref), ref_dst)

            locks_src = os.path.join(src, ".locks")
            locks_dst = os.path.join(model_dir, ".locks")
            if os.path.isdir(locks_src) and not os.path.exists(locks_dst):
                os.makedirs(locks_dst, exist_ok=True)

        elif entry == ".locks":
            dst = os.path.join(_WRITABLE_HF_HOME, entry)
            if not os.path.exists(dst):
                os.makedirs(dst, exist_ok=True)
        elif not os.path.exists(os.path.join(_WRITABLE_HF_HOME, entry)):
            os.symlink(src, os.path.join(_WRITABLE_HF_HOME, entry))

    os.environ["HF_HOME"] = _WRITABLE_HF_HOME
    os.environ["HF_HUB_CACHE"] = _WRITABLE_HF_HOME
    os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules")
    os.environ.setdefault("XET_LOG_PATH", "/tmp/xet_logs")
    os.environ.setdefault("PIP_CACHE_DIR", "/tmp/pip_cache")

    print(f"[reranker_server] Writable HF overlay created at {_WRITABLE_HF_HOME}", flush=True)


_setup_hf_cache_overlay()

# Block outgoing HF requests — load from cache only.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import CrossEncoder
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Energy measurement — runs on a GPU node so NVML should be available
# hardware_metrics.py lives in the top-level scripts/ directory
_top_level_scripts = str(Path(os.path.abspath(__file__)).parent.parent.parent / "scripts")
sys.path.insert(0, _top_level_scripts)
from hardware_metrics import NVMLEnergyCounter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
HOST = os.environ.get("RERANKER_HOST", "0.0.0.0")
PORT = int(os.environ.get("RERANKER_PORT", "8082"))
DEVICE = os.environ.get("RERANKER_DEVICE", None)  # None = auto
BATCH_SIZE = int(os.environ.get("RERANKER_BATCH", "32"))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="KohakuRAG Reranker Server", version="1.0.0")

_model: CrossEncoder | None = None
_energy_counter: NVMLEnergyCounter | None = None


class RerankRequest(BaseModel):
    query: str
    texts: list[str]


class RerankResponse(BaseModel):
    scores: list[float]
    count: int
    elapsed_ms: float
    energy_wh: float = 0.0  # GPU energy consumed by this request


class InfoResponse(BaseModel):
    model: str
    status: str


@app.on_event("startup")
async def startup():
    global _model, _energy_counter
    print(f"[reranker_server] Loading {MODEL_NAME} (device={DEVICE})...", flush=True)
    t0 = time.time()
    _model = CrossEncoder(MODEL_NAME, device=DEVICE)
    elapsed = time.time() - t0
    print(f"[reranker_server] Model loaded in {elapsed:.1f}s. Serving on {HOST}:{PORT}", flush=True)
    _energy_counter = NVMLEnergyCounter()
    if _energy_counter.available:
        print("[reranker_server] NVML energy counter available", flush=True)
    else:
        print("[reranker_server] NVML energy counter not available", flush=True)


@app.get("/health")
async def health():
    if _model is None:
        return {"status": "loading"}
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
async def info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return InfoResponse(model=MODEL_NAME, status="ready")


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not request.texts:
        return RerankResponse(scores=[], count=0, elapsed_ms=0.0)

    pairs = [(request.query, t) for t in request.texts]

    if _energy_counter and _energy_counter.available:
        _energy_counter.start()

    t0 = time.time()
    scores = _model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
    elapsed_ms = (time.time() - t0) * 1000

    energy_wh = 0.0
    if _energy_counter and _energy_counter.available:
        per_gpu = _energy_counter.stop()
        energy_wh = sum(per_gpu.values()) if per_gpu else 0.0

    return RerankResponse(
        scores=[float(s) for s in scores],
        count=len(request.texts),
        elapsed_ms=round(elapsed_ms, 2),
        energy_wh=round(energy_wh, 8),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
