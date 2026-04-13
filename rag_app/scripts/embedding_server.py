#!/usr/bin/env python3
"""FastAPI embedding server wrapping JinaV4EmbeddingModel.

Serves embeddings over HTTP so the Streamlit app (running in a separate
RunAI inference job) can call it without loading the model locally.

Launch:
    python scripts/embedding_server.py
    # or with uvicorn directly:
    uvicorn scripts.embedding_server:app --host 0.0.0.0 --port 8080

Environment variables:
    EMBEDDING_MODEL    - HuggingFace model ID (default: jinaai/jina-embeddings-v4)
    EMBEDDING_TASK     - Task mode: retrieval, text-matching, code (default: retrieval)
    EMBEDDING_DIM      - Matryoshka dimension (default: 1024)
    EMBEDDING_PORT     - Server port (default: 8080)
    EMBEDDING_HOST     - Server host (default: 0.0.0.0)
"""

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Writable HF cache overlay — MUST run before any HuggingFace imports.
#
# The shared PVC at /models is often read-only despite being configured for
# read-write (a RunAI/cluster admin issue).  HuggingFace tries to write
# metadata (refs, .no_exist negative cache, xet logs) which produces noisy
# errors on a read-only FS.
#
# Solution: create a writable cache at /tmp/hf_home and symlink model weight
# directories back to the PVC.  HF can write metadata freely to /tmp, while
# reading the large model weights from the PVC via symlinks.
# ---------------------------------------------------------------------------
_PVC_HF_CACHE = "/models/.cache/huggingface"
_WRITABLE_HF_HOME = "/tmp/hf_home"


def _setup_hf_cache_overlay():
    """Create a writable HF cache at /tmp/hf_home that symlinks to read-only PVC model files."""
    if not os.path.isdir(_PVC_HF_CACHE):
        return  # No PVC mounted, use defaults

    os.makedirs(_WRITABLE_HF_HOME, exist_ok=True)

    # Scan PVC for model directories and build writable overlay
    for entry in os.listdir(_PVC_HF_CACHE):
        src = os.path.join(_PVC_HF_CACHE, entry)
        if entry.startswith("models--") and os.path.isdir(src):
            model_dir = os.path.join(_WRITABLE_HF_HOME, entry)
            # Create writable dirs for metadata HF wants to write
            os.makedirs(os.path.join(model_dir, "snapshots"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, "refs"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, ".no_exist"), exist_ok=True)

            # Symlink each snapshot hash dir (the actual model weights)
            snap_src = os.path.join(src, "snapshots")
            if os.path.isdir(snap_src):
                for snap in os.listdir(snap_src):
                    snap_dst = os.path.join(model_dir, "snapshots", snap)
                    if not os.path.exists(snap_dst):
                        os.symlink(os.path.join(snap_src, snap), snap_dst)

            # Copy refs (tiny text files with commit hashes) so HF can overwrite
            refs_src = os.path.join(src, "refs")
            if os.path.isdir(refs_src):
                for ref in os.listdir(refs_src):
                    ref_dst = os.path.join(model_dir, "refs", ref)
                    if not os.path.exists(ref_dst):
                        shutil.copy2(os.path.join(refs_src, ref), ref_dst)

            # Symlink .locks if present
            locks_src = os.path.join(src, ".locks")
            locks_dst = os.path.join(model_dir, ".locks")
            if os.path.isdir(locks_src) and not os.path.exists(locks_dst):
                os.makedirs(locks_dst, exist_ok=True)

        elif entry == ".locks":
            dst = os.path.join(_WRITABLE_HF_HOME, entry)
            if not os.path.exists(dst):
                os.makedirs(dst, exist_ok=True)
        elif not os.path.exists(os.path.join(_WRITABLE_HF_HOME, entry)):
            # Symlink other items (e.g. hub/)
            os.symlink(src, os.path.join(_WRITABLE_HF_HOME, entry))

    os.environ["HF_HOME"] = _WRITABLE_HF_HOME
    os.environ["HF_HUB_CACHE"] = _WRITABLE_HF_HOME
    os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules")
    # Redirect xet logging (used by HF for large file downloads)
    os.environ.setdefault("XET_LOG_PATH", "/tmp/xet_logs")
    # Redirect pip cache in case pip runs inside the container
    os.environ.setdefault("PIP_CACHE_DIR", "/tmp/pip_cache")

    print(f"[embedding_server] Writable HF overlay created at {_WRITABLE_HF_HOME}", flush=True)


_setup_hf_cache_overlay()

# ---------------------------------------------------------------------------
# TEMP WORKAROUND: The shared PVC has jina-embeddings-v4 weights but is
# missing the adapters/ directory (admin needs to re-download).  If adapters
# aren't in the snapshot, download them to /tmp before going offline.
# This re-downloads on each cold start (~few hundred MB) but avoids needing
# write access to any PVC.  Remove once the admin fixes the shared PVC.
# ---------------------------------------------------------------------------
def _ensure_adapters():
    """Download jina-v4 adapters to /tmp if missing from snapshot, then symlink into overlay."""
    hf_cache = os.environ.get("HF_HUB_CACHE", os.environ.get("HF_HOME", ""))
    model_dir = "models--jinaai--jina-embeddings-v4"
    snap_dir = os.path.join(hf_cache, model_dir, "snapshots")

    # Find the snapshot directory (if it exists)
    snap_path = None
    if os.path.isdir(snap_dir):
        snaps = sorted(os.listdir(snap_dir))
        if snaps:
            snap_path = os.path.join(snap_dir, snaps[-1])
            if os.path.isdir(os.path.join(snap_path, "adapters")):
                print("[embedding_server] Adapters found in snapshot, no download needed", flush=True)
                return

    tmp_adapters = "/tmp/jina-embeddings-v4/adapters"
    if not os.path.isdir(tmp_adapters):
        print("[embedding_server] Adapters missing — downloading to /tmp...", flush=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                "jinaai/jina-embeddings-v4",
                allow_patterns=["adapters/*"],
                local_dir="/tmp/jina-embeddings-v4",
            )
            print(f"[embedding_server] Adapters downloaded to {tmp_adapters}", flush=True)
        except Exception as e:
            print(f"[embedding_server] WARNING: adapter download failed: {e}", flush=True)
            return
    else:
        print(f"[embedding_server] Adapters already at {tmp_adapters}", flush=True)

    os.environ.setdefault("JINA_ADAPTERS_DIR", tmp_adapters)

    # Symlink adapters into the writable overlay snapshot so from_pretrained
    # finds them directly (avoids the separate merged-snapshot workaround).
    if snap_path and os.path.isdir(tmp_adapters):
        adapters_dst = os.path.join(snap_path, "adapters")
        if not os.path.exists(adapters_dst):
            try:
                os.symlink(tmp_adapters, adapters_dst)
                print(f"[embedding_server] Adapters symlinked into overlay snapshot", flush=True)
            except OSError:
                # Snapshot dir might be a symlink to read-only PVC — that's fine,
                # the merged-snapshot fallback in embeddings.py will handle it.
                print(f"[embedding_server] Could not symlink adapters into snapshot (read-only target)", flush=True)

_ensure_adapters()

# Block all outgoing HF requests — we load exclusively from cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Import embeddings module directly to avoid kohakurag.__init__ pulling in
# kohakuvault (a Rust extension that isn't needed for the embedding server).
import importlib.util as _ilu

_repo_root = Path(__file__).resolve().parent.parent
_emb_path = _repo_root / "vendor" / "KohakuRAG" / "src" / "kohakurag" / "embeddings.py"
_spec = _ilu.spec_from_file_location("kohakurag.embeddings", _emb_path)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
JinaV4EmbeddingModel = _mod.JinaV4EmbeddingModel

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Energy measurement — runs on a GPU node so NVML should be available
# hardware_metrics.py lives in the top-level scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from hardware_metrics import NVMLEnergyCounter

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "jinaai/jina-embeddings-v4")
TASK = os.environ.get("EMBEDDING_TASK", "retrieval")
DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
HOST = os.environ.get("EMBEDDING_HOST", "0.0.0.0")
PORT = int(os.environ.get("EMBEDDING_PORT", "8080"))

# Startup diagnostics — print cache paths so we can verify the mount & overlay
_hf_home = os.environ.get("HF_HOME", "NOT SET")
_hf_hub = os.environ.get("HF_HUB_CACHE", "NOT SET")
print(f"[embedding_server] HF_HOME={_hf_home}", flush=True)
print(f"[embedding_server] HF_HUB_CACHE={_hf_hub}", flush=True)
if os.path.isdir(_PVC_HF_CACHE):
    print(f"[embedding_server] PVC cache at {_PVC_HF_CACHE}: {os.listdir(_PVC_HF_CACHE)}", flush=True)
    # Check if PVC is writable (for diagnostics only)
    _test_file = os.path.join(_PVC_HF_CACHE, ".write_test")
    try:
        with open(_test_file, "w") as f:
            f.write("test")
        os.remove(_test_file)
        print(f"[embedding_server] PVC is writable", flush=True)
    except OSError:
        print(f"[embedding_server] PVC is read-only (overlay handles this)", flush=True)
else:
    print(f"[embedding_server] WARNING: {_PVC_HF_CACHE} does not exist! PVC not mounted?", flush=True)
if os.path.isdir(_WRITABLE_HF_HOME):
    print(f"[embedding_server] Overlay at {_WRITABLE_HF_HOME}: {os.listdir(_WRITABLE_HF_HOME)}", flush=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="KohakuRAG Embedding Server", version="1.0.0")

# Global embedder — initialized on startup
_embedder: JinaV4EmbeddingModel | None = None
_energy_counter: NVMLEnergyCounter | None = None


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int
    count: int
    elapsed_ms: float
    energy_wh: float = 0.0  # GPU energy consumed by this request


class InfoResponse(BaseModel):
    model: str
    task: str
    dimension: int
    status: str


@app.on_event("startup")
async def startup():
    global _embedder, _energy_counter
    print(f"[embedding_server] Loading {MODEL_NAME} (task={TASK}, dim={DIM})...", flush=True)
    t0 = time.time()
    _embedder = JinaV4EmbeddingModel(
        model_name=MODEL_NAME,
        task=TASK,
        truncate_dim=DIM,
    )
    # Force model load at startup so we fail fast if model is missing
    _embedder._ensure_model()
    elapsed = time.time() - t0
    print(f"[embedding_server] Model loaded in {elapsed:.1f}s. Serving on {HOST}:{PORT}", flush=True)
    # Initialize NVML energy counter for per-request measurement
    _energy_counter = NVMLEnergyCounter()
    if _energy_counter.available:
        print("[embedding_server] NVML energy counter available — per-request energy will be reported", flush=True)
    else:
        print("[embedding_server] NVML energy counter not available — energy_wh will be 0", flush=True)


@app.get("/health")
async def health():
    """Always return 200 so Knative's readiness probe doesn't kill the container.

    Callers that need to know if the model is actually ready should check
    /info or POST /embed (which returns 503 while loading).
    """
    if _embedder is None:
        return {"status": "loading"}
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
async def info():
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return InfoResponse(
        model=MODEL_NAME,
        task=TASK,
        dimension=_embedder.dimension,
        status="ready",
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not request.texts:
        return EmbedResponse(
            embeddings=[], dimension=_embedder.dimension, count=0, elapsed_ms=0.0
        )

    # Measure GPU energy around the inference call
    if _energy_counter and _energy_counter.available:
        _energy_counter.start()

    t0 = time.time()
    vectors = await _embedder.embed(request.texts)
    elapsed_ms = (time.time() - t0) * 1000

    energy_wh = 0.0
    if _energy_counter and _energy_counter.available:
        per_gpu = _energy_counter.stop()
        energy_wh = sum(per_gpu.values()) if per_gpu else 0.0

    return EmbedResponse(
        embeddings=vectors.tolist(),
        dimension=_embedder.dimension,
        count=len(request.texts),
        elapsed_ms=round(elapsed_ms, 2),
        energy_wh=round(energy_wh, 8),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
