"""
WattBot RAG — Streamlit App

Interactive UI for querying the WattBot RAG pipeline.

Supports two deployment modes:
  - **local**  (default): HF models loaded directly on GPU in this process.
  - **remote**: LLM served by vLLM, embeddings by a FastAPI server, both as
    separate RunAI inference jobs. Set RAG_MODE=remote + service URLs.

Launch (local):
    streamlit run app.py

Launch (remote — vLLM + embedding server):
    RAG_MODE=remote \\
    VLLM_BASE_URL=http://wattbot-vllm:8000/v1 \\
    EMBEDDING_SERVICE_URL=http://wattbot-embedding:8080 \\
    streamlit run app.py

    The served model name is auto-detected via GET /v1/models.
    Set VLLM_MODEL as a fallback if the vLLM server is unreachable at startup.
"""
from __future__ import annotations

import asyncio
import csv
import gc
import importlib.util
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent
_top_root = _repo_root.parent  # top-level repo directory
sys.path.insert(0, str(_repo_root / "vendor" / "KohakuRAG" / "src"))
sys.path.insert(0, str(_top_root / "scripts"))

from kohakurag import RAGPipeline, LLMQueryPlanner, SimpleQueryPlanner
from kohakurag.datastore import KVaultNodeStore, ImageStore
from kohakurag.semantic_scholar import SemanticScholarRetriever
from hardware_metrics import NVMLEnergyCounter, GPUPowerMonitor

# Cross-encoder reranker — requires sentence-transformers (optional)
try:
    from kohakurag.reranker import CrossEncoderReranker
    RERANKER_AVAILABLE = True
except Exception:
    RERANKER_AVAILABLE = False

# Local-only imports (require torch/transformers) — skip in remote mode
_rag_mode = os.environ.get("RAG_MODE", "local")
if _rag_mode == "local":
    from kohakurag.embeddings import JinaV4EmbeddingModel
    from kohakurag.llm import HuggingFaceLocalChatModel

# Remote inference clients (vLLM + embedding server + reranker)
try:
    from kohakurag.remote import RemoteEmbeddingModel, VLLMChatModel, RemoteCrossEncoderReranker
    REMOTE_AVAILABLE = True
except ImportError:
    REMOTE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Mode detection from environment
# ---------------------------------------------------------------------------
RAG_MODE = os.environ.get("RAG_MODE", "local")  # "local" or "remote"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "1024"))
VLLM_TEMPERATURE = float(os.environ.get("VLLM_TEMPERATURE", "0.2"))
EMBEDDING_SERVICE_URL = os.environ.get("EMBEDDING_SERVICE_URL", "http://localhost:8080")
RERANKER_SERVICE_URL = os.environ.get("RERANKER_SERVICE_URL", "")

# Multi-endpoint and knowledge-base discovery (remote mode)
# VLLM_ENDPOINTS: comma-separated name=url pairs for multiple vLLM servers.
#   e.g. "qwen7b=http://wattbot-qwen:8000/v1,llama8b=http://wattbot-llama:8000/v1"
VLLM_ENDPOINTS = os.environ.get("VLLM_ENDPOINTS", "")
# SHARED_MODELS_PATH: HuggingFace cache on the shared PVC (for model listing).
SHARED_MODELS_PATH = os.environ.get("SHARED_MODELS_PATH", "/models/.cache/huggingface")
# VECTOR_DB_DIRS: comma-separated directories to scan for knowledge base .db files.
# Each data volume mount should contain an embeddings/ subdirectory with .db files.
# e.g. "/climate-data/embeddings,/bio-data/embeddings,/custom-corpus/embeddings"
VECTOR_DB_DIRS = os.environ.get("VECTOR_DB_DIRS", os.environ.get("VECTOR_DB_DIR", ""))

# ---------------------------------------------------------------------------
# Print RunAI access URL at startup (if running inside a RunAI workspace)
# ---------------------------------------------------------------------------
_runai_project = os.environ.get("RUNAI_PROJECT", "")
_runai_job_name = os.environ.get("RUNAI_JOB_NAME", "")
_runai_cluster_host = os.environ.get("RUNAI_CLUSTER_HOST", "deepthought.doit.wisc.edu")
if _runai_project and _runai_job_name:
    _app_url = f"https://{_runai_cluster_host}/{_runai_project}/{_runai_job_name}/proxy/8501/"
    print(f"\n{'='*60}")
    print(f"  WattBot RAG is available at:")
    print(f"  {_app_url}")
    print(f"{'='*60}\n")



def _detect_vllm_model(base_url: str) -> str:
    """Query the vLLM /v1/models endpoint to discover the served model.

    Falls back to VLLM_MODEL env var (then a hardcoded default) if the
    server is unreachable.
    """
    fallback = os.environ.get("VLLM_MODEL", "unknown")
    try:
        import httpx
        resp = httpx.get(f"{base_url}/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return fallback


# ---------------------------------------------------------------------------
# Discovery helpers (remote mode: models on PVC, knowledge bases, endpoints)
# ---------------------------------------------------------------------------

def _scan_pvc_models(cache_dir: str | None = None) -> list[dict]:
    """Scan the shared HuggingFace cache for available models.

    Returns list of dicts with keys: name, size_gb.
    """
    cache_dir = cache_dir or SHARED_MODELS_PATH
    if not os.path.isdir(cache_dir):
        return []
    results = []
    try:
        entries = sorted(
            e for e in os.listdir(cache_dir)
            if e.startswith("models--") and os.path.isdir(os.path.join(cache_dir, e))
        )
    except OSError:
        return []
    for dir_name in entries:
        model_name = dir_name.replace("models--", "").replace("--", "/")
        model_path = os.path.join(cache_dir, dir_name)
        total_bytes = 0
        try:
            for root, _dirs, files in os.walk(model_path):
                for f in files:
                    fp = os.path.join(root, f)
                    if not os.path.islink(fp):
                        total_bytes += os.path.getsize(fp)
        except OSError:
            pass
        results.append({"name": model_name, "size_gb": total_bytes / (1024 ** 3)})
    return results


def _detect_table_prefix(db_path: str) -> str | None:
    """Auto-detect the KVault table prefix from a .db file.

    KVaultNodeStore creates tables named '{prefix}_kv', '{prefix}_vec', etc.
    Opens the DB read-only and returns the first detected prefix, or None.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%\\_kv' ESCAPE '\\'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        # Filter out system/internal tables
        prefixes = [t.removesuffix("_kv") for t in tables if t.endswith("_kv")]
        return prefixes[0] if prefixes else None
    except Exception:
        return None


def _scan_knowledge_bases() -> list[dict]:
    """Discover vector DB files across configured mount points.

    Supports multiple data volumes — each researcher can attach their own
    PVC and have its .db files appear in the sidebar selector.

    Search order:
      1. VECTOR_DB_DIRS env var (comma-separated list of directories)
      2. Auto-discovered data volume mounts (/*-data/embeddings/, /*-corpus/embeddings/)
      3. Hardcoded fallback paths (wattbot-data, local dev)

    Returns list of dicts with keys: path, name, size_mb, table_prefix.
    """
    search_dirs: list[str] = []

    def _add_dir(d: str) -> None:
        d = d.rstrip("/")
        if d and os.path.isdir(d) and d not in search_dirs:
            search_dirs.append(d)

    # 1. VECTOR_DB_DIRS env var (user-specified, comma-separated)
    for d in VECTOR_DB_DIRS.split(","):
        _add_dir(d.strip())

    # 2. Auto-discover data volume mounts at filesystem root.
    #    RunAI PVCs are typically mounted at /<name> (e.g. /wattbot-data,
    #    /climate-corpus). Look for any top-level mount that contains an
    #    embeddings/ subdirectory with .db files.
    try:
        for entry in os.listdir("/"):
            candidate = f"/{entry}/embeddings"
            if entry.startswith(".") or entry in (
                "bin", "boot", "dev", "etc", "home", "lib", "lib64",
                "media", "mnt", "opt", "proc", "root", "run", "sbin",
                "srv", "sys", "tmp", "usr", "var", "models", "snap",
            ):
                continue
            _add_dir(candidate)
    except OSError:
        pass

    # 3. Hardcoded fallback paths
    _add_dir("/wattbot-data/embeddings")
    _add_dir("/tmp/vectordb")
    _add_dir(str(_repo_root / "data" / "embeddings"))

    seen_paths: set[str] = set()
    results: list[dict] = []
    for d in search_dirs:
        try:
            for fname in sorted(os.listdir(d)):
                if not fname.endswith(".db"):
                    continue
                fpath = os.path.join(d, fname)
                # Resolve symlinks to avoid duplicates
                real = os.path.realpath(fpath)
                if real in seen_paths:
                    continue
                seen_paths.add(real)
                try:
                    size_mb = os.path.getsize(real) / (1024 ** 2)
                except OSError:
                    size_mb = 0
                prefix = _detect_table_prefix(real)
                if prefix is None:
                    continue  # Skip files that aren't KVault databases
                results.append({
                    "path": fpath,
                    "name": fname,
                    "size_mb": size_mb,
                    "table_prefix": prefix,
                })
        except OSError:
            continue
    return results


def _list_vllm_endpoints() -> list[dict]:
    """Build list of available vLLM endpoints from env config.

    Sources:
      1. VLLM_BASE_URL env var (always included as 'default')
      2. VLLM_ENDPOINTS env var (comma-separated name=url pairs)

    Returns list of dicts: {name, url, model, status}.
    """
    endpoints: list[dict] = []

    # Primary endpoint
    primary_model = _detect_vllm_model(VLLM_BASE_URL)
    endpoints.append({
        "name": "default",
        "url": VLLM_BASE_URL,
        "model": primary_model,
        "status": "online" if primary_model != "unknown" else "offline",
    })

    # Additional endpoints from VLLM_ENDPOINTS
    if VLLM_ENDPOINTS:
        for part in VLLM_ENDPOINTS.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                name, url = part.split("=", 1)
            else:
                name, url = part, part
            name = name.strip()
            url = url.strip()
            # Skip if same URL as primary
            if url == VLLM_BASE_URL:
                continue
            model = _detect_vllm_model(url)
            endpoints.append({
                "name": name,
                "url": url,
                "model": model,
                "status": "online" if model != "unknown" else "offline",
            })

    return endpoints


# ---------------------------------------------------------------------------
# Energy tracking
# ---------------------------------------------------------------------------

# GPU TDP (watts) — used for local NVML fallback and vLLM estimation.
# Override via env var to match your cluster hardware (e.g. 300 for A100,
# 72 for L4, 450 for 4090).
_GPU_TDP_WATTS = float(os.environ.get("ENERGY_GPU_TDP_WATTS", "300"))

# GPU VRAM capacity in GB — used to scale model power fractions.
# Override via env var for your hardware (e.g. 80 for A100-80GB, 24 for L4).
_GPU_VRAM_GB = float(os.environ.get("ENERGY_GPU_VRAM_GB", "80"))

# Typical GPU utilisation during active inference (used for estimation).
# 0.50 is a conservative default — real utilisation varies by batch size and
# model; override via env var if you have empirical measurements.
_GPU_UTIL = float(os.environ.get("ENERGY_GPU_UTIL", "0.50"))

# In remote mode, retrieval_s includes network round-trips, CPU-based vector
# search, BM25 scoring, and Semantic Scholar API calls — only a small fraction
# is actual GPU compute (embedding + reranking).  This factor estimates the
# GPU-active share of total retrieval wall-clock time.
_REMOTE_RETRIEVAL_GPU_FRAC = float(os.environ.get("ENERGY_RETRIEVAL_GPU_FRAC", "0.10"))

# Bascom Hill energy comparison (fun UW-Madison context for session energy).
# Climbing one stair step burns ~0.15 Wh of metabolic energy for an average
# adult (70 kg × 9.8 m/s² × 0.2 m step height ÷ 25% muscle efficiency).
# Bascom Hill: 110 steps from Park St to Bascom Hall.
_WH_PER_BASCOM_STEP = 0.15
_BASCOM_HILL_STEPS = 110

# DCGM (Data Center GPU Manager) exporter URL — if available on the cluster,
# provides real GPU power measurements.  Set to empty string to disable.
_DCGM_URL = os.environ.get("DCGM_EXPORTER_URL", "")

# Token-based energy estimation: baseline Wh per token at 8B parameters.
# Derived from Luccioni et al. 2024 measurements on 8B-class models:
#   ~0.093 Wh total for a typical query (~2000 prompt + ~200 completion tokens)
# Prefill (prompt) tokens are cheaper than decode (completion) tokens because
# they're processed in a single batch, while decode is sequential.
# Rates scale with model size via a power law (see _token_energy_wh).
_BASE_WH_PER_PROMPT_TOKEN = float(os.environ.get("ENERGY_WH_PER_PROMPT_TOKEN", "2.0e-5"))
_BASE_WH_PER_COMPLETION_TOKEN = float(os.environ.get("ENERGY_WH_PER_COMPLETION_TOKEN", "2.5e-4"))
_BASE_MODEL_PARAMS_B = 8.0  # baseline model size for the above rates
# Scaling exponent: energy ~ params^1.3 (super-linear due to memory bandwidth,
# multi-GPU communication).  Fit to published data: 6.7B→0.082 Wh, 8B→0.093 Wh,
# 175B→4.0 Wh, 405B→17.3 Wh (Luccioni et al. 2024, Moore et al. 2025).
_ENERGY_SCALING_EXPONENT = float(os.environ.get("ENERGY_SCALING_EXPONENT", "1.3"))


def _parse_param_billions(model_name: str) -> float:
    """Extract *active* parameter count (billions) from a model name.

    For dense models this is the total parameter count.  For MoE (Mixture
    of Experts) models like ``Mixtral-8x7B``, only a fraction of experts
    are active per token, so we estimate the active parameter count:

        active ≈ (num_active_experts / num_experts) × expert_params × num_experts + shared_params
               ≈ num_active_experts × expert_params + shared_overhead

    Heuristic: MoE models typically activate 2 experts per token.  With
    a ~25% shared-parameter overhead (attention, embedding, etc.), the
    active params for ``8x7B`` ≈ 2×7 + 0.25×56 ≈ 28 → ~13B active.
    We approximate as: ``2 × per_expert + 0.15 × total``.

    Examples::

        "OpenSciLM/Llama-3.1_OpenScholar-8B"     → 8.0
        "Qwen/Qwen2.5-72B-Instruct"              → 72.0
        "mistralai/Mixtral-8x7B-Instruct"         → 22.4  (2×7 + 0.15×56 active)
        "Qwen/Qwen3-30B-A3B"                      → 3.0   (explicit active count)

    Returns a sensible default (8.0) if no pattern matches.
    """
    if not model_name:
        return 8.0

    # Explicit active-param pattern: "30B-A3B" (total 30B, active 3B)
    m = re.search(r"(\d+\.?\d*)\s*[bB][-_]A(\d+\.?\d*)\s*[bB]", model_name, re.IGNORECASE)
    if m:
        return float(m.group(2))

    # MoE pattern: 8x7B → estimate active params
    m = re.search(r"(\d+)[xX](\d+\.?\d*)[bB]", model_name, re.IGNORECASE)
    if m:
        num_experts = float(m.group(1))
        expert_params = float(m.group(2))
        total = num_experts * expert_params
        # ~2 active experts + ~15% shared overhead (attention, embeddings)
        active = 2 * expert_params + 0.15 * total
        return active

    # Standard dense model: 72B, 1.5B, etc.
    m = re.search(r"(\d+\.?\d*)\s*[bB](?!\w*yte)", model_name)
    if m:
        return float(m.group(1))
    return 8.0


def _model_power_fraction(param_billions: float) -> float:
    """Estimate GPU power draw as a fraction of TDP from model size.

    Uses model VRAM footprint (≈0.75 GB per billion params at 4-bit)
    relative to total GPU VRAM as a proxy, with a baseline for compute
    overhead even on small models.
    """
    model_vram = param_billions * 0.75          # approx 4-bit VRAM
    fraction = 0.25 + (model_vram / _GPU_VRAM_GB)
    return max(0.15, min(0.95, fraction))


class DCGMPowerSampler:
    """Poll DCGM exporter for real GPU power during a query.

    Two measurement strategies (in priority order):

    1. **Energy counter** — reads ``DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION``
       (a monotonic counter in millijoules) at start and stop, giving an
       exact delta.  Most accurate since it's a hardware counter.

    2. **Power polling** — samples ``DCGM_FI_DEV_POWER_USAGE`` (gauge, watts)
       in a background thread and integrates with the trapezoidal rule.
       Used when the energy counter isn't available.

    Falls back gracefully (energy_wh = 0) if the endpoint is unreachable.
    """

    def __init__(self, dcgm_url: str, interval: float = 0.5):
        self._url = dcgm_url.rstrip("/") + "/metrics"
        self._interval = interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._power_readings: list[float] = []  # watts
        self._timestamps: list[float] = []
        self._available = bool(dcgm_url)
        # Energy counter (millijoules) — read at start/stop for exact delta
        self._start_energy_mj: float | None = None
        self._use_energy_counter = False

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        if not self._available:
            return
        self._stop_event.clear()
        self._power_readings = []
        self._timestamps = []
        self._start_energy_mj = None
        self._use_energy_counter = False

        # Try to read the energy counter first
        metrics = self._fetch_metrics()
        if metrics is not None:
            energy_mj = self._parse_metric(metrics, "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION")
            if energy_mj is not None:
                self._start_energy_mj = energy_mj
                self._use_energy_counter = True
                return  # No polling thread needed

        # Fallback: poll power in a background thread
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._use_energy_counter:
            # Will compute delta in energy_wh property
            metrics = self._fetch_metrics()
            if metrics is not None:
                end_mj = self._parse_metric(metrics, "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION")
                if end_mj is not None and self._start_energy_mj is not None:
                    self._end_energy_mj = end_mj
                    return
            self._use_energy_counter = False  # Counter disappeared, fall through
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5)
            self._thread = None

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            metrics = self._fetch_metrics()
            if metrics is not None:
                power = self._parse_metric(metrics, "DCGM_FI_DEV_POWER_USAGE")
                if power is not None:
                    self._power_readings.append(power)
                    self._timestamps.append(time.time())
            self._stop_event.wait(self._interval)

    def _fetch_metrics(self) -> str | None:
        """Fetch the full /metrics text from DCGM exporter."""
        try:
            import httpx
            resp = httpx.get(self._url, timeout=2.0)
            return resp.text if resp.status_code == 200 else None
        except Exception:
            return None

    @staticmethod
    def _parse_metric(text: str, metric_name: str) -> float | None:
        """Extract the first value for *metric_name* from Prometheus text."""
        for line in text.splitlines():
            if line.startswith(metric_name) and not line.startswith("#"):
                # e.g. DCGM_FI_DEV_POWER_USAGE{gpu="0",...} 142.5
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass
        return None

    @property
    def energy_wh(self) -> float:
        """Total energy consumed in Watt-hours."""
        # Prefer hardware energy counter delta
        if self._use_energy_counter:
            end = getattr(self, "_end_energy_mj", None)
            if end is not None and self._start_energy_mj is not None:
                delta_mj = end - self._start_energy_mj
                return (delta_mj / 1000.0) / 3600.0  # mJ → J → Wh
            return 0.0

        # Fallback: trapezoidal integration of power samples
        if len(self._timestamps) < 2:
            return 0.0
        total_joules = 0.0
        for i in range(1, len(self._timestamps)):
            dt = self._timestamps[i] - self._timestamps[i - 1]
            avg_power = (self._power_readings[i] + self._power_readings[i - 1]) / 2
            total_joules += avg_power * dt
        return total_joules / 3600.0


def _token_energy_wh(prompt_tokens: int, completion_tokens: int,
                     param_billions: float = 8.0) -> float:
    """Estimate LLM inference energy from token counts and model size.

    Per-token rates are calibrated from published measurements on 8B models
    (Luccioni et al. 2024, Moore et al. 2025) and scale with parameter
    count via a power law (exponent ~1.3), consistent with empirical data:

        6.7B → 0.082 Wh,  8B → 0.093 Wh,  175B → 4.0 Wh,  405B → 17.3 Wh

    Uses *active* parameter count (see ``_parse_param_billions``), so MoE
    models are scaled by their per-token active params, not total.

    Args:
        prompt_tokens: Number of input (prefill) tokens — batched, cheaper.
        completion_tokens: Number of output (decode) tokens — sequential, costlier.
        param_billions: Active model parameters in billions (e.g. 8.0, 72.0).
    """
    scale = (param_billions / _BASE_MODEL_PARAMS_B) ** _ENERGY_SCALING_EXPONENT
    return (prompt_tokens * _BASE_WH_PER_PROMPT_TOKEN * scale
            + completion_tokens * _BASE_WH_PER_COMPLETION_TOKEN * scale)


class EnergyTracker:
    """Track energy consumed by a RAG query across distributed services.

    Supports three measurement strategies (in priority order):

    **Local mode** (all models loaded in this process):
      Uses NVML hardware counters or nvidia-smi power sampling to measure
      actual GPU energy on this node.

    **Remote mode** (vLLM + embedding + reranker on separate RunAI jobs):
      1. **DCGM** — if a DCGM exporter URL is configured, polls real GPU
         power in a background thread during the query.
      2. **Token-based** — uses ``prompt_tokens`` and ``completion_tokens``
         from the vLLM response with calibrated per-token energy rates.
      3. **Time-based** (fallback) — estimates from wall-clock time × TDP.

      Embedding/reranker energy comes from server-reported values when
      available, otherwise estimated from retrieval time.

    Usage::

        tracker = EnergyTracker(is_remote=True, vllm_base_url="http://...")
        tracker.start()
        # ... run query ...
        wh = tracker.stop(elapsed_s, timing=result.timing)
    """

    def __init__(self, *, is_remote: bool = False,
                 llm_model: str = "", embed_model: str = "",
                 vllm_base_url: str = ""):
        self._is_remote = is_remote
        self._method = "estimate"

        # Power fractions derived from model size (falls back to defaults)
        llm_params = _parse_param_billions(llm_model)
        embed_params = _parse_param_billions(embed_model) if embed_model else 1.0
        self._llm_params_b = llm_params  # stored for token-based scaling
        self._llm_power_frac = _model_power_fraction(llm_params)
        self._embed_power_frac = _model_power_fraction(embed_params)

        # Remote-mode: try DCGM exporter for real GPU power measurement
        self._dcgm: DCGMPowerSampler | None = None
        if is_remote and _DCGM_URL:
            self._dcgm = DCGMPowerSampler(_DCGM_URL, interval=0.5)

        # Local-mode: try direct GPU measurement on this node
        self._nvml: NVMLEnergyCounter | None = None
        self._power_monitor: GPUPowerMonitor | None = None
        if not is_remote:
            self._nvml = NVMLEnergyCounter()
            if not self._nvml.available:
                self._nvml = None
                self._power_monitor = GPUPowerMonitor(interval=0.5)

    @property
    def method(self) -> str:
        """Return the measurement method used.

        One of: 'nvml', 'power_sampling', 'dcgm', 'server_reported',
        'token_based', or 'estimate'.
        """
        return self._method

    def start(self) -> None:
        if self._is_remote:
            if self._dcgm and self._dcgm.available:
                self._dcgm.start()
                self._method = "dcgm"
            else:
                self._method = "server_reported"
            return
        if self._nvml:
            self._nvml.start()
            self._method = "nvml"
        elif self._power_monitor and self._power_monitor.available:
            self._power_monitor.start()
            self._method = "power_sampling"
        else:
            self._method = "estimate"

    def stop(self, elapsed_s: float = 0.0, timing: dict | None = None) -> float:
        """Stop measurement and return total energy in Watt-hours.

        Args:
            elapsed_s: Total wall-clock seconds for the query.
            timing: The ``result.timing`` dict from the pipeline. In remote
                mode this contains ``embed_energy_wh`` and ``reranker_energy_wh``
                reported by the servers, plus token counts for estimation.
        """
        # --- Local mode: direct GPU measurement ---
        if self._method == "nvml" and self._nvml:
            per_gpu = self._nvml.stop()
            if per_gpu:
                return sum(per_gpu.values())
        elif self._method == "power_sampling" and self._power_monitor:
            self._power_monitor.stop()
            wh = self._power_monitor.energy_wh
            if wh > 0:
                return wh

        # --- DCGM: real power measurement from cluster GPU ---
        if self._method == "dcgm" and self._dcgm:
            self._dcgm.stop()
            dcgm_wh = self._dcgm.energy_wh
            if dcgm_wh > 0:
                return dcgm_wh
            # DCGM was configured but returned no data — fall through

        # --- Remote mode: aggregate server-reported + vLLM estimate ---
        if timing:
            embed_wh = timing.get("embed_energy_wh", 0.0)
            reranker_wh = timing.get("reranker_energy_wh", 0.0)
            retrieval_s = timing.get("retrieval_s", 0.0)
            gen_s = timing.get("generation_s", 0.0)

            # vLLM energy: prefer token-based estimation from actual usage
            prompt_tokens = timing.get("llm_prompt_tokens", 0)
            completion_tokens = timing.get("llm_completion_tokens", 0)
            if prompt_tokens > 0 or completion_tokens > 0:
                vllm_wh = _token_energy_wh(prompt_tokens, completion_tokens, self._llm_params_b)
                self._method = "token_based"
            else:
                # Fallback: time-based estimation
                vllm_wh = (_GPU_TDP_WATTS * self._llm_power_frac * _GPU_UTIL * gen_s) / 3600.0
                self._method = "estimate"

            total = embed_wh + reranker_wh + vllm_wh
            # When servers don't report energy, estimate retrieval energy from
            # retrieval time.  Only a fraction of wall-clock retrieval time is
            # actual GPU compute (the rest is network, CPU search, API calls).
            if embed_wh == 0 and reranker_wh == 0 and retrieval_s > 0:
                gpu_retrieval_s = retrieval_s * _REMOTE_RETRIEVAL_GPU_FRAC
                total += (_GPU_TDP_WATTS * self._embed_power_frac * _GPU_UTIL * gpu_retrieval_s) / 3600.0

            if embed_wh > 0 or reranker_wh > 0:
                # Server reported retrieval energy; keep vLLM method label
                if self._method == "estimate":
                    self._method = "server_reported"
            return total

        # Pure fallback: estimate everything from wall-clock time
        self._method = "estimate"
        return (_GPU_TDP_WATTS * _GPU_UTIL * elapsed_s) / 3600.0


def _format_energy(wh: float, *, split: bool = False) -> str | tuple[str, str]:
    """Human-friendly energy string (auto-scale kWh / Wh).

    Args:
        wh: Energy in watt-hours.
        split: If True, return (value, unit) tuple for st.metric display.
    """
    if wh >= 1000.0:
        val, unit = f"{wh / 1000:.2f}", "kWh"
    elif wh >= 1.0:
        val, unit = f"{wh:.2f}", "Wh"
    else:
        # Sub-1 Wh: show enough decimal places to be meaningful
        val, unit = f"{wh:.4f}", "Wh"
    if split:
        return val, unit
    return f"{val} {unit}"


# ---------------------------------------------------------------------------
# Trusted metric comparisons for contextualizing numbers in LLM responses
# ---------------------------------------------------------------------------
METRIC_COMPARISONS = """
When your answer includes specific numeric metrics (energy, water, emissions, etc.), \
add a brief real-world comparison in parentheses to help non-expert readers. Use ONLY \
the trusted comparisons below — do not invent your own:

Energy:
- 1 MWh ≈ monthly electricity for ~1 average US home
- 1 GWh (1,000 MWh) ≈ annual electricity for ~100 US homes
- 1 kWh ≈ running a window AC unit for ~1 hour
- 1 TWh (1,000,000 MWh) ≈ annual electricity for ~100,000 US homes

Water:
- 1,000 liters ≈ filling ~6 standard bathtubs
- 1 million liters ≈ about half an Olympic swimming pool
- 1 gallon ≈ 3.8 liters

Carbon emissions:
- 1 metric ton CO2 ≈ one passenger's round-trip flight from New York to London
- 1 kg CO2 ≈ driving ~2.5 miles in an average gasoline car
- 500 metric tons CO2 ≈ annual emissions of ~55 average US households

Example: "Training consumed 1,287 MWh of energy (roughly enough to power 1,287 US homes for a month) [luccioni2025c]."

Only add comparisons for the most important metrics — do not clutter every number. \
If a metric does not map to any comparison above, omit the parenthetical.
""".strip()

# ---------------------------------------------------------------------------
# Prompts (shared with run_experiment.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = ("""
You must answer strictly based on the provided context snippets.
Do NOT use external knowledge or assumptions.
If the context does not clearly support an answer, you must output the literal string "is_blank" for both answer_value and ref_id.
For True/False questions, you MUST output "1" for True and "0" for False in answer_value. Do NOT output the words "True" or "False".
IMPORTANT: When you use information from a context snippet, you MUST cite it using the cite_as label from the context header in [Author et al., Year] format (e.g., [Luccioni et al., 2025]). Do NOT use raw ref_ids or numeric citations like [1]. Include every cited ref_id in the ref_id list. Never give an answer without citing the source.

""" + METRIC_COMPARISONS).strip()

SYSTEM_PROMPT_BEST_GUESS = ("""
You must answer based on the provided context snippets.
If the context strongly supports an answer, answer normally.
If the context only partially or weakly supports an answer, still provide your best guess but set confidence to "low".
Set confidence to "high" when the context clearly and directly answers the question.
For True/False questions, you MUST output "1" for True and "0" for False in answer_value. Do NOT output the words "True" or "False".
IMPORTANT: Cite sources using the cite_as label from the context header in [Author et al., Year] format (e.g., [Luccioni et al., 2025]). Do NOT use raw ref_ids or numeric citations like [1].

""" + METRIC_COMPARISONS).strip()

SYSTEM_PROMPT_RESEARCH = ("""
You are a scientific research assistant specializing in AI's environmental impact. \
You synthesize information from multiple academic sources to provide comprehensive, \
well-cited answers. Write in an academic but accessible style, similar to a literature \
review or survey paper. Always cite your sources using the cite_as label provided \
in the context headers in square brackets (e.g., [Luccioni et al., 2025]). When the \
context contains specific numbers, statistics, or claims, include them with citations. \
If the provided context is limited, acknowledge this explicitly rather than inventing \
information.

IMPORTANT: You must ALWAYS provide an answer. Never output "is_blank" for the answer \
or explanation fields. Even when the context only partially covers the question, \
synthesize whatever relevant information IS available and note any gaps. The user \
has opted into research mode specifically to get comprehensive answers.

""" + METRIC_COMPARISONS).strip()

USER_TEMPLATE = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use only the provided context; do not rely on external knowledge.
- If the context does not clearly support an answer, use "is_blank" for all fields except explanation.
- For unanswerable questions, set answer to "Unable to answer with confidence based on the provided documents."
- For True/False questions: answer_value must be "1" for True or "0" for False (not the words "True" or "False").
- Cite ALL relevant sources, not just one. Use [Author et al., Year] format from the cite_as labels in context headers. Do NOT use raw ref_ids or numeric citations like [1], [2].

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys, in this order:
- explanation          (1-3 sentences that directly answer the question. Cite ALL supporting sources in [Author et al., Year] format, e.g. "According to [Wu et al., 2021] and [Luccioni et al., 2025], ...". Do NOT use vague phrases like "the context states" or "the passage mentions". When citing specific metrics, add a brief real-world comparison in parentheses where applicable.)
- answer               (short natural-language response, e.g. "1438 lbs", "Water consumption", "TRUE")
- answer_value         (ONLY the numeric or categorical value, e.g. "1438", "Water consumption", "1"; or "is_blank")
- ref_id               (list of ALL document ids (ref_id values) from the context used as evidence, e.g. ["wu2021a", "luccioni2025c"]; or "is_blank". Include every source that supports the answer.)
- supporting_materials (verbatim quote, table reference, or figure reference from the cited document; or "is_blank")

JSON Answer:
""".strip()

USER_TEMPLATE_BEST_GUESS = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use the provided context as your primary source.
- If the context clearly answers the question, answer normally with confidence "high".
- If the context only partially relates, provide your best-effort answer with confidence "low".
- For True/False questions: answer_value must be "1" for True or "0" for False (not the words "True" or "False").
- Cite ALL relevant sources, not just one. Use [Author et al., Year] format from the cite_as labels in context headers. Do NOT use raw ref_ids or numeric citations like [1], [2].

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys, in this order:
- explanation          (1-3 sentences that directly answer the question. Cite ALL supporting sources in [Author et al., Year] format, e.g. "According to [Wu et al., 2021] and [Luccioni et al., 2025], ...". Do NOT use vague phrases like "the context states" or "the passage mentions". When citing specific metrics, add a brief real-world comparison in parentheses where applicable.)
- answer               (short natural-language response, e.g. "1438 lbs", "Water consumption", "TRUE")
- answer_value         (ONLY the numeric or categorical value, e.g. "1438", "Water consumption", "1"; or "is_blank")
- confidence           ("high" if the context clearly supports the answer, "low" if this is a best guess)
- ref_id               (list of ALL document ids from the context used as evidence, e.g. ["wu2021a", "luccioni2025c"]; or "is_blank". Include every source that supports the answer.)
- supporting_materials (verbatim quote, table reference, or figure reference from the cited document; or "is_blank")

JSON Answer:
""".strip()

USER_TEMPLATE_RESEARCH = """
You will be given a question and context snippets taken from academic papers and reports.
Each snippet has a header with cite_as (the citation label to use) and ref_id (the document id).

Your task is to write a comprehensive, multi-paragraph answer that synthesizes information \
from the provided sources, similar to a literature review. Follow these rules:

1. Write 3-6 paragraphs that thoroughly address the question from multiple angles.
2. EVERY sentence that states a fact, number, or claim MUST have an inline citation in \
square brackets immediately after, using the cite_as label from the context header. \
Example: "Training GPT-3 consumed approximately 1,287 MWh of energy [Luccioni et al., 2025]." \
Do NOT write any factual claim without a citation.
3. Do NOT use numeric citations like [1], [2], [5] — always use [Author et al., Year] format.
4. Include specific numbers, statistics, and quantitative findings when available in context. \
For key metrics, add a brief real-world comparison in parentheses so non-experts can grasp \
the scale (e.g., "3,500 MWh (enough to power ~3,500 US homes for a month)").
5. Organize your answer logically: start with a direct answer, then expand with details, \
comparisons, and implications.
6. If multiple sources discuss the same topic, synthesize and compare their findings, \
citing each: "While [Wu et al., 2021] reports X, [Luccioni et al., 2025] found Y."
7. End with a brief summary or outlook paragraph.
8. If the context is insufficient to fully answer the question, state what IS known from \
the context and note the gaps. You MUST still provide an answer — never use "is_blank" for \
the answer or explanation fields.

Example of properly cited text:
"The energy required to train GPT-3 was approximately 1,287 MWh (enough to power ~1,287 \
US homes for a month) [Luccioni et al., 2025], while GPT-4 training consumed an estimated \
50 GWh (annual electricity for ~4,500 US homes) [Islam et al., 2025]. Fine-tuning typically \
requires substantially less computation [Samsi et al., 2024], but its cumulative impact \
across organizations can be significant [Dodge et al., 2022]."

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys:
- explanation          (your multi-paragraph answer with inline [Author et al., Year] citations on EVERY factual sentence)
- answer               (one-sentence summary of the key finding)
- answer_value         (the most important numeric or categorical value, or "is_blank")
- ref_id               (list of ALL document ref_ids cited in your answer, e.g. ["luccioni2025c", "islam2025"])
- supporting_materials (key quotes or data points that support the answer, or "is_blank")

JSON Answer:
""".strip()


# ---------------------------------------------------------------------------
# OpenScholar-style self-feedback prompts (Asai et al., 2024 Section 2.2)
# ---------------------------------------------------------------------------

FEEDBACK_PROMPT_RESEARCH = """
You are reviewing your own response to a scientific question. Your task is to \
identify specific ways the response can be improved.

Question: {question}

Your current response:
{response}

Context used:
{context}

Analyze your response and generate feedback. For each issue, provide a specific, \
actionable improvement suggestion. If the response references missing information \
that might be found in additional papers, generate a retrieval query to find them.

Additionally, flag any numeric metrics (energy in MWh/GWh, water in liters, CO2 in \
tons/kg) that lack a real-world comparison in parentheses — readers need context to \
interpret these numbers.

Return STRICT JSON with the following keys:
- feedback      (list of 1-3 specific improvement suggestions, e.g. \
["Add quantitative energy figures for GPT-4 training", "Compare water usage across different model sizes"])
- retrieval_query  (a search query to find additional papers addressing the gaps, or "" if no retrieval needed)
- done          (true if the response is already comprehensive and needs no improvement, false otherwise)

JSON:
""".strip()

REFINEMENT_PROMPT_RESEARCH = """
You are refining your scientific response based on self-feedback. Incorporate the \
feedback to produce an improved, more comprehensive answer.

Question: {question}

Your previous response:
{response}

Feedback to address:
{feedback}

Original context:
{context}

Additional context from re-retrieval:
{new_context}

Instructions:
1. Address each feedback item by incorporating relevant information.
2. Maintain all existing correct citations and add new ones from the additional context.
3. Use [Author et al., Year] citation format (from the cite_as labels in context headers).
4. Keep the same academic style and structure, but improve coverage and accuracy.
5. Do NOT remove correct information from the previous response — only add or refine.
6. For key numeric metrics (energy, water, emissions), add a brief real-world comparison \
in parentheses so non-expert readers can grasp the scale. Use only these trusted benchmarks:
   - 1 MWh ≈ monthly electricity for ~1 average US home
   - 1 GWh ≈ annual electricity for ~100 US homes
   - 1,000 liters of water ≈ ~6 standard bathtubs
   - 1 million liters ≈ about half an Olympic swimming pool
   - 1 metric ton CO2 ≈ one passenger's round-trip flight NYC–London
   - 1 kg CO2 ≈ driving ~2.5 miles in an average gasoline car

Return STRICT JSON with the following keys:
- explanation          (your improved multi-paragraph answer with inline [Author et al., Year] citations)
- answer               (one-sentence summary of the key finding)
- answer_value         (the most important numeric or categorical value, or "is_blank")
- ref_id               (list of ALL document ref_ids cited in your answer)
- supporting_materials (key quotes or data points that support the answer, or "is_blank")

JSON Answer:
""".strip()

# ---------------------------------------------------------------------------
# Citation verification prompt (lightweight post-hoc pass)
# Only triggered when the LLM answer lacks inline [ref_id] citations.
# Based on OpenScholar Section 2.2 step 3: Citation Verification.
# ---------------------------------------------------------------------------
CITATION_VERIFY_PROMPT = """
You are a citation verification assistant. The explanation below answers a scientific \
question but is missing inline citations or has incorrect numeric citations like [1], [5]. \
You MUST attribute every factual claim to a source using [Author et al., Year] notation.

IMPORTANT RULES:
1. Use ONLY the cite_as labels from the source list below for citations. \
Valid citation labels include: {example_ref_ids}.
2. Do NOT use numeric citations like [1], [2], [5] — these are WRONG. Use the \
[Author et al., Year] format from the cite_as labels.
3. EVERY sentence that states a fact, statistic, or claim MUST end with at least one \
citation in brackets. Sentences without citations are NOT acceptable.
4. Do NOT change the meaning, remove sentences, or add new information — only insert citations.
5. If a sentence is supported by multiple sources, cite all of them: \
[Author1 et al., Year1][Author2 et al., Year2].

Example of correct citations:
"Training GPT-3 consumed approximately 1,287 MWh [{example_ref_ids}]."

Explanation to fix:
{explanation}

Available sources — use the cite_as labels for citations:
{sources}

Return ONLY the rewritten explanation with [Author et al., Year] citations inserted \
inline after every factual sentence. Do not wrap in JSON or add any other text.
""".strip()

# ---------------------------------------------------------------------------
# Claim-attribution verification prompt (OpenScholar Section 2.2 step 3)
# Checks whether cited statements are actually supported by their sources.
# Applied in BOTH research and standard modes.
# ---------------------------------------------------------------------------
ATTRIBUTION_VERIFY_PROMPT = """
You are a scientific claim attribution verifier. For each claim-citation pair below, \
determine whether the cited source passage ACTUALLY SUPPORTS the claim made in the sentence.

A claim is "supported" if the source passage contains information that directly backs \
the factual statement. A claim is "unsupported" if:
- The source says something different from the claim
- The source does not mention the topic at all
- The numbers/statistics in the claim do not match the source
- The claim extrapolates or inverts findings from the source

Claims to verify:
{claims_and_sources}

Return a JSON array with one object per claim:
[
  {{"claim": 1, "supported": true, "reason": "source confirms the energy figure"}},
  {{"claim": 2, "supported": false, "reason": "source mentions 500W not 300W"}}
]

Be STRICT: if the source does not clearly support the specific claim, mark it unsupported. \
Only mark "supported": true when the source genuinely backs the statement.

JSON:
""".strip()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIGS_DIR = _repo_root / "vendor" / "KohakuRAG" / "configs"

# Approximate 4-bit NF4 VRAM (GB) per config. Used for planning only.
VRAM_4BIT_GB = {
    "hf_qwen1_5b": 2, "hf_qwen3b": 3, "hf_qwen7b": 6, "hf_qwen14b": 10,
    "hf_qwen32b": 20, "hf_qwen72b": 40, "hf_llama3_8b": 6, "hf_gemma2_9b": 7,
    "hf_gemma2_27b": 17, "hf_mixtral_8x7b": 26, "hf_mixtral_8x22b": 80,
    "hf_mistral7b": 6, "hf_phi3_mini": 3, "hf_qwen3_30b_a3b": 18,
    "hf_qwen3_next_80b_a3b": 40, "hf_qwen3_next_80b_a3b_thinking": 40,
    "hf_olmoe_1b7b": 4, "hf_qwen1_5_110b": 60, "hf_openscholar_8b": 6,
}
EMBEDDER_OVERHEAD_GB = 3  # Jina V4 embedder + store + misc
PRECISION_MULTIPLIER = {"4bit": 1.0, "bf16": 4.0, "fp16": 4.0, "auto": 4.0}

# ---------------------------------------------------------------------------
# Metadata URL lookup  (ref_id → URL from metadata.csv)
# ---------------------------------------------------------------------------
_METADATA_CSV = _repo_root / "data" / "metadata.csv"

def _load_metadata_urls() -> dict[str, str]:
    """Build a ref_id → url mapping from metadata.csv."""
    mapping: dict[str, str] = {}
    if not _METADATA_CSV.exists():
        return mapping
    with open(_METADATA_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            doc_id = row.get("id", "").strip()
            url = row.get("url", "").strip()
            if doc_id and url:
                mapping[doc_id] = url
    return mapping

METADATA_URLS: dict[str, str] = _load_metadata_urls()


def _build_corpus_summary() -> dict:
    """Summarize the corpus from metadata.csv for the welcome message."""
    if not _METADATA_CSV.exists():
        return {"count": 0, "types": {}, "year_range": "", "titles": []}

    titles = []
    types: dict[str, int] = {}
    years = []
    with open(_METADATA_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            title = row.get("title", "").strip()
            doc_type = row.get("type", "unknown").strip()
            year = row.get("year", "").strip()
            if title:
                titles.append(title)
            types[doc_type] = types.get(doc_type, 0) + 1
            if year.isdigit():
                years.append(int(year))

    year_range = f"{min(years)}\u2013{max(years)}" if years else ""
    return {
        "count": len(titles),
        "types": types,
        "year_range": year_range,
        "titles": titles,
    }


CORPUS_SUMMARY: dict = _build_corpus_summary()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _debug(msg: str) -> None:
    """Print debug info to terminal and, if debug mode is on, to the Streamlit UI."""
    logger.info(msg)
    print(f"[DEBUG] {msg}", flush=True)


def discover_configs() -> dict[str, Path]:
    """Find all hf_*.py config files and return {display_name: path}."""
    return {p.stem: p for p in sorted(CONFIGS_DIR.glob("hf_*.py"))}


def load_config(config_path: Path) -> dict:
    """Load a Python config file into a dict."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {}
    for key in [
        "db", "table_prefix", "questions", "output", "metadata",
        "llm_provider", "top_k", "planner_max_queries", "deduplicate_retrieval",
        "rerank_strategy", "top_k_final", "retrieval_threshold",
        "max_retries", "max_concurrent",
        "embedding_model", "embedding_dim", "embedding_task", "embedding_model_id",
        "hf_model_id", "hf_dtype", "hf_max_new_tokens", "hf_temperature",
    ]:
        if hasattr(module, key):
            config[key] = getattr(module, key)
    return config


def estimate_vram(config_name: str, precision: str) -> float:
    """Estimate VRAM (GB) for a model at given precision."""
    base = VRAM_4BIT_GB.get(config_name, 8)
    return base * PRECISION_MULTIPLIER.get(precision, 1.0)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def get_gpu_info() -> dict:
    """Detect GPU count, names, and free VRAM per GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu_count": 0, "gpus": [], "total_free_gb": 0}
    except ImportError:
        return {"gpu_count": 0, "gpus": [], "total_free_gb": 0}

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "total_gb": total / (1024**3),
            "free_gb": free / (1024**3),
        })
    total_free = sum(g["free_gb"] for g in gpus)
    return {"gpu_count": len(gpus), "gpus": gpus, "total_free_gb": total_free}


def plan_ensemble(config_names: list[str], precision: str, gpu_info: dict) -> dict:
    """Decide parallel vs sequential execution based on available VRAM.

    Returns:
        {"mode": "parallel"|"sequential"|"error", "model_vrams": [...], ...}
    """
    model_vrams = [estimate_vram(n, precision) for n in config_names]
    total_needed = sum(model_vrams) + EMBEDDER_OVERHEAD_GB
    total_free = gpu_info["total_free_gb"]

    if total_free == 0:
        return {"mode": "error", "model_vrams": model_vrams,
                "reason": "No GPU detected"}

    max_single_gpu = max(g["free_gb"] for g in gpu_info["gpus"])
    largest_model = max(model_vrams)

    if largest_model + EMBEDDER_OVERHEAD_GB > max_single_gpu:
        return {"mode": "error", "model_vrams": model_vrams,
                "reason": (f"Largest model needs ~{largest_model + EMBEDDER_OVERHEAD_GB:.0f} GB "
                           f"but largest GPU only has {max_single_gpu:.0f} GB free")}

    if total_needed <= total_free:
        return {"mode": "parallel", "model_vrams": model_vrams}
    return {"mode": "sequential", "model_vrams": model_vrams}


# ---------------------------------------------------------------------------
# Pipeline init
# ---------------------------------------------------------------------------
def _load_shared_resources(config: dict) -> tuple[JinaV4EmbeddingModel, KVaultNodeStore, Path]:
    """Load embedder and vector store from config.

    Returns:
        Tuple of (embedder, store, db_path).  The *db_path* is forwarded so
        callers can open an :class:`ImageStore` on the same database file.
    """
    embedding_dim = config.get("embedding_dim", 1024)
    embedding_task = config.get("embedding_task", "retrieval")
    db_path = _resolve_vector_db_path(config)
    db_path = _ensure_writable_db(db_path)
    table_prefix = config.get("table_prefix", "wattbot_jv4")

    _debug(
        f"Loading shared resources:\n"
        f"  db_path       = {db_path} (exists={db_path.exists()})\n"
        f"  table_prefix  = {table_prefix}\n"
        f"  embedding_dim = {embedding_dim}\n"
        f"  embedding_task= {embedding_task}"
    )

    embedder = JinaV4EmbeddingModel(
        task=embedding_task,
        truncate_dim=embedding_dim,
    )
    _debug(f"Embedder loaded: dimension={embedder.dimension}")

    store = KVaultNodeStore(
        db_path,
        table_prefix=table_prefix,
        dimensions=embedding_dim,
        paragraph_search_mode="averaged",
    )
    _debug(
        f"Store opened: dimensions={store._dimensions}, "
        f"vec_count={store._vectors.info().get('count', '?')}"
    )
    return embedder, store, db_path


def _load_chat_model(config: dict, precision: str) -> HuggingFaceLocalChatModel:
    """Create a HuggingFaceLocalChatModel from config."""
    return HuggingFaceLocalChatModel(
        model=config.get("hf_model_id", "Qwen/Qwen2.5-7B-Instruct"),
        system_prompt=SYSTEM_PROMPT,
        dtype=precision,
        max_new_tokens=config.get("hf_max_new_tokens", 512),
        temperature=config.get("hf_temperature", 0.2),
        max_concurrent=config.get("max_concurrent", 2),
    )


def _unload_chat_model(chat_model: HuggingFaceLocalChatModel) -> None:
    """Free GPU memory from a loaded model."""
    import torch
    if hasattr(chat_model, "_model"):
        del chat_model._model
    if hasattr(chat_model, "_tokenizer"):
        del chat_model._tokenizer
    del chat_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



@st.cache_resource(show_spinner="Loading cross-encoder reranker...")
def _load_cross_encoder(model_name: str) -> CrossEncoderReranker:
    """Load and cache a cross-encoder reranker model."""
    return CrossEncoderReranker(model_name)


def _apply_retrieval_enhancements(
    pipeline: RAGPipeline,
    *,
    enable_cross_encoder: bool = False,
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
    enable_semantic_scholar: bool = False,
    s2_top_k: int = 5,
    enable_query_planner: bool = False,
    planner_max_queries: int = 3,
) -> None:
    """Configure cross-encoder reranker, Semantic Scholar, and query planner on an existing pipeline."""
    # Cross-encoder reranker — use remote service if URL is set, else local
    if enable_cross_encoder:
        if RERANKER_SERVICE_URL and REMOTE_AVAILABLE:
            pipeline._cross_encoder = RemoteCrossEncoderReranker(
                base_url=RERANKER_SERVICE_URL,
            )
        elif RERANKER_AVAILABLE:
            pipeline._cross_encoder = _load_cross_encoder(cross_encoder_model)
        else:
            pipeline._cross_encoder = None
    else:
        pipeline._cross_encoder = None

    # Semantic Scholar retriever
    if enable_semantic_scholar:
        pipeline._semantic_scholar = SemanticScholarRetriever(max_results=s2_top_k)
        pipeline._semantic_scholar_top_k = s2_top_k
    else:
        pipeline._semantic_scholar = None

    # LLM query planner — expands a single question into diverse retrieval queries
    if enable_query_planner:
        pipeline._planner = LLMQueryPlanner(
            chat_model=pipeline._chat,
            max_queries=planner_max_queries,
        )
        # Enable dedup + reranking to handle multi-query overlap
        pipeline._deduplicate = True
        pipeline._rerank_strategy = pipeline._rerank_strategy or "combined"
        _debug(f"LLM query planner enabled (max_queries={planner_max_queries})")
    else:
        pipeline._planner = SimpleQueryPlanner()


@st.cache_resource(show_spinner="Loading model and vector store...")
def init_single_pipeline(config_name: str, precision: str) -> RAGPipeline:
    """Load a single-model pipeline. Cached across reruns."""
    config = load_config(CONFIGS_DIR / f"{config_name}.py")
    embedder, store, db_path = _load_shared_resources(config)
    chat_model = _load_chat_model(config, precision)
    image_store = ImageStore(db_path)
    return RAGPipeline(
        store=store, embedder=embedder, chat_model=chat_model, planner=None,
        image_store=image_store,
    )


@st.cache_resource(show_spinner="Loading ensemble models...")
def init_ensemble_parallel(config_names: tuple[str, ...], precision: str) -> dict[str, RAGPipeline]:
    """Load all ensemble models into memory (parallel mode). Cached."""
    # Use first config for shared resources (db/embedder are the same across configs)
    ref_config = load_config(CONFIGS_DIR / f"{config_names[0]}.py")
    # db_path is needed for ImageStore but not used directly here — see below
    embedder, store, db_path = _load_shared_resources(ref_config)
    image_store = ImageStore(db_path)

    pipelines = {}
    for name in config_names:
        config = load_config(CONFIGS_DIR / f"{name}.py")
        chat_model = _load_chat_model(config, precision)
        pipelines[name] = RAGPipeline(
            store=store, embedder=embedder, chat_model=chat_model, planner=None,
            image_store=image_store,
        )
    return pipelines


@st.cache_resource(show_spinner="Loading embedder and vector store...")
def init_shared_only() -> tuple[JinaV4EmbeddingModel, KVaultNodeStore, ImageStore]:
    """Load only the embedder + store (for sequential ensemble). Cached."""
    ref_config = load_config(next(CONFIGS_DIR.glob("hf_*.py")))
    embedder, store, db_path = _load_shared_resources(ref_config)
    image_store = ImageStore(db_path)
    return embedder, store, image_store


# ---------------------------------------------------------------------------
# Vector DB path resolution
# ---------------------------------------------------------------------------
# Common PPVC / NFS mount paths where the pre-built vector DB may live.
_PPVC_DB_CANDIDATES = [
    "/tmp/vectordb/wattbot_jinav4.db",  # RunAI startup copies DB here
    "/wattbot-data/embeddings/wattbot_jinav4.db",
    "/home/jovyan/work/KohakuRAG_UI/data/embeddings/wattbot_jinav4.db",
    "/home/jovyan/work/wattbot-data/embeddings/wattbot_jinav4.db",
]


def _resolve_vector_db_path(ref_config: dict) -> Path:
    """Find the vector DB, checking VECTOR_DB_PATH env, then PPVC candidates, then config default."""
    # 1. Explicit env var (highest priority)
    db_env = os.environ.get("VECTOR_DB_PATH")
    if db_env:
        p = Path(db_env)
        if p.exists():
            _debug(f"Using VECTOR_DB_PATH from env: {p}")
            return p
        _debug(f"WARNING: VECTOR_DB_PATH={db_env} does not exist, trying fallbacks...")

    # 2. Config-derived path (repo-relative)
    db_raw = ref_config.get("db", "data/embeddings/wattbot_jinav4.db")
    config_path = _repo_root / db_raw.removeprefix("../").removeprefix("../")
    if config_path.exists():
        _debug(f"Using config-derived DB path: {config_path}")
        return config_path

    # 3. Check if data/embeddings is a symlink to a PPVC
    symlink_target = _repo_root / "data" / "embeddings" / "wattbot_jinav4.db"
    if symlink_target.exists():
        _debug(f"Using symlinked DB path: {symlink_target}")
        return symlink_target

    # 4. Auto-discover from common PPVC mount points
    for candidate in _PPVC_DB_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            _debug(f"Auto-discovered DB at PPVC path: {p}")
            return p

    # 5. Fall through to config default (will create empty DB — warn loudly)
    _debug(
        f"WARNING: Vector DB not found! Checked:\n"
        f"  - VECTOR_DB_PATH env var: {db_env or '(not set)'}\n"
        f"  - Config path: {config_path}\n"
        f"  - PPVC candidates: {_PPVC_DB_CANDIDATES}\n"
        f"  The app will start with an EMPTY vector store (0 documents).\n"
        f"  Set VECTOR_DB_PATH=/path/to/wattbot_jinav4.db to fix this."
    )
    return config_path


def _ensure_writable_db(db_path: Path) -> Path:
    """If the DB file is on a read-only filesystem, copy it to a writable temp dir.

    KVaultNodeStore writes metadata on open (auto_pack, META_KEY), so the DB
    must be writable. Read-only PPVCs will cause failures or silent empty DBs.
    """
    if not db_path.exists():
        return db_path  # Nothing to copy; KVault will create a new empty DB

    # Test if the directory is writable
    try:
        test_file = db_path.parent / ".write_test"
        test_file.touch()
        test_file.unlink()
        return db_path  # Writable, use as-is
    except OSError:
        pass

    # Read-only filesystem — copy to /tmp
    import shutil
    tmp_dir = Path("/tmp/wattbot_db_cache")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_db = tmp_dir / db_path.name

    # Only copy if not already cached (or source is newer)
    if not tmp_db.exists() or db_path.stat().st_mtime > tmp_db.stat().st_mtime:
        _debug(f"Copying DB from read-only volume to {tmp_db} ...")
        shutil.copy2(db_path, tmp_db)
        _debug(f"DB copy complete ({tmp_db.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        _debug(f"Using cached writable DB copy at {tmp_db}")

    return tmp_db


# ---------------------------------------------------------------------------
# Remote pipeline init (vLLM + embedding server)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to remote inference services...")
def init_remote_pipeline(
    vllm_url: str,
    vllm_model: str,
    embedding_url: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    db_path_override: str | None = None,
    table_prefix_override: str | None = None,
) -> RAGPipeline:
    """Create a pipeline backed by remote vLLM and embedding services.

    The vector store (SQLite) is still read locally — only the LLM and
    embedding model run on separate GPU pods.

    Args:
        db_path_override: If provided, use this DB path instead of the
            config-derived default. Enables KB switching from the sidebar.
        table_prefix_override: If provided, use this table prefix instead
            of reading it from the config file.
    """
    if not REMOTE_AVAILABLE:
        raise ImportError(
            "Remote mode requires 'openai' and 'httpx' packages. "
            "Install with: pip install openai httpx"
        )

    _debug(f"Connecting to vLLM at {vllm_url} (model={vllm_model})")
    chat_model = VLLMChatModel(
        base_url=vllm_url,
        model=vllm_model,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    _debug(f"Connecting to embedding server at {embedding_url}")
    embedder = RemoteEmbeddingModel(base_url=embedding_url)

    # Vector store is local (lightweight SQLite reads, no GPU needed).
    if db_path_override:
        db_path = Path(db_path_override)
        table_prefix = table_prefix_override or _detect_table_prefix(str(db_path)) or "wattbot_jv4"
        _debug(f"Using sidebar-selected KB: {db_path} (prefix={table_prefix})")
    else:
        # VECTOR_DB_PATH env var overrides the config-derived path so the
        # Streamlit pod can point directly at the PPVC without symlinks.
        ref_config = load_config(next(CONFIGS_DIR.glob("hf_*.py")))
        table_prefix = ref_config.get("table_prefix", "wattbot_jv4")
        db_path = _resolve_vector_db_path(ref_config)

    embedding_dim = embedder.dimension

    # If the DB is on a read-only filesystem (e.g. PPVC), copy it to a
    # writable temp location because KVaultNodeStore writes metadata on open.
    db_path = _ensure_writable_db(db_path)

    _debug(
        f"Opening local vector store: {db_path} "
        f"(exists={db_path.exists()}, is_symlink={db_path.parent.is_symlink()})"
    )
    store = KVaultNodeStore(
        db_path,
        table_prefix=table_prefix,
        dimensions=embedding_dim,
        paragraph_search_mode="averaged",
    )
    _debug(
        f"Store opened: dimensions={store._dimensions}, "
        f"vec_count={store._vectors.info().get('count', '?')}"
    )

    image_store = ImageStore(db_path)
    return RAGPipeline(
        store=store, embedder=embedder, chat_model=chat_model, planner=None,
        image_store=image_store,
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def _run_qa_sync(
    pipeline: RAGPipeline,
    question: str,
    top_k: int,
    best_guess: bool = False,
    max_retries: int = 0,
    with_images: bool = False,
    top_k_images: int = 0,
    send_images_to_llm: bool = False,
    research_mode: bool = False,
    max_tokens_override: int = 0,
):
    """Run pipeline.run_qa synchronously, retrying on failures.

    Args:
        max_retries: Number of additional attempts after the first failure.
                     0 means no retries (single attempt).
        research_mode: If True, use detailed multi-paragraph prompt and system prompt.
        max_tokens_override: If > 0, temporarily override the chat model's max_tokens.
    """
    if research_mode:
        sys_prompt = SYSTEM_PROMPT_RESEARCH
        usr_template = USER_TEMPLATE_RESEARCH
    elif best_guess:
        sys_prompt = SYSTEM_PROMPT_BEST_GUESS
        usr_template = USER_TEMPLATE_BEST_GUESS
    else:
        sys_prompt = SYSTEM_PROMPT
        usr_template = USER_TEMPLATE
    # When both research_mode and best_guess are on, reinforce that the LLM
    # must always provide an answer (research prompt already says this, but
    # best_guess adds the explicit low-confidence fallback instruction).
    if research_mode and best_guess:
        sys_prompt += (
            "\nIf the context only partially or weakly supports an answer, "
            "still provide your best-effort synthesis but set confidence to \"low\"."
        )
    # Temporarily override max_tokens on the chat model if requested
    original_max_tokens = None
    if max_tokens_override > 0 and hasattr(pipeline._chat, '_max_tokens'):
        original_max_tokens = pipeline._chat._max_tokens
        pipeline._chat._max_tokens = max_tokens_override

    last_exc: Exception | None = None
    try:
        for attempt in range(max_retries + 1):
            loop = asyncio.new_event_loop()
            try:
                # Use self-feedback loop for research mode (OpenScholar-style)
                if research_mode:
                    return loop.run_until_complete(
                        pipeline.run_qa_with_feedback(
                            question,
                            system_prompt=sys_prompt,
                            user_template=usr_template,
                            feedback_prompt=FEEDBACK_PROMPT_RESEARCH,
                            refinement_prompt=REFINEMENT_PROMPT_RESEARCH,
                            max_feedback_rounds=2,
                            top_k=top_k,
                            with_images=with_images,
                            top_k_images=top_k_images,
                            send_images_to_llm=send_images_to_llm,
                        )
                    )
                else:
                    return loop.run_until_complete(
                        pipeline.run_qa(
                            question,
                            system_prompt=sys_prompt,
                            user_template=usr_template,
                            top_k=top_k,
                            with_images=with_images,
                            top_k_images=top_k_images,
                            send_images_to_llm=send_images_to_llm,
                        )
                    )
            except Exception as exc:
                last_exc = exc
                _debug(f"Attempt {attempt + 1}/{max_retries + 1} failed: {exc}")
                if attempt < max_retries:
                    time.sleep(1)  # brief pause before retry
            finally:
                loop.close()

        raise last_exc  # type: ignore[misc]
    finally:
        # Restore original max_tokens
        if original_max_tokens is not None:
            pipeline._chat._max_tokens = original_max_tokens


def run_single_query(
    pipeline: RAGPipeline, question: str, top_k: int,
    best_guess: bool = False, max_retries: int = 0,
    with_images: bool = False, top_k_images: int = 0,
    send_images_to_llm: bool = False,
    research_mode: bool = False, max_tokens_override: int = 0,
):
    """Run a single model query with post-hoc citation verification."""
    result = _run_qa_sync(
        pipeline, question, top_k,
        best_guess=best_guess, max_retries=max_retries,
        with_images=with_images, top_k_images=top_k_images,
        send_images_to_llm=send_images_to_llm,
        research_mode=research_mode, max_tokens_override=max_tokens_override,
    )

    # Post-hoc citation verification: if the explanation lacks inline
    # [ref_id] citations, run a lightweight LLM pass to insert them.
    # Conditional — skips the LLM call if citations are already present.
    try:
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                pipeline.verify_citations(result, CITATION_VERIFY_PROMPT)
            )
        finally:
            loop.close()
    except Exception as e:
        _debug(f"Citation verification skipped: {e}")

    # Claim-attribution verification (OpenScholar Section 2.2 step 3):
    # Check that each cited statement is actually supported by its source.
    # Runs in BOTH research and standard modes — removes unsupported citations
    # rather than leaving misleading attributions.
    try:
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                pipeline.verify_claim_attribution(result, ATTRIBUTION_VERIFY_PROMPT)
            )
        finally:
            loop.close()
    except Exception as e:
        _debug(f"Claim attribution verification skipped: {e}")

    return result


def run_ensemble_parallel_query(
    pipelines: dict[str, RAGPipeline], question: str, top_k: int,
    best_guess: bool = False, max_retries: int = 0,
    with_images: bool = False, top_k_images: int = 0,
    send_images_to_llm: bool = False,
) -> dict[str, object]:
    """Query all pre-loaded models concurrently."""
    results = {}
    for name, pipeline in pipelines.items():
        t0 = time.time()
        result = _run_qa_sync(
            pipeline, question, top_k,
            best_guess=best_guess, max_retries=max_retries,
            with_images=with_images, top_k_images=top_k_images,
            send_images_to_llm=send_images_to_llm,
        )
        results[name] = {"result": result, "time": time.time() - t0}
    return results


def run_ensemble_sequential_query(
    config_names: list[str],
    precision: str,
    question: str,
    top_k: int,
    progress_callback=None,
    best_guess: bool = False,
    max_retries: int = 0,
    with_images: bool = False,
    top_k_images: int = 0,
    send_images_to_llm: bool = False,
    enhancement_kwargs: dict | None = None,
) -> dict[str, object]:
    """Load each model one at a time, query, unload. Saves VRAM."""
    embedder, store, image_store = init_shared_only()
    results = {}

    for i, name in enumerate(config_names):
        if progress_callback:
            progress_callback(i, len(config_names), name)

        config = load_config(CONFIGS_DIR / f"{name}.py")
        chat_model = _load_chat_model(config, precision)
        pipeline = RAGPipeline(
            store=store, embedder=embedder, chat_model=chat_model, planner=None,
            image_store=image_store,
        )
        if enhancement_kwargs:
            _apply_retrieval_enhancements(pipeline, **enhancement_kwargs)

        t0 = time.time()
        result = _run_qa_sync(
            pipeline, question, top_k,
            best_guess=best_guess, max_retries=max_retries,
            with_images=with_images, top_k_images=top_k_images,
            send_images_to_llm=send_images_to_llm,
        )
        elapsed = time.time() - t0
        results[name] = {"result": result, "time": elapsed}

        # Free model memory before loading next
        _unload_chat_model(chat_model)
        del pipeline

    return results


# ---------------------------------------------------------------------------
# Ensemble aggregation
# ---------------------------------------------------------------------------
def aggregate_majority(answers: list[str]) -> str:
    """Most common answer. Ties go to first occurrence."""
    valid = [a for a in answers if a and a.strip() and a != "is_blank"]
    if not valid:
        return "is_blank"
    return Counter(valid).most_common(1)[0][0]


def aggregate_first_non_blank(answers: list[str]) -> str:
    """First non-blank answer in model order."""
    for a in answers:
        if a and a.strip() and a != "is_blank":
            return a
    return "is_blank"


def aggregate_refs(ref_lists: list) -> list[str]:
    """Union of all reference IDs across models."""
    all_refs = set()
    for refs in ref_lists:
        if isinstance(refs, list):
            all_refs.update(r for r in refs if r and r != "is_blank")
        elif isinstance(refs, str) and refs != "is_blank":
            try:
                parsed = json.loads(refs.replace("'", '"'))
                all_refs.update(parsed)
            except (json.JSONDecodeError, TypeError):
                all_refs.add(refs)
    return sorted(all_refs) if all_refs else []


def build_ensemble_answer(
    model_results: dict[str, object], strategy: str,
) -> dict:
    """Aggregate individual model results into an ensemble answer."""
    answers = []
    values = []
    explanations = []
    ref_lists = []

    for name, entry in model_results.items():
        ans = entry["result"].answer
        answers.append(ans.answer)
        values.append(ans.answer_value)
        explanations.append(ans.explanation)
        ref_lists.append(ans.ref_id)

    agg_fn = aggregate_majority if strategy == "majority" else aggregate_first_non_blank

    best_answer = agg_fn(answers)
    best_value = agg_fn(values)
    best_explanation = agg_fn(explanations)

    # Scope refs to runs that agree with the winning answer
    winning_refs = [r for a, r in zip(answers, ref_lists) if a == best_answer]

    return {
        "answer": best_answer,
        "answer_value": best_value,
        "explanation": best_explanation,
        "ref_id": aggregate_refs(winning_refs),
        "individual": {
            name: {
                "answer": entry["result"].answer.answer,
                "answer_value": entry["result"].answer.answer_value,
                "explanation": entry["result"].answer.explanation,
                "ref_id": entry["result"].answer.ref_id,
                "time": entry["time"],
                "raw_response": entry["result"].raw_response,
            }
            for name, entry in model_results.items()
        },
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def _detect_gpu_available() -> bool:
    """Return True if at least one CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False



def main():
    st.set_page_config(page_title="WattBot RAG", page_icon="lightning", layout="wide")
    # Capitalize "app" in the sidebar navigation (filename-derived label is lowercase)
    st.markdown(
        "<style>[data-testid='stSidebarNav'] li:first-child span"
        " { text-transform: capitalize; }</style>",
        unsafe_allow_html=True,
    )
    st.title("WattBot RAG Pipeline")

    is_remote = RAG_MODE == "remote"

    # ---- Remote mode: sidebar with model/KB selection ----
    if is_remote:
        mode = "Single model"  # vLLM serves one model
        precision = "auto"     # handled by vLLM
        ensemble_strategy = None
        selected_configs = []  # not used in remote mode
        gpu_info = {"gpu_count": 0, "gpus": [], "total_free_gb": 0}

        # Discover endpoints and knowledge bases
        endpoints = _list_vllm_endpoints()
        kb_options = _scan_knowledge_bases()

        with st.sidebar:
            st.header("Settings")
            st.caption(f"**Remote mode** — vLLM + embedding server")

            # -- LLM Server section --
            st.subheader("LLM Server")

            # Endpoint selectbox (only show if multiple endpoints available)
            if len(endpoints) > 1:
                endpoint_labels = []
                for e in endpoints:
                    status = "" if e["status"] == "online" else " [OFFLINE]"
                    model_info = e["model"] if e["model"] != "unknown" else "?"
                    endpoint_labels.append(f"{e['name']} ({model_info}){status}")
                selected_ep_idx = st.selectbox(
                    "vLLM endpoint", range(len(endpoint_labels)),
                    format_func=lambda i: endpoint_labels[i],
                    help="Select which vLLM inference server to use.",
                )
                active_endpoint = endpoints[selected_ep_idx]
            else:
                active_endpoint = endpoints[0] if endpoints else {
                    "name": "default", "url": VLLM_BASE_URL, "model": "unknown", "status": "offline",
                }

            # Custom URL override
            custom_vllm_url = st.text_input(
                "Custom vLLM URL",
                placeholder="http://my-server:8000/v1",
                help="Override the selected endpoint with a custom vLLM server URL. Leave empty to use the endpoint above.",
            )
            if custom_vllm_url.strip():
                actual_vllm_url = custom_vllm_url.strip()
                vllm_model = _detect_vllm_model(actual_vllm_url)
            else:
                actual_vllm_url = active_endpoint["url"]
                vllm_model = active_endpoint["model"]

            # Status display
            if vllm_model != "unknown":
                st.caption(f"Serving: **{vllm_model}**")
            else:
                st.caption(":orange[Server unreachable] — check URL or wait for startup.")
            st.caption(f"Embeddings: `{EMBEDDING_SERVICE_URL}`")

            # Models on shared PVC (collapsible)
            pvc_models = _scan_pvc_models()
            if pvc_models:
                with st.expander(f"Models on shared PVC ({len(pvc_models)})"):
                    for m in pvc_models:
                        marker = " **(active)**" if m["name"] == vllm_model else ""
                        st.caption(f"- `{m['name']}` ({m['size_gb']:.1f} GB){marker}")
            elif os.path.isdir(SHARED_MODELS_PATH):
                with st.expander("Models on shared PVC"):
                    st.caption("No models found on PVC.")
            # If PVC isn't mounted, don't show the expander at all

            # -- Knowledge Base section --
            st.divider()
            st.subheader("Knowledge Base")
            selected_kb = None
            if kb_options:
                # Include parent volume in label so researchers can distinguish
                # .db files with the same name from different data volumes.
                kb_labels = []
                for kb in kb_options:
                    vol = os.path.basename(os.path.dirname(kb["path"]))
                    kb_labels.append(f"{kb['name']} ({kb['size_mb']:.0f} MB) — {vol}/")
                selected_kb_idx = st.selectbox(
                    "Vector database", range(len(kb_labels)),
                    format_func=lambda i: kb_labels[i],
                    help="Select the knowledge base (vector DB) to query against. Each attached data volume is scanned automatically.",
                )
                selected_kb = kb_options[selected_kb_idx]
                st.caption(f"Path: `{selected_kb['path']}`")
                st.caption(f"Table prefix: `{selected_kb['table_prefix']}`")
            else:
                st.info(
                    "No knowledge bases found. Mount a data volume with an "
                    "`embeddings/` subdirectory, or set `VECTOR_DB_DIRS`."
                )

            # Track KB changes for chat history warning
            if "active_kb" not in st.session_state:
                st.session_state.active_kb = None
            _new_kb_path = selected_kb["path"] if selected_kb else None
            if (
                _new_kb_path
                and st.session_state.active_kb is not None
                and st.session_state.active_kb != _new_kb_path
                and st.session_state.get("messages")
            ):
                st.warning(
                    "Knowledge base changed. Previous answers reference the old corpus."
                )
                if st.button("Clear chat history"):
                    st.session_state.messages = []
                    st.session_state.query_count = 0
                    st.session_state.total_energy_wh = 0.0
                    st.session_state.active_kb = _new_kb_path
                    st.rerun()
            st.session_state.active_kb = _new_kb_path

            # -- Query settings (unchanged) --
            st.divider()
            top_k = st.slider("Retrieved chunks (top_k)", min_value=1, max_value=20, value=8)
            best_guess = st.toggle("Allow best-guess answers", value=True,
                                   help="When enabled, out-of-scope questions get a best-effort answer labelled as a guess.")
            research_mode = st.toggle(
                "Research mode", value=False,
                help=(
                    "Generate detailed, multi-paragraph answers with comprehensive "
                    "citations, similar to OpenScholar. Uses more tokens and takes "
                    "longer but produces higher quality, synthesis-style answers."
                ),
            )
            if research_mode:
                max_tokens_override = st.slider(
                    "Max generation tokens", min_value=512, max_value=4096,
                    value=2048, step=256,
                    help="Maximum tokens for the LLM response. Research mode needs more tokens for detailed answers.",
                )
            else:
                max_tokens_override = 0  # 0 = use default

            st.divider()
            max_retries = st.number_input(
                "Max retries", min_value=0, max_value=10, value=2,
                help="Maximum retry attempts when the LLM response cannot be parsed.",
            )

            # Images are always retrieved from PDF page renders
            with_images = True
            top_k_images = 3  # Additional images from dedicated image index
            send_images_to_llm = False  # Enable when using a vision-capable LLM

            st.divider()
            st.subheader("Retrieval enhancements")

            # Research mode auto-configures sensible defaults
            _s2_default = True if research_mode else False
            _s2_topk_default = 12 if research_mode else 5
            _qp_default = True if research_mode else False

            enable_semantic_scholar = st.toggle(
                "Semantic Scholar search", value=_s2_default,
                help=(
                    "Supplement local retrieval with paper abstracts from the "
                    "Semantic Scholar API (~200M papers)."
                ),
            )
            s2_top_k = _s2_topk_default
            if enable_semantic_scholar:
                s2_top_k = st.slider(
                    "S2 papers to include", min_value=1, max_value=20,
                    value=_s2_topk_default,
                    help="Number of external paper abstracts to append to context.",
                )
                # Show API key status
                _s2_key_set = bool(os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""))
                if _s2_key_set:
                    st.caption("S2 API key: **set** (100 req/s)")
                else:
                    st.caption(
                        "S2 API key: **not set** (1 req/s, frequent 429s). "
                        "Set `SEMANTIC_SCHOLAR_API_KEY` env var for better results."
                    )

            enable_query_planner = st.toggle(
                "Query expansion", value=_qp_default,
                help=(
                    "Use the LLM to expand your question into 3 diverse search "
                    "queries with different terminology. Improves retrieval recall "
                    "at the cost of ~1-2s extra latency."
                ),
            )
            planner_max_queries = 3
            if enable_query_planner and not research_mode:
                planner_max_queries = st.slider(
                    "Expansion queries", min_value=2, max_value=5, value=3,
                    help="Number of diverse queries to generate from your question.",
                )
            elif research_mode:
                planner_max_queries = 4  # more diverse queries for research

            if RERANKER_SERVICE_URL:
                enable_cross_encoder = st.toggle(
                    "Cross-encoder reranker", value=True,
                    help=(
                        "Use a cross-encoder model to rescore passages after "
                        "retrieval. Improves relevance ranking. Calls remote "
                        "reranker service."
                    ),
                )
            else:
                enable_cross_encoder = False
            cross_encoder_model = "BAAI/bge-reranker-v2-m3"

    # ---- Local mode: full sidebar with model selection ----
    else:
        configs = discover_configs()
        if not configs:
            st.error("No HF config files found in vendor/KohakuRAG/configs/")
            return

        with st.sidebar:
            st.header("Settings")
            mode = st.radio("Mode", ["Single model", "Ensemble"], horizontal=True)
            precision = st.selectbox("Precision", ["4bit", "bf16", "fp16", "auto"], index=0)
            top_k = st.slider("Retrieved chunks (top_k)", min_value=1, max_value=20, value=8)
            best_guess = st.toggle("Allow best-guess answers", value=True,
                                   help="When enabled, out-of-scope questions get a best-effort answer labelled as a guess.")
            research_mode = st.toggle(
                "Research mode", value=False,
                help=(
                    "Generate detailed, multi-paragraph answers with comprehensive "
                    "citations, similar to OpenScholar. Uses more tokens and takes "
                    "longer but produces higher quality, synthesis-style answers."
                ),
            )
            if research_mode:
                max_tokens_override = st.slider(
                    "Max generation tokens", min_value=512, max_value=4096,
                    value=2048, step=256,
                    help="Maximum tokens for the LLM response. Research mode needs more tokens for detailed answers.",
                )
            else:
                max_tokens_override = 0

            st.divider()
            max_retries = st.number_input(
                "Max retries", min_value=0, max_value=10, value=2,
                help="Maximum retry attempts when the LLM response cannot be parsed.",
            )

            # Images are always retrieved from PDF page renders
            with_images = True
            top_k_images = 3  # Additional images from dedicated image index
            send_images_to_llm = False  # Enable when using a vision-capable LLM

            st.divider()
            st.subheader("Retrieval enhancements")

            # Research mode auto-configures sensible defaults
            _s2_default = True if research_mode else False
            _s2_topk_default = 12 if research_mode else 5
            _qp_default = True if research_mode else False

            enable_semantic_scholar = st.toggle(
                "Semantic Scholar search", value=_s2_default,
                help=(
                    "Supplement local retrieval with paper abstracts from the "
                    "Semantic Scholar API (~200M papers). Useful for questions "
                    "that go beyond the curated corpus."
                ),
            )
            s2_top_k = _s2_topk_default
            if enable_semantic_scholar:
                s2_top_k = st.slider(
                    "S2 papers to include", min_value=1, max_value=20,
                    value=_s2_topk_default,
                    help="Number of external paper abstracts to append to context.",
                )
                # Show API key status
                _s2_key_set = bool(os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""))
                if _s2_key_set:
                    st.caption("S2 API key: **set** (100 req/s)")
                else:
                    st.caption(
                        "S2 API key: **not set** (1 req/s, frequent 429s). "
                        "Set `SEMANTIC_SCHOLAR_API_KEY` env var for better results."
                    )

            enable_query_planner = st.toggle(
                "Query expansion", value=_qp_default,
                help=(
                    "Use the LLM to expand your question into 3 diverse search "
                    "queries with different terminology. Improves retrieval recall "
                    "at the cost of ~1-2s extra latency."
                ),
            )
            planner_max_queries = 3
            if enable_query_planner and not research_mode:
                planner_max_queries = st.slider(
                    "Expansion queries", min_value=2, max_value=5, value=3,
                    help="Number of diverse queries to generate from your question.",
                )
            elif research_mode:
                planner_max_queries = 4

            enable_cross_encoder = False
            cross_encoder_model = "BAAI/bge-reranker-v2-m3"
            if RERANKER_AVAILABLE:
                enable_cross_encoder = st.toggle(
                    "Cross-encoder reranker", value=True,
                    help=(
                        "Use a cross-encoder model to rescore passages after "
                        "retrieval. Improves relevance ranking but adds ~1-2s latency. "
                        "Requires ~0.5-2 GB VRAM depending on model."
                    ),
                )
                if enable_cross_encoder:
                    cross_encoder_model = st.selectbox(
                        "Reranker model",
                        [
                            "BAAI/bge-reranker-v2-m3",
                            "BAAI/bge-reranker-large",
                            "OpenSciLM/OpenScholar_Reranker",
                        ],
                        index=0,
                        help=(
                            "bge-reranker-v2-m3: small & fast (~0.5 GB). "
                            "bge-reranker-large: better quality (~1.3 GB). "
                            "OpenScholar_Reranker: science-tuned (~1.2 GB)."
                        ),
                    )

            st.divider()
            config_list = list(configs.keys())

            if mode == "Single model":
                default_idx = config_list.index("hf_qwen7b") if "hf_qwen7b" in config_list else 0
                selected_config = st.selectbox("Model config", config_list, index=default_idx)
                selected_configs = [selected_config]
                ensemble_strategy = None
            else:
                selected_configs = st.multiselect(
                    "Ensemble models (pick 2+)", config_list,
                    default=["hf_qwen7b", "hf_llama3_8b"] if all(
                        c in config_list for c in ["hf_qwen7b", "hf_llama3_8b"]
                    ) else config_list[:2],
                )
                ensemble_strategy = st.selectbox(
                    "Aggregation", ["majority", "first_non_blank"],
                )

            # GPU info
            st.divider()
            gpu_info = get_gpu_info()
            if gpu_info["gpu_count"] > 0:
                st.caption(f"**{gpu_info['gpu_count']} GPU(s)** detected")
                for g in gpu_info["gpus"]:
                    st.caption(f"  GPU {g['index']}: {g['name']} — "
                               f"{g['free_gb']:.1f} / {g['total_gb']:.1f} GB free")
            else:
                st.caption("No GPU detected")

            if mode == "Ensemble" and len(selected_configs) >= 2:
                plan = plan_ensemble(selected_configs, precision, gpu_info)
                vram_list = [f"{n}: ~{v:.0f}GB" for n, v in
                             zip(selected_configs, plan["model_vrams"])]
                st.caption(f"VRAM: {', '.join(vram_list)}")
                if plan["mode"] == "parallel":
                    st.caption("Strategy: **parallel** (all models in memory)")
                elif plan["mode"] == "sequential":
                    st.caption("Strategy: **sequential** (load one at a time)")
                else:
                    st.warning(plan["reason"])

    # ---- Session energy accumulator (sidebar) ----
    # Use a placeholder so we can update it immediately after each query
    with st.sidebar:
        st.divider()
        st.subheader("Session energy")
        _energy_placeholder = st.empty()

    def _refresh_energy_display():
        """Re-render the session energy metrics into the sidebar placeholder."""
        with _energy_placeholder.container():
            _total_e = st.session_state.get("total_energy_wh", 0.0)
            _n_queries = st.session_state.get("query_count", 0)
            if _n_queries > 0:
                _e_val, _e_unit = _format_energy(_total_e, split=True)
                e_col1, e_col2 = st.columns(2)
                e_col1.metric(f"Total ({_e_unit})", _e_val)
                e_col2.metric("Queries", _n_queries)
                st.caption(f"Avg per query: {_format_energy(_total_e / _n_queries)}")

                # Bascom Hill comparison
                _steps = _total_e / _WH_PER_BASCOM_STEP
                _climbs = _steps / _BASCOM_HILL_STEPS
                if _climbs >= 1.0:
                    _climb_str = f"{_climbs:.1f} climbs"
                else:
                    _climb_str = f"{_climbs:.2f} climbs"
                _hill_col1, _hill_col2 = st.columns(2)
                _hill_col1.metric(
                    "Bascom Hill steps", f"~{_steps:.1f}",
                    help=(
                        "How much energy is that? Climbing one stair step burns "
                        "about 0.15 Wh of metabolic energy. Bascom Hill has 110 "
                        "steps from Park St to Bascom Hall — so your session "
                        "energy is equivalent to climbing that many steps up "
                        "the hill."
                    ),
                )
                _hill_col2.metric("Hill climbs", _climb_str)
            else:
                st.caption("No queries yet — energy will be tracked as you ask questions.")

    _refresh_energy_display()

    # ---- Validate ensemble selection ----
    if not is_remote and mode == "Ensemble" and len(selected_configs) < 2:
        st.info("Select at least 2 models for ensemble mode.")
        return

    # ---- Retrieval enhancement kwargs (shared across all modes) ----
    _enhancement_kwargs = dict(
        enable_cross_encoder=enable_cross_encoder,
        cross_encoder_model=cross_encoder_model,
        enable_semantic_scholar=enable_semantic_scholar,
        s2_top_k=s2_top_k,
        enable_query_planner=enable_query_planner,
        planner_max_queries=planner_max_queries,
    )

    # ---- Load pipelines ----
    try:
        if is_remote:
            pipeline = init_remote_pipeline(
                actual_vllm_url, vllm_model, EMBEDDING_SERVICE_URL,
                max_tokens=VLLM_MAX_TOKENS, temperature=VLLM_TEMPERATURE,
                db_path_override=selected_kb["path"] if selected_kb else None,
                table_prefix_override=selected_kb["table_prefix"] if selected_kb else None,
            )
            _apply_retrieval_enhancements(pipeline, **_enhancement_kwargs)
        elif mode == "Single model":
            pipeline = init_single_pipeline(selected_configs[0], precision)
            _apply_retrieval_enhancements(pipeline, **_enhancement_kwargs)
        elif mode == "Ensemble":
            plan = plan_ensemble(selected_configs, precision, gpu_info)
            if plan["mode"] == "error":
                st.error(f"Cannot run ensemble: {plan['reason']}")
                return
            if plan["mode"] == "parallel":
                ensemble_pipelines = init_ensemble_parallel(
                    tuple(selected_configs), precision,
                )
                for _p in ensemble_pipelines.values():
                    _apply_retrieval_enhancements(_p, **_enhancement_kwargs)
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        tb = traceback.format_exc()
        _debug(f"Load error:\n{tb}")
        with st.expander("Full traceback"):
            st.code(tb, language="python")
        return

    # ---- Chat interface ----
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_energy_wh" not in st.session_state:
        st.session_state.total_energy_wh = 0.0
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0

    # Welcome message (shown only when chat is empty)
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            s = CORPUS_SUMMARY
            type_parts = [
                f"{count} {t}{'s' if count != 1 else ''}"
                for t, count in s["types"].items()
            ]
            type_str = " and ".join(type_parts)

            st.markdown(
                f"**Welcome to WattBot!** Ask me questions about AI's environmental "
                f"impact \u2014 energy consumption, carbon emissions, water usage, "
                f"and sustainability.\n\n"
                f"My knowledge base contains **{s['count']} documents** ({type_str}) "
                f"spanning **{s['year_range']}**, covering topics like:"
            )
            for t in s["titles"][:5]:
                st.markdown(f"- *{t}*")
            if s["count"] > 5:
                st.markdown(f"- ...and {s['count'] - 5} more")
            st.page_link("pages/1_Corpus.py", label="\U0001F4DA Browse full corpus")

            st.caption(
                "Tip: Enable **Semantic Scholar search** in the sidebar to "
                "supplement the local corpus with abstracts from ~200M papers."
            )

    # Render history
    # Resolve image_store for figure display during history replay
    _hist_image_store = getattr(pipeline, "_image_store", None) if pipeline else None
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                details = msg.get("details", {})
                linked = _linkify_citations(
                    msg["content"],
                    ref_ids=details.get("ref_id"),
                    snippet_urls=details.get("snippet_urls"),
                )
                # Use regular markdown for long research-mode answers
                if len(linked) > 500 or "\n\n" in linked:
                    st.markdown(linked)
                else:
                    st.markdown(f"**{linked}**")
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant" and "details" in msg:
                _render_details(msg["details"], image_store=_hist_image_store)

    # User input
    if question := st.chat_input("Ask a question about the WattBot documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            t0 = time.time()
            energy_tracker = EnergyTracker(
                is_remote=is_remote,
                llm_model=vllm_model if is_remote else "",
                embed_model="jinaai/jina-embeddings-v4" if is_remote else "",
                vllm_base_url=VLLM_BASE_URL if is_remote else "",
            )
            energy_tracker.start()
            # Research mode uses more context for comprehensive answers
            effective_top_k = max(top_k, 15) if research_mode else top_k

            if is_remote or mode == "Single model":
                with st.spinner("Retrieving and generating..."):
                    try:
                        result = run_single_query(
                            pipeline, question, effective_top_k,
                            best_guess=best_guess, max_retries=max_retries,
                            with_images=with_images, top_k_images=top_k_images,
                            send_images_to_llm=send_images_to_llm,
                            research_mode=research_mode,
                            max_tokens_override=max_tokens_override,
                        )
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
                        tb = traceback.format_exc()
                        _debug(f"Pipeline error:\n{tb}")
                        with st.expander("Full traceback"):
                            st.code(tb, language="python")
                        return
                elapsed = time.time() - t0
                query_energy_wh = energy_tracker.stop(elapsed, timing=result.timing)
                st.session_state.total_energy_wh += query_energy_wh
                st.session_state.query_count += 1
                _refresh_energy_display()
                _display_single_result(
                    result, elapsed, pipeline=pipeline,
                    energy_wh=query_energy_wh, energy_method=energy_tracker.method,
                    chat_settings=_build_chat_settings_dict(locals()),
                )

            else:  # Ensemble (local mode only)
                try:
                    if plan["mode"] == "parallel":
                        with st.spinner(
                            f"Querying {len(selected_configs)} models in parallel..."
                        ):
                            model_results = run_ensemble_parallel_query(
                                ensemble_pipelines, question, effective_top_k,
                                best_guess=best_guess,
                                max_retries=max_retries,
                                with_images=with_images, top_k_images=top_k_images,
                                send_images_to_llm=send_images_to_llm,
                            )
                    else:
                        status = st.status(
                            f"Querying {len(selected_configs)} models sequentially...",
                            expanded=True,
                        )
                        def _progress(i, total, name):
                            status.update(label=f"[{i+1}/{total}] Loading {name}...")
                        model_results = run_ensemble_sequential_query(
                            selected_configs, precision, question, effective_top_k,
                            progress_callback=_progress,
                            best_guess=best_guess,
                            max_retries=max_retries,
                            with_images=with_images, top_k_images=top_k_images,
                            send_images_to_llm=send_images_to_llm,
                            enhancement_kwargs=_enhancement_kwargs,
                        )
                        status.update(label="Aggregating results...", state="complete")
                except Exception as e:
                    st.error(f"Ensemble error: {e}")
                    tb = traceback.format_exc()
                    _debug(f"Ensemble error:\n{tb}")
                    with st.expander("Full traceback"):
                        st.code(tb, language="python")
                    return

                elapsed = time.time() - t0
                # Ensemble is local-only, so local NVML/power sampling works;
                # no per-service timing to pass here.
                query_energy_wh = energy_tracker.stop(elapsed)
                st.session_state.total_energy_wh += query_energy_wh
                st.session_state.query_count += 1
                _refresh_energy_display()
                agg = build_ensemble_answer(model_results, ensemble_strategy)
                _display_ensemble_result(
                    agg, model_results, elapsed, ensemble_strategy,
                    energy_wh=query_energy_wh, energy_method=energy_tracker.method,
                    chat_settings=_build_chat_settings_dict(locals()),
                )


def _extract_confidence(raw_response: str) -> str:
    """Extract confidence field from raw JSON or bullet-list response."""
    # Try JSON first
    try:
        start = raw_response.index("{")
        end = raw_response.rindex("}") + 1
        data = json.loads(raw_response[start:end])
        return str(data.get("confidence", "")).strip().lower()
    except Exception:
        pass
    # Fallback: bullet-list format (- confidence   high/low)
    m = re.search(r"-\s*confidence\s{2,}(\S+)", raw_response)
    if m:
        return m.group(1).strip().strip('"').lower()
    return ""


def _humanize_ref_id(rid: str) -> str:
    """Convert a ref_id like ``luccioni2025c`` to ``Luccioni et al., 2025``.

    Expects the common ``<surname><4-digit-year>[suffix]`` pattern.
    Handles ``s2_`` prefix for Semantic Scholar references.
    Falls back to the raw id if the pattern doesn't match.
    """
    # Strip Semantic Scholar prefix for display
    display_rid = rid.removeprefix("s2_")
    m = re.match(r"([a-zA-Z]+)(\d{4})([a-z]?)", display_rid)
    if m:
        author = m.group(1).capitalize()
        year = m.group(2)
        suffix = m.group(3)  # e.g. "b" in luccioni2024b
        label = f"{author} et al., {year}{suffix}"
        if rid.startswith("s2_"):
            label += " [S2]"
        return label
    return rid


def _clean_ref_ids(ref_ids) -> list[str]:
    """Normalize ref_ids: always return a clean list, filtering out 'is_blank'."""
    if not ref_ids:
        return []
    if isinstance(ref_ids, str):
        if ref_ids == "is_blank":
            return []
        return [ref_ids]
    if isinstance(ref_ids, list):
        return [r for r in ref_ids if r and str(r) != "is_blank"]
    return []


def _linkify_citations(
    text: str,
    ref_ids=None,
    snippet_urls: dict[str, str] | None = None,
) -> str:
    """Replace citation references in *text* with clickable markdown links.

    Handles two citation styles:
    * Bracket-style:  ``[luccioni2025]`` — raw ref_id in brackets
    * Parenthetical:  ``(Luccioni et al., 2025)`` — humanized form in parens

    Converts raw ids to human-readable labels, inserts comma separators
    between adjacent citations, and looks up URLs from METADATA_URLS
    (primary) and snippet metadata (secondary).  LLM-provided ref_urls
    are intentionally ignored to prevent hallucinated URLs.
    """
    if not text:
        return text

    # Build URL map from verified sources only (snippet metadata, NOT LLM ref_urls)
    answer_urls: dict[str, str] = {}
    if snippet_urls:
        answer_urls.update(snippet_urls)
    clean_ids = _clean_ref_ids(ref_ids)

    # Build reverse map: "Luccioni et al., 2025" → first matching ref_id
    # so we can resolve parenthetical citations like (Luccioni et al., 2025)
    humanized_to_rid: dict[str, str] = {}
    all_rids = list(METADATA_URLS.keys()) + list(answer_urls.keys())
    for rid in clean_ids:
        if rid not in all_rids:
            all_rids.append(rid)
    for rid in all_rids:
        label = _humanize_ref_id(rid)
        if label != rid and label not in humanized_to_rid:
            humanized_to_rid[label] = rid

    def _replace_bracket(match: re.Match) -> str:
        inner = match.group(1)

        # Case 1: raw ref_id like "luccioni2025c"
        url = METADATA_URLS.get(inner) or answer_urls.get(inner)
        label = _humanize_ref_id(inner)
        if url:
            return f"[{label}]({url})"
        if label != inner:
            # No URL but we can humanize — check if it's already humanized
            return f"({label})"

        # Case 2: already-humanized "Author et al., Year" — resolve via reverse map
        rid = humanized_to_rid.get(inner)
        if rid:
            url = METADATA_URLS.get(rid) or answer_urls.get(rid)
            if url:
                return f"[{inner}]({url})"
            return f"({inner})"

        return match.group(0)

    # Normalise bold-wrapped citations: __Author et al., Year__ → [Author et al., Year]
    # Some LLMs emit markdown bold instead of brackets for inline citations.
    text = re.sub(
        r"__([A-Z][a-z]+(?:\s+et\s+al\.)?(?:,?\s*\d{4}[a-z]?))__",
        r"[\1]",
        text,
    )

    # Match [something] NOT already followed by '(' (avoids double-linking)
    text = re.sub(r"\[([^\]]+)\](?!\()", _replace_bracket, text)

    # Insert ", " between adjacent markdown links: ...](url)[... → ...](url), [...
    text = re.sub(r"\]\(([^)]+)\)\[", r"](\1), [", text)

    # Match parenthetical citations: (Author et al., Year)
    # Pattern: (Capitalized-word et al., 4-digit-year)
    def _replace_paren(match: re.Match) -> str:
        full = match.group(0)  # e.g. "(Luccioni et al., 2025)"
        inner = match.group(1)  # e.g. "Luccioni et al., 2025"
        rid = humanized_to_rid.get(inner)
        if rid:
            url = METADATA_URLS.get(rid) or answer_urls.get(rid)
            if url:
                return f"[{inner}]({url})"
        return full

    text = re.sub(
        r"\(([A-Z][a-z]+ et al\., \d{4}[a-z]?)\)",
        _replace_paren,
        text,
    )

    return text


def _display_retrieved_images(image_nodes, image_store=None):
    """Show retrieved PDF figures as a compact thumbnail grid.

    Each thumbnail shows:
    - The figure image
    - Caption text underneath (paper caption + VLM description)
    - A clickable link to the source paper (if available)

    Works both for live results (StoredNode objects) and history replay
    (serialized dicts with storage_key/caption/page/doc_id).
    """
    if not image_nodes:
        return
    with st.expander(f"Retrieved figures ({len(image_nodes)})", expanded=False):
        # Collect image data first
        images = []
        for node in image_nodes:
            if hasattr(node, "metadata"):
                meta = node.metadata
                storage_key = meta.get("image_storage_key")
                caption = node.text or ""
                page = meta.get("page", "?")
                doc_id = meta.get("document_id", "unknown")
                caption_text = meta.get("caption_text", "")
                vlm_description = meta.get("vlm_description", "")
                figure_type = meta.get("figure_type", "")
                source_url = meta.get("source_url", "")
                source_title = meta.get("source_title", "")
            else:
                storage_key = node.get("storage_key")
                caption = node.get("caption", "")
                page = node.get("page", "?")
                doc_id = node.get("doc_id", "unknown")
                caption_text = node.get("caption_text", "")
                vlm_description = node.get("vlm_description", "")
                figure_type = node.get("figure_type", "")
                source_url = node.get("source_url", "")
                source_title = node.get("source_title", "")

            # Build display caption: prefer structured fields, fall back to raw text
            display_caption = ""
            if caption_text:
                display_caption = caption_text
            elif caption:
                # Strip embedding-format prefixes like "[Figure 3]" or "[figure:doc p5]"
                display_caption = re.sub(
                    r"^\[(?:Fig(?:ure|\.)\s*\d+|figure:\S+\s+p\d+)\]\s*",
                    "", caption,
                ).strip()

            short_label = f"{doc_id} p.{page}"
            if figure_type:
                short_label += f" ({figure_type})"

            img_bytes = None
            if storage_key and image_store:
                img_bytes = image_store._sync_get(storage_key)

            images.append({
                "bytes": img_bytes,
                "short_label": short_label,
                "display_caption": display_caption,
                "vlm_description": vlm_description,
                "source_url": source_url,
                "source_title": source_title,
                "caption": caption,
                "doc_id": doc_id,
                "page": page,
            })

        # Render as a thumbnail grid (3 columns)
        has_images = [img for img in images if img["bytes"]]
        text_only = [img for img in images if not img["bytes"] and img["caption"]]

        if has_images:
            n_cols = min(3, len(has_images))
            cols = st.columns(n_cols)
            for idx, img in enumerate(has_images):
                col = cols[idx % n_cols]
                with col:
                    st.image(img["bytes"], caption=img["short_label"], width=200)

                    # Show caption underneath the figure
                    if img["display_caption"]:
                        st.caption(img["display_caption"][:200])
                    if img["vlm_description"]:
                        st.caption(f"*{img['vlm_description'][:200]}*")

                    # Source link
                    if img["source_url"]:
                        link_text = img["source_title"] or img["doc_id"]
                        st.markdown(
                            f"[{link_text}]({img['source_url']})",
                            help="View source paper",
                        )

                    with st.popover(f"Expand: {img['short_label']}"):
                        st.image(img["bytes"], caption=img["display_caption"] or img["short_label"])
                        if img["vlm_description"]:
                            st.write(img["vlm_description"])
                        if img["source_url"]:
                            link_text = img["source_title"] or img["doc_id"]
                            st.markdown(f"Source: [{link_text}]({img['source_url']})")

        # Show text-only fallbacks for images not found in store
        if text_only:
            for img in text_only:
                # Clean up raw metadata from caption (e.g. "[Image page=3 idx=1...]")
                clean_caption = re.sub(
                    r"\[Image page=\d+ idx=\d+ name=\S+\]\s*Size:\s*\d+x\d+,?\s*Data:\s*\d+ bytes\s*",
                    "", img["caption"],
                ).strip()
                if clean_caption:
                    st.caption(f"{img['short_label']}: {clean_caption[:200]}")
                else:
                    st.caption(f"{img['short_label']}: (image not available)")


def _build_chat_settings_dict(local_vars: dict) -> dict:
    """Extract chat settings from the caller's local variables for display."""
    keys = [
        "top_k", "effective_top_k", "best_guess", "research_mode",
        "max_tokens_override", "max_retries", "enable_semantic_scholar",
        "s2_top_k", "enable_query_planner", "planner_max_queries",
        "enable_cross_encoder", "with_images", "send_images_to_llm",
        "mode", "ensemble_strategy", "vllm_model", "precision",
    ]
    return {k: local_vars[k] for k in keys if k in local_vars}


def _render_chat_settings(settings: dict):
    """Render chat settings as a small caption below the answer."""
    if not settings:
        return
    parts = []
    if settings.get("research_mode"):
        parts.append("research")
    elif settings.get("best_guess"):
        parts.append("best-guess")
    else:
        parts.append("strict")
    top_k = settings.get("effective_top_k") or settings.get("top_k")
    if top_k:
        parts.append(f"top_k={top_k}")
    if settings.get("max_tokens_override"):
        parts.append(f"max_tokens={settings['max_tokens_override']}")
    if settings.get("enable_semantic_scholar"):
        parts.append(f"S2(k={settings.get('s2_top_k', '?')})")
    if settings.get("enable_query_planner"):
        parts.append(f"query_exp(n={settings.get('planner_max_queries', '?')})")
    if settings.get("enable_cross_encoder"):
        parts.append("reranker")
    model = settings.get("vllm_model") or settings.get("mode")
    if model:
        parts.append(f"model={model}")
    st.caption(f"Settings: {' · '.join(parts)}")


def _display_single_result(
    result, elapsed: float, *, pipeline: RAGPipeline | None = None,
    energy_wh: float = 0.0, energy_method: str = "",
    chat_settings: dict | None = None,
):
    """Display a single-model answer."""
    answer = result.answer
    timing = result.timing
    confidence = _extract_confidence(result.raw_response)

    # Build URL map from retrieved snippets (especially S2 papers which have URLs)
    _snippet_urls: dict[str, str] = {}
    for s in result.retrieval.snippets:
        meta = s.metadata or {}
        doc_id = meta.get("document_id", "")
        url = meta.get("url", "")
        if doc_id and url and doc_id not in _snippet_urls:
            _snippet_urls[doc_id] = url

    # Linkify inline [ref_id] citations so they match the Sources section
    linked_explanation = _linkify_citations(
        answer.explanation, ref_ids=answer.ref_id,
        snippet_urls=_snippet_urls,
    )

    if linked_explanation and linked_explanation != "is_blank":
        # For long multi-paragraph answers (research mode), use regular markdown
        # instead of wrapping everything in bold
        if len(linked_explanation) > 500 or "\n\n" in linked_explanation:
            st.markdown(linked_explanation)
        else:
            st.markdown(f"**{linked_explanation}**")
        if confidence == "low":
            st.warning("Best guess — the retrieved context only partially supports this answer.")
    elif answer.answer and answer.answer != "is_blank":
        st.markdown(f"**{answer.answer}**")
    else:
        st.markdown("**Out-of-scope** — the provided documents do not contain enough information to answer this question.")
    if answer.answer_value and answer.answer_value != "is_blank":
        st.markdown(f"Value: `{answer.answer_value}`")

    # Sources are rendered by _render_details() below (avoids duplication)

    # Serialize image nodes for history replay (displayed via _render_details below)
    image_details = []
    if result.retrieval.image_nodes:
        for node in result.retrieval.image_nodes:
            image_details.append({
                "storage_key": node.metadata.get("image_storage_key"),
                "caption": node.text or "",
                "page": node.metadata.get("page", "?"),
                "doc_id": node.metadata.get("document_id", "unknown"),
            })

    # Count S2 snippets for debug visibility
    s2_snippet_count = sum(
        1 for s in result.retrieval.snippets
        if s.node_id.startswith("s2:")
    )
    total_snippet_count = len(result.retrieval.snippets)
    if s2_snippet_count:
        _debug(f"Retrieval: {total_snippet_count} snippets ({s2_snippet_count} from Semantic Scholar)")
    else:
        _debug(f"Retrieval: {total_snippet_count} snippets (no Semantic Scholar results)")

    # Fallback: if the LLM answered but didn't cite sources, infer from top snippets
    effective_ref_ids = _clean_ref_ids(answer.ref_id)
    if not effective_ref_ids and answer.explanation and answer.explanation != "is_blank":
        # Extract unique doc_ids from the top retrieved snippets
        seen = set()
        for s in result.retrieval.snippets[:5]:
            doc_id = (s.metadata or {}).get("document_id", "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                effective_ref_ids.append(doc_id)
        if effective_ref_ids:
            _debug(f"No ref_ids from LLM; inferred {len(effective_ref_ids)} from top snippets: {effective_ref_ids}")

    details = {
        "timing": timing,
        "elapsed": elapsed,
        "energy_wh": energy_wh,
        "energy_method": energy_method,
        "ref_id": effective_ref_ids,
        "snippet_urls": _snippet_urls,
        "supporting_materials": answer.supporting_materials,
        "snippets": [
            {"rank": s.rank, "score": s.score, "title": s.document_title, "text": s.text, "node_id": s.node_id}
            for s in result.retrieval.snippets
        ],
        "raw_response": result.raw_response,
        "image_nodes": image_details,
        "s2_snippet_count": s2_snippet_count,
        "chat_settings": chat_settings,
    }
    image_store = getattr(pipeline, "_image_store", None) if pipeline else None
    _render_details(details, image_store=image_store)

    if answer.explanation and answer.explanation != "is_blank":
        display_answer = answer.explanation
    elif answer.answer and answer.answer != "is_blank":
        display_answer = answer.answer
    else:
        display_answer = "Out-of-scope"
    st.session_state.messages.append({
        "role": "assistant", "content": display_answer, "details": details,
    })


def _display_ensemble_result(
    agg: dict, model_results: dict, elapsed: float, strategy: str,
    energy_wh: float = 0.0, energy_method: str = "",
    chat_settings: dict | None = None,
):
    """Display aggregated ensemble answer + per-model breakdown."""
    # Build snippet URL map early so _linkify_citations can use it
    _first = next(iter(model_results.values()))["result"]
    _ensemble_snippet_urls: dict[str, str] = {}
    for s in _first.retrieval.snippets:
        meta = s.metadata or {}
        doc_id = meta.get("document_id", "")
        url = meta.get("url", "")
        if doc_id and url and doc_id not in _ensemble_snippet_urls:
            _ensemble_snippet_urls[doc_id] = url

    linked_explanation = _linkify_citations(
        agg["explanation"], ref_ids=agg.get("ref_id"),
        snippet_urls=_ensemble_snippet_urls,
    )

    if linked_explanation and linked_explanation != "is_blank":
        st.markdown(f"**{linked_explanation}**")
    elif agg["answer"] and agg["answer"] != "is_blank":
        st.markdown(f"**{agg['answer']}**")
    else:
        st.markdown("**Out-of-scope** — the provided documents do not contain enough information to answer this question.")
    if agg["answer_value"] and agg["answer_value"] != "is_blank":
        st.markdown(f"Value: `{agg['answer_value']}`")

    n_models = len(model_results)
    model_times = [e["time"] for e in model_results.values()]
    total_gen = sum(model_times)

    _e_label = "Energy" if energy_method in ("nvml", "power_sampling", "dcgm", "server_reported", "token_based") else "Est. energy"
    _e_str = _format_energy(energy_wh) if energy_wh > 0 else "—"
    cols = st.columns(4)
    cols[0].metric("Models", n_models)
    cols[1].metric("Aggregation", strategy)
    cols[2].metric("Total", f"{elapsed:.1f}s")
    cols[3].metric(_e_label, _e_str)

    # Per-model answers
    with st.expander(f"Individual model answers ({n_models} models)"):
        for name, info in agg["individual"].items():
            agreed = info["answer_value"] == agg["answer_value"]
            marker = "+" if agreed else "-"
            val = info["answer_value"] if info["answer_value"] and info["answer_value"] != "is_blank" else "Out-of-scope"
            ans = info["answer"] if info["answer"] and info["answer"] != "is_blank" else "Out-of-scope"
            st.markdown(
                f"**{name}** ({info['time']:.1f}s) [{marker}]  \n"
                f"Answer: `{val}` — {ans}"
            )
            if info["explanation"] and info["explanation"] != "is_blank":
                st.caption(_linkify_citations(
                    info["explanation"], ref_ids=info.get("ref_id"),
                ))
            st.divider()

    # Clickable reference links (verified URLs only)
    if agg["ref_id"]:
        links = []
        for rid in agg["ref_id"]:
            url = METADATA_URLS.get(rid)
            if not url:
                url = _ensemble_snippet_urls.get(rid)
            label = _humanize_ref_id(rid)
            if url:
                links.append(f"[{label}]({url})")
            else:
                links.append(label)
        st.markdown("Sources: " + " · ".join(links))
    snippets = _first.retrieval.snippets
    if snippets:
        display_snippets = snippets[:5]
        label = f"Retrieved context ({len(display_snippets)} of {len(snippets)} chunks)"
        with st.expander(label):
            for s in display_snippets:
                tag = " **[S2]**" if s.node_id.startswith("s2:") else ""
                st.markdown(f"**#{s.rank}** _{s.document_title}_ (score: {s.score:.3f}){tag}")
                st.text(s.text[:500] + ("..." if len(s.text) > 500 else ""))
                st.divider()

    # Show retrieved figures from first model's retrieval
    image_nodes = _first.retrieval.image_nodes
    _display_retrieved_images(image_nodes[:5] if image_nodes else None)

    # Raw responses available via debug logging (removed from UI for cleanliness)

    image_details = []

    details = {
        "elapsed": elapsed,
        "energy_wh": energy_wh,
        "energy_method": energy_method,
        "ensemble": True,
        "strategy": strategy,
        "models": list(model_results.keys()),
        "answer": agg["answer"],
        "answer_value": agg["answer_value"],
        "image_nodes": image_details,
        "chat_settings": chat_settings,
    }
    if total_cost is not None:
        details["total_cost"] = total_cost
    if agg["explanation"] and agg["explanation"] != "is_blank":
        display_answer = agg["explanation"]
    elif agg["answer"] and agg["answer"] != "is_blank":
        display_answer = agg["answer"]
    else:
        display_answer = "Out-of-scope"
    st.session_state.messages.append({
        "role": "assistant", "content": display_answer, "details": details,
    })


def _render_details(details: dict, *, image_store=None):
    """Render expandable sections for a stored message (history replay)."""
    energy_wh = details.get("energy_wh", 0.0)
    energy_method = details.get("energy_method", "")
    _measured = energy_method in ("nvml", "power_sampling", "dcgm", "server_reported", "token_based")
    energy_label = "Energy" if _measured else "Est. energy"
    energy_str = _format_energy(energy_wh) if energy_wh > 0 else "—"

    if details.get("ensemble"):
        # Minimal replay for ensemble messages
        total_cost = details.get("total_cost")
        if total_cost is not None:
            cols = st.columns(5)
            cols[0].metric("Models", len(details.get("models", [])))
            cols[1].metric("Aggregation", details.get("strategy", ""))
            cols[2].metric("Total", f"{details.get('elapsed', 0):.1f}s")
            cols[3].metric("Est. cost", f"${total_cost:.4f}")
            cols[4].metric(energy_label, energy_str)
        else:
            cols = st.columns(4)
            cols[0].metric("Models", len(details.get("models", [])))
            cols[1].metric("Aggregation", details.get("strategy", ""))
            cols[2].metric("Total", f"{details.get('elapsed', 0):.1f}s")
            cols[3].metric(energy_label, energy_str)
        image_details = details.get("image_nodes", [])
        if image_details:
            _display_retrieved_images(image_details[:5], image_store)
        _render_chat_settings(details.get("chat_settings", {}))
        return

    timing = details.get("timing", {})
    elapsed = details.get("elapsed", 0)
    feedback_rounds = timing.get("feedback_rounds", 0)

    if feedback_rounds:
        cols = st.columns(5)
        cols[0].metric("Retrieval", f"{timing.get('retrieval_s', 0):.1f}s")
        cols[1].metric("Generation", f"{timing.get('generation_s', 0):.1f}s")
        cols[2].metric("Feedback rounds", feedback_rounds)
        cols[3].metric("Total", f"{elapsed:.1f}s")
        cols[4].metric(energy_label, energy_str)
    else:
        cols = st.columns(4)
        cols[0].metric("Retrieval", f"{timing.get('retrieval_s', 0):.1f}s")
        cols[1].metric("Generation", f"{timing.get('generation_s', 0):.1f}s")
        cols[2].metric("Total", f"{elapsed:.1f}s")
        cols[3].metric(energy_label, energy_str)

    ref_ids = _clean_ref_ids(details.get("ref_id", []))
    snippet_urls = details.get("snippet_urls", {})  # URLs from S2 and other retrieved snippets
    if ref_ids:
        links = []
        for i, rid in enumerate(ref_ids if isinstance(ref_ids, list) else [ref_ids]):
            # Only use verified URLs from metadata.csv or retrieved snippet metadata.
            # Never use LLM-provided ref_url — it is prone to hallucination.
            url = METADATA_URLS.get(rid)
            if not url:
                url = snippet_urls.get(rid)  # Try S2/snippet URLs
            label = _humanize_ref_id(rid)
            if url and url != "is_blank":
                links.append(f"[{label}]({url})")
            else:
                links.append(label)
        st.markdown("Sources: " + " · ".join(links))
        sm = details.get("supporting_materials", "")
        if sm and sm != "is_blank":
            st.caption(f"Supporting: {sm}")

    snippets = details.get("snippets", [])
    if snippets:
        # Diversify: pick best chunk per unique source first, then fill remaining
        # slots with next-best chunks (regardless of source).
        max_display = 5
        seen_sources: set[str] = set()
        diverse_snippets: list[dict] = []
        remaining: list[dict] = []
        for s in snippets:
            source = s.get("title", "")
            if source not in seen_sources:
                seen_sources.add(source)
                diverse_snippets.append(s)
            else:
                remaining.append(s)
            if len(diverse_snippets) >= max_display:
                break
        # If we have fewer than max_display unique sources, fill with top remaining
        if len(diverse_snippets) < max_display:
            for s in remaining:
                diverse_snippets.append(s)
                if len(diverse_snippets) >= max_display:
                    break

        n_sources = len({s.get("title", "") for s in diverse_snippets})
        label = f"Retrieved context ({len(diverse_snippets)} chunks from {n_sources} sources, {len(snippets)} total)"
        with st.expander(label):
            for s in diverse_snippets:
                tag = " **[S2]**" if s.get("node_id", "").startswith("s2:") else ""
                st.markdown(f"**#{s['rank']}** _{s['title']}_ (score: {s['score']:.3f}){tag}")
                st.text(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))
                st.divider()

    image_details = details.get("image_nodes", [])
    if image_details:
        _display_retrieved_images(image_details[:5], image_store)

    # Chat settings footer for easy log review
    _render_chat_settings(details.get("chat_settings", {}))

    # Raw LLM response available in debug logs (removed from UI for cleanliness)


if __name__ == "__main__":
    main()
