"""Embedding utilities used across KohakuRAG."""

import asyncio
import contextlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol, Sequence

import numpy as np


@contextlib.contextmanager
def _suppress_tqdm():
    """Temporarily disable tqdm progress bars via environment variable."""
    old = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = old

try:
    import torch
    from transformers import AutoModel

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingModel(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimension(self) -> int:  # pragma: no cover - interface only
        ...

    async def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        """Return a 2D numpy array of shape (len(texts), dimension)."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length, handling zero vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    return vectors / norms


def average_embeddings(
    child_vectors: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Compute the normalized weighted-mean vector for parent nodes.

    When *weights* is provided (typically text lengths), longer child segments
    contribute more to the parent's semantic representation.  When *weights* is
    ``None`` the behaviour falls back to a simple unweighted mean.
    """
    if not child_vectors:
        raise ValueError("average_embeddings requires at least one child vector.")

    stacked = np.vstack(child_vectors)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            w = np.ones(len(child_vectors), dtype=np.float64) / len(child_vectors)
        mean = np.average(stacked, axis=0, weights=w).reshape(1, -1)
    else:
        mean = np.mean(stacked, axis=0, keepdims=True)

    return _normalize(mean.astype(np.float32))[0]


def _detect_device() -> Any:
    """Auto-detect best available device (CUDA > MPS > CPU).

    Requires torch; raises ImportError if torch is not installed.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for local embedding models")

    if torch.cuda.is_available():
        return torch.device("cuda")

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")

    return torch.device("cpu")


class JinaEmbeddingModel:
    """Wrapper around jinaai/jina-embeddings-v3 using HuggingFace AutoModel."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        *,
        pooling: str = "cls",
        normalize: bool = True,
        batch_size: int = 8,
        device: Any | None = None,
    ) -> None:
        """Initialize Jina embedding model with lazy loading.

        Args:
            model_name: HuggingFace model identifier
            pooling: Pooling strategy (unused, kept for compatibility)
            normalize: Whether to normalize embeddings (unused, kept for compatibility)
            batch_size: Batch size for encoding (unused, kept for compatibility)
            device: Target device (auto-detected if None)
        """
        resolved_device = _detect_device() if device is None else torch.device(device)

        self._model_name = model_name
        self._pooling = pooling
        self._normalize = normalize
        self._batch_size = batch_size
        self._device = resolved_device

        # Use FP16 on GPU for 2x speedup
        self._dtype = (
            torch.float16 if resolved_device.type in {"cuda", "mps"} else torch.float32
        )

        # Lazy initialization - model loaded on first use
        self._model: Any | None = None
        self._dimension: int | None = None

        # Single-worker executor for thread-safe async embedding
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    def _ensure_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        # Load model from HuggingFace
        model = AutoModel.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )
        model = model.to(self._device, dtype=self._dtype)
        model.eval().requires_grad_(False)
        self._model = model

        # Infer embedding dimension from model attributes
        dim = getattr(model, "embedding_size", None)
        if dim is None:
            dim = getattr(getattr(model, "config", None), "hidden_size", None)

        # Fall back to probing with a test input
        if dim is None:
            with torch.no_grad():
                probe = model.encode(["dimension probe"])

            if isinstance(probe, torch.Tensor):
                dim = probe.shape[-1]
            else:
                probe_arr = np.asarray(probe)
                dim = probe_arr.shape[-1]

        if dim is None:
            raise RuntimeError("Unable to infer embedding dimension for the model.")

        self._dimension = int(dim)

    @property
    def dimension(self) -> int:
        self._ensure_model()
        assert self._dimension is not None
        return self._dimension

    def _sync_encode(self, texts: Sequence[str]) -> np.ndarray:
        """Synchronous encoding logic (called via executor).

        Returns:
            Array of shape (len(texts), dimension) with float32 dtype
        """
        self._ensure_model()
        assert self._model is not None

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        # Ensure all inputs are strings
        str_texts = [str(t) for t in texts]

        # Run inference without gradients
        with torch.no_grad():
            embeddings = self._model.encode(str_texts)

        # Convert to numpy, handling both Tensor and array outputs
        if isinstance(embeddings, torch.Tensor):
            arr = embeddings.detach().float().cpu().numpy()
        else:
            arr = np.asarray(embeddings)

        return arr.astype(np.float32, copy=False)

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into embedding vectors (async).

        Uses single-worker executor to ensure thread safety for GPU operations.

        Returns:
            Array of shape (len(texts), dimension) with float32 dtype
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_encode, texts)


class JinaV4EmbeddingModel:
    """Wrapper around jinaai/jina-embeddings-v4 with multimodal support.

    Supports:
    - Text embedding with configurable Matryoshka dimensions
    - Image embedding (unified embedding space with text)
    - Task-specific adapters (retrieval, text-matching, code)
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        *,
        task: str = "retrieval",
        truncate_dim: int = 1024,
        batch_size: int = 64,
        device: Any | None = None,
    ) -> None:
        """Initialize Jina V4 embedding model with lazy loading.

        Args:
            model_name: HuggingFace model identifier
            task: Task mode - "retrieval", "text-matching", or "code"
            truncate_dim: Matryoshka dimension (128, 256, 512, 1024, 2048)
            batch_size: Max texts per encode_text call (avoids GPU OOM on large batches)
            device: Target device (auto-detected if None)
        """
        # Validate truncate_dim
        valid_dims = [128, 256, 512, 1024, 2048]
        if truncate_dim not in valid_dims:
            raise ValueError(
                f"truncate_dim must be one of {valid_dims}, got {truncate_dim}"
            )

        resolved_device = _detect_device() if device is None else torch.device(device)

        self._model_name = model_name
        self._task = task
        self._truncate_dim = truncate_dim
        self._batch_size = batch_size
        self._device = resolved_device

        # Use FP16 on GPU for faster inference
        self._dtype = (
            torch.float16 if resolved_device.type in {"cuda", "mps"} else torch.float32
        )

        # Lazy initialization
        self._model: Any | None = None

        # Single-worker executor for thread-safe async operations
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    @staticmethod
    def _find_snapshot(model_name: str) -> str | None:
        """Locate the snapshot directory for *model_name* in the HF cache."""
        hf_home = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
        hub_cache = os.environ.get(
            "HF_HUB_CACHE", os.path.join(hf_home, "hub")
        )
        model_dir = f"models--{model_name.replace('/', '--')}"
        snapshots_dir = os.path.join(hub_cache, model_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return None
        snapshots = sorted(os.listdir(snapshots_dir))
        if not snapshots:
            return None
        return os.path.join(snapshots_dir, snapshots[-1])

    @staticmethod
    def _prepare_snapshot(snapshot_path: str) -> str:
        """If adapters/ is missing from a read-only snapshot, build a merged
        directory in /tmp that symlinks all original files plus any adapters
        found at a fallback location (JINA_ADAPTERS_DIR env var or
        /tmp/jina-embeddings-v4/adapters).

        Returns the path to use for model loading (original or merged).
        """
        adapters_in_snap = os.path.join(snapshot_path, "adapters")
        if os.path.isdir(adapters_in_snap):
            return snapshot_path  # already present

        # Check fallback locations for adapters
        fallback = os.environ.get(
            "JINA_ADAPTERS_DIR",
            "/tmp/jina-embeddings-v4/adapters",
        )
        if not os.path.isdir(fallback):
            print(
                f"[embeddings] WARNING: adapters/ missing from snapshot and "
                f"fallback {fallback} not found",
                flush=True,
            )
            return snapshot_path  # let downstream error naturally

        print(
            f"[embeddings] Merging read-only snapshot with adapters from {fallback}",
            flush=True,
        )
        merged = "/tmp/jina-merged-snapshot"
        os.makedirs(merged, exist_ok=True)

        # Symlink every file/dir from the original snapshot
        for entry in os.listdir(snapshot_path):
            src = os.path.join(snapshot_path, entry)
            dst = os.path.join(merged, entry)
            if not os.path.exists(dst):
                os.symlink(src, dst)

        # Symlink the adapters directory
        dst_adapters = os.path.join(merged, "adapters")
        if not os.path.exists(dst_adapters):
            os.symlink(fallback, dst_adapters)

        print(f"[embeddings] Merged snapshot at {merged}", flush=True)
        return merged

    def _ensure_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        hf_home = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
        hub_cache = os.environ.get(
            "HF_HUB_CACHE", os.path.join(hf_home, "hub")
        )
        print(
            f"[embeddings] Loading {self._model_name} "
            f"(HF_HOME={hf_home}, hub_cache={hub_cache})",
            flush=True,
        )

        # Strategy 1: normal from_pretrained with explicit cache_dir
        try:
            model = AutoModel.from_pretrained(
                self._model_name,
                trust_remote_code=True,
                local_files_only=True,
                cache_dir=hub_cache,
            )
        except Exception as e1:
            print(f"[embeddings] Strategy 1 failed: {e1}", flush=True)

            # Strategy 2: find snapshot, dump diagnostics, and load the
            # custom model class manually.
            snapshot_path = self._find_snapshot(self._model_name)
            if snapshot_path is None:
                if os.path.isdir(hub_cache):
                    print(f"[embeddings] hub_cache contents: {os.listdir(hub_cache)}", flush=True)
                else:
                    print("[embeddings] hub_cache dir does not exist!", flush=True)
                raise RuntimeError(
                    f"Cannot find model {self._model_name} in {hub_cache}"
                ) from e1

            # --- diagnostics ---
            snap_files = os.listdir(snapshot_path)
            print(f"[embeddings] Snapshot dir: {snapshot_path}", flush=True)
            print(f"[embeddings] Snapshot contents: {snap_files}", flush=True)

            import json
            config_path = os.path.join(snapshot_path, "config.json")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                print(f"[embeddings] config.json model_type={cfg.get('model_type')}", flush=True)
                print(f"[embeddings] config.json auto_map={cfg.get('auto_map')}", flush=True)
            else:
                print("[embeddings] WARNING: no config.json in snapshot!", flush=True)
                raise RuntimeError(f"No config.json in {snapshot_path}") from e1

            # Merge adapters from fallback location if needed
            snapshot_path = self._prepare_snapshot(snapshot_path)

            # Strategy 2a: add snapshot to sys.path so trust_remote_code
            # can resolve the custom .py modules, then retry.
            import sys
            if snapshot_path not in sys.path:
                sys.path.insert(0, snapshot_path)
                print(f"[embeddings] Added {snapshot_path} to sys.path", flush=True)

            try:
                model = AutoModel.from_pretrained(
                    snapshot_path,
                    trust_remote_code=True,
                )
            except Exception as e2:
                print(f"[embeddings] Strategy 2a failed: {e2}", flush=True)

                # Strategy 2b: manually import the custom class from auto_map
                auto_map = cfg.get("auto_map", {})
                auto_model_entry = auto_map.get("AutoModel", "")
                if "--" in auto_model_entry:
                    # Format: "jinaai/jina-embeddings-v4--module.ClassName"
                    auto_model_entry = auto_model_entry.split("--", 1)[1]
                if "." in auto_model_entry:
                    mod_name, cls_name = auto_model_entry.rsplit(".", 1)
                    print(f"[embeddings] Strategy 2b: importing {mod_name}.{cls_name}", flush=True)
                    import importlib
                    custom_mod = importlib.import_module(mod_name)
                    custom_cls = getattr(custom_mod, cls_name)
                    model = custom_cls.from_pretrained(
                        snapshot_path,
                        trust_remote_code=True,
                    )
                else:
                    raise RuntimeError(
                        f"Cannot resolve auto_map entry: {auto_model_entry}"
                    ) from e2

        model = model.to(self._device, dtype=self._dtype)
        model.eval().requires_grad_(False)
        self._model = model

    @property
    def dimension(self) -> int:
        """Return the configured Matryoshka dimension."""
        return self._truncate_dim

    def _sync_encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Synchronous text encoding (called via executor).

        Chunks inputs into ``batch_size`` pieces to keep GPU memory bounded
        while still saturating the device.

        Returns:
            Array of shape (len(texts), truncate_dim) with float32 dtype
        """
        self._ensure_model()
        assert self._model is not None

        if not texts:
            return np.zeros((0, self._truncate_dim), dtype=np.float32)

        str_texts = [str(t) for t in texts]

        chunks: list[np.ndarray] = []
        for start in range(0, len(str_texts), self._batch_size):
            batch = str_texts[start : start + self._batch_size]
            with torch.no_grad(), _suppress_tqdm():
                embeddings = self._model.encode_text(
                    texts=batch,
                    task=self._task,
                    prompt_name="query",
                    truncate_dim=self._truncate_dim,
                    max_length=8192,
                )
            if isinstance(embeddings, torch.Tensor):
                arr = embeddings.detach().float().cpu().numpy()
            elif isinstance(embeddings, list):
                arr = torch.stack(embeddings).detach().float().cpu().numpy()
            else:
                arr = np.asarray(embeddings)
            chunks.append(arr.astype(np.float32, copy=False))

        return np.vstack(chunks) if len(chunks) > 1 else chunks[0]

    def _sync_encode_images(self, image_bytes_list: Sequence[bytes]) -> np.ndarray:
        """Synchronous image encoding (called via executor).

        Args:
            image_bytes_list: List of image data as bytes

        Returns:
            Array of shape (len(images), truncate_dim) with float32 dtype
        """
        self._ensure_model()
        assert self._model is not None

        if not image_bytes_list:
            return np.zeros((0, self._truncate_dim), dtype=np.float32)

        # Convert bytes to PIL Images
        pil_images = []
        for img_bytes in image_bytes_list:
            try:
                img = Image.open(io.BytesIO(img_bytes))
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pil_images.append(img)
            except Exception as e:
                raise ValueError(f"Failed to load image: {e}")

        # Run inference with JinaV4's encode_image method
        with torch.no_grad():
            with _suppress_tqdm():
                embeddings = self._model.encode_image(
                    images=pil_images,
                    task=self._task,
                    truncate_dim=self._truncate_dim,
                )

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            arr = embeddings.detach().float().cpu().numpy()
        elif isinstance(embeddings, list):
            arr = torch.stack(embeddings).detach().float().cpu().numpy()
        else:
            arr = np.asarray(embeddings)

        return arr.astype(np.float32, copy=False)

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into embedding vectors (async).

        Args:
            texts: List of text strings to embed

        Returns:
            Array of shape (len(texts), truncate_dim) with float32 dtype
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_encode_text, texts)

    async def embed_images(self, images: Sequence[bytes]) -> np.ndarray:
        """Encode images into embedding vectors (async).

        Args:
            images: List of image data as bytes

        Returns:
            Array of shape (len(images), truncate_dim) with float32 dtype
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._sync_encode_images, images
        )


class LocalHFEmbeddingModel:
    """Local embedding model using sentence-transformers.

    Runs entirely on-device with no network calls. Suitable for
    fully local RAG pipelines.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        *,
        device: Any | None = None,
    ) -> None:
        """Initialize local embedding model.

        Args:
            model_name: HuggingFace model identifier for sentence-transformers
                       Recommended: "BAAI/bge-base-en-v1.5" (768-dim)
                                    "intfloat/e5-base-v2" (768-dim)
            device: Target device (auto-detected if None)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for LocalHFEmbeddingModel. "
                "Install with: pip install sentence-transformers"
            )

        resolved_device = _detect_device() if device is None else torch.device(device)
        device_str = str(resolved_device)

        self._model_name = model_name
        self._st_model = SentenceTransformer(model_name, device=device_str)
        self._dimension = self._st_model.get_sentence_embedding_dimension()

        # Single-worker executor for thread-safe async embedding
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _sync_encode(self, texts: Sequence[str]) -> np.ndarray:
        """Synchronous encoding (called via executor).

        Returns:
            Array of shape (len(texts), dimension) with float32 dtype
        """
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        str_texts = [str(t) for t in texts]
        vecs = self._st_model.encode(
            str_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into embedding vectors (async).

        Returns:
            Array of shape (len(texts), dimension) with float32 dtype
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_encode, texts)
