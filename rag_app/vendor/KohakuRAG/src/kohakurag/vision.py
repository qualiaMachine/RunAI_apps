"""Vision model integrations for image captioning."""

import asyncio
import base64
import io
import os
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError

try:
    from openrouter import OpenRouter

    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    # AutoModelForImageTextToText is the recommended class in transformers v5+.
    # Fall back to AutoModelForVision2Seq for older versions.
    try:
        from transformers import AutoModelForImageTextToText as _VisionAutoModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as _VisionAutoModel

    HF_VISION_AVAILABLE = True
except ImportError:
    HF_VISION_AVAILABLE = False


def _load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Load environment variables from a .env file."""
    env_path = Path(path)
    if not env_path.exists():
        return {}

    env_vars: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_vars[key.strip()] = value.strip().strip('"').strip("'")

    return env_vars


class VisionModel(ABC):
    """Abstract base class for vision models that can caption images."""

    @abstractmethod
    async def caption(
        self,
        image_data: bytes,
        *,
        prompt: str | None = None,
        max_tokens: int = 300,
    ) -> str:
        """Generate a caption for an image.

        Args:
            image_data: Raw image bytes (any format PIL supports)
            prompt: Custom prompt for captioning (uses default if None)
            max_tokens: Maximum tokens in response

        Returns:
            Generated caption text
        """
        pass


class OpenAIVisionModel(VisionModel):
    """Vision model using OpenAI's Chat Completions API with vision capabilities.

    Supports both OpenAI models (gpt-4o, gpt-4o-mini) and OpenRouter models
    (qwen/qwen3-vl-235b-a22b-instruct) via base_url parameter.
    """

    DEFAULT_CAPTION_PROMPT = (
        "This is a figure/chart/diagram from a scientific research paper. "
        "Describe what this image represents in detail, focusing on the actual content and data. "
        "If it's a plot/chart: provide specific metric values, axis labels, trends, and comparisons shown. "
        "If it's a diagram/architecture: describe the components, connections, and system structure. "
        "If it's a table: extract key data points and comparisons. "
        "Be specific and detailed (3-5 sentences), as this will be used for information retrieval."
    )

    def __init__(
        self,
        *,
        model: str = "qwen/qwen3-vl-235b-a22b-instruct",
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
        base_retry_delay: float = 3.0,
        max_concurrent: int = 5,
    ) -> None:
        """Initialize OpenAI vision model with automatic rate limit retry.

        Args:
            model: Model identifier (default: qwen/qwen3-vl-235b-a22b-instruct for OpenRouter)
                   Other options: gpt-4o, gpt-4o-mini, gpt-4-turbo
            api_key: API key (reads from OPENAI_API_KEY env if not provided)
            organization: OpenAI organization ID (optional)
            base_url: API base URL (default: https://openrouter.ai/api/v1 for OpenRouter models)
                     Set OPENAI_BASE_URL env variable or pass explicitly
            max_retries: Maximum retry attempts on rate limit errors
            base_retry_delay: Base delay for exponential backoff (seconds)
            max_concurrent: Maximum concurrent API requests (default: 5 for vision)
                          Set to 0 or negative to disable rate limiting
        """
        dotenv_vars: dict[str, str] | None = None

        # Try multiple sources for API key
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            dotenv_vars = _load_dotenv()
            key = dotenv_vars.get("OPENAI_API_KEY")

        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIVisionModel.")

        # Resolve base URL - default to OpenRouter for qwen models
        resolved_base_url = base_url
        if resolved_base_url is None:
            env_base_url = os.environ.get("OPENAI_BASE_URL")
            if env_base_url is None:
                if dotenv_vars is None:
                    dotenv_vars = _load_dotenv()
                env_base_url = dotenv_vars.get("OPENAI_BASE_URL")

            # Default to OpenRouter if using qwen model
            if env_base_url is None and "qwen" in model.lower():
                resolved_base_url = "https://openrouter.ai/api/v1"
            else:
                resolved_base_url = env_base_url

        self._client = AsyncOpenAI(
            api_key=key,
            organization=organization,
            base_url=resolved_base_url,
        )
        self._model = model
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay
        self._semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

    def _parse_retry_after(self, error_message: str) -> float | None:
        """Extract wait time from rate limit error message.

        Handles formats like:
        - "Please try again in 23ms"
        - "Please try again in 1.5s"
        - "Please try again in 2m"
        """
        patterns = [
            (r"try again in (\d+(?:\.\d+)?)ms", 0.001),  # milliseconds
            (r"try again in (\d+(?:\.\d+)?)s", 1.0),  # seconds
            (r"try again in (\d+(?:\.\d+)?)m", 60.0),  # minutes
        ]
        for pattern, multiplier in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return value * multiplier
        return None

    async def caption(
        self,
        image_data: bytes,
        *,
        prompt: str | None = None,
        max_tokens: int = 300,
    ) -> str:
        """Generate a caption for an image using vision model.

        Args:
            image_data: Raw image bytes (JPEG, PNG, WebP, etc.)
            prompt: Custom captioning prompt (uses DEFAULT_CAPTION_PROMPT if None)
            max_tokens: Maximum tokens in response

        Returns:
            Generated caption text

        Raises:
            ValueError: If image_data is invalid
            RuntimeError: If API call fails after all retries
        """
        if not image_data:
            raise ValueError("image_data cannot be empty")

        # Encode image to base64
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        # Use default prompt if not provided
        caption_prompt = prompt or self.DEFAULT_CAPTION_PROMPT

        for attempt in range(self._max_retries + 1):
            try:
                # Build message with vision content
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": caption_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ]

                # Make API call with optional rate limiting
                if self._semaphore is not None:
                    async with self._semaphore:
                        response = await self._client.chat.completions.create(
                            model=self._model,
                            messages=messages,
                            max_tokens=max_tokens,
                        )
                        return response.choices[0].message.content or ""
                else:
                    # No rate limiting - make request directly
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content or ""

            except Exception as e:
                # Check if it's a rate limit error
                is_rate_limit = isinstance(e, RateLimitError) or (
                    "rate" in str(e).lower() and "limit" in str(e).lower()
                )

                if not is_rate_limit or attempt >= self._max_retries:
                    raise  # Not a rate limit or exhausted retries

                # Calculate wait time: server-recommended or exponential backoff
                error_msg = str(e)
                retry_after = self._parse_retry_after(error_msg)

                if retry_after is not None:
                    wait_time = retry_after + 1  # Add 1s buffer
                else:
                    # Exponential backoff: 3s, 6s, 12s, 24s, 48s...
                    wait_time = self._base_retry_delay * (2**attempt)

                # Add jitter to prevent thundering herd (75-125% of wait_time)
                jitter_factor = random.random() * 0.5 + 0.75
                wait_time = wait_time * jitter_factor

                print(
                    f"Vision API rate limit hit (attempt {attempt + 1}/{self._max_retries + 1}). "
                    f"Waiting {wait_time:.2f}s before retry..."
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError("Unexpected end of retry loop")


class OpenRouterVisionModel(VisionModel):
    """Vision model using OpenRouter's native SDK for image captioning.

    Supports vision-capable models through OpenRouter's unified API.
    """

    DEFAULT_CAPTION_PROMPT = (
        "This is a figure/chart/diagram from a scientific research paper. "
        "Describe what this image represents in detail, focusing on the actual content and data. "
        "If it's a plot/chart: provide specific metric values, axis labels, trends, and comparisons shown. "
        "If it's a diagram/architecture: describe the components, connections, and system structure. "
        "If it's a table: extract key data points and comparisons. "
        "Be specific and detailed (3-5 sentences), as this will be used for information retrieval."
    )

    def __init__(
        self,
        *,
        model: str = "qwen/qwen3-vl-235b-a22b-instruct",
        api_key: str | None = None,
        max_retries: int = 5,
        base_retry_delay: float = 3.0,
        max_concurrent: int = 5,
    ) -> None:
        """Initialize OpenRouter vision model.

        Args:
            model: Model identifier (e.g., "qwen/qwen3-vl-235b-a22b-instruct")
            api_key: OpenRouter API key (fallback to OPENROUTER_API_KEY env var)
            max_retries: Maximum retry attempts on rate limit errors
            base_retry_delay: Base delay for exponential backoff (seconds)
            max_concurrent: Maximum concurrent API requests
        """
        # Check if OpenRouter SDK is available
        if not OPENROUTER_AVAILABLE:
            raise ImportError(
                "openrouter package is required for OpenRouterVisionModel. "
                "Install with: pip install openrouter"
            )

        # Resolve API key
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            dotenv_vars = _load_dotenv()
            key = dotenv_vars.get("OPENROUTER_API_KEY")

        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is required. Set via env variable or pass as api_key parameter."
            )

        self._api_key = key
        self._model = model
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay

        # Rate limiting
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

    async def caption(
        self,
        image_data: bytes,
        *,
        prompt: str | None = None,
        max_tokens: int = 300,
    ) -> str:
        """Generate a caption for an image using OpenRouter vision model.

        Args:
            image_data: Raw image bytes (JPEG, PNG, WebP, etc.)
            prompt: Custom captioning prompt (uses DEFAULT_CAPTION_PROMPT if None)
            max_tokens: Maximum tokens in response

        Returns:
            Generated caption text

        Raises:
            ValueError: If image_data is invalid
            RuntimeError: If API call fails after all retries
        """
        if not image_data:
            raise ValueError("image_data cannot be empty")

        # Encode image to base64
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        # Use default prompt if not provided
        caption_prompt = prompt or self.DEFAULT_CAPTION_PROMPT

        # Build message with vision content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ]

        # Retry loop with exponential backoff
        for attempt in range(self._max_retries + 1):
            try:
                # Make API call with optional rate limiting
                if self._semaphore is not None:
                    async with self._semaphore:
                        response = await self._make_request(messages, max_tokens)
                        return response
                else:
                    response = await self._make_request(messages, max_tokens)
                    return response

            except Exception as e:
                error_str = str(e).lower()

                # Check if it's a retryable error
                is_rate_limit = "rate" in error_str and "limit" in error_str
                is_server_error = any(
                    code in error_str
                    for code in ["500", "502", "503", "504", "429", "cloudflare"]
                )
                is_validation_error = "validation error" in error_str
                is_connection_error = any(
                    phrase in error_str
                    for phrase in [
                        "peer closed connection",
                        "incomplete chunked read",
                        "connection reset",
                        "remotprotocolerror",
                    ]
                )

                # Validation errors often mean API returned error response instead of success
                # Check if error message contains actual API error
                if is_validation_error and "error" in error_str:
                    is_server_error = True  # Treat as retryable

                is_retryable = is_rate_limit or is_server_error or is_connection_error

                if not is_retryable or attempt >= self._max_retries:
                    raise  # Not retryable or exhausted retries

                # Determine error type for logging
                if is_rate_limit:
                    error_type = "Rate limit"
                elif "502" in error_str or "cloudflare" in error_str:
                    error_type = "Cloudflare/Server error (502)"
                elif "500" in error_str or "503" in error_str:
                    error_type = "Server error"
                elif is_connection_error:
                    error_type = "Connection error (peer closed)"
                elif is_validation_error:
                    error_type = "API error (validation failure)"
                else:
                    error_type = "Transient error"

                # Calculate wait time with exponential backoff
                wait_time = self._base_retry_delay * (2**attempt)

                # Add jitter to prevent thundering herd (75-125% of wait_time)
                jitter_factor = random.random() * 0.5 + 0.75
                wait_time = wait_time * jitter_factor

                print(
                    f"OpenRouter vision {error_type} (attempt {attempt + 1}/{self._max_retries + 1}). "
                    f"Waiting {wait_time:.2f}s before retry..."
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError("Unexpected end of retry loop")

    async def _make_request(self, messages: list[dict], max_tokens: int) -> str:
        """Make the actual API request to OpenRouter.

        Args:
            messages: List of message dictionaries with vision content
            max_tokens: Maximum tokens in response

        Returns:
            Response text from the model

        Raises:
            Exception: If API returns an error or response is invalid
        """
        # Use OpenRouter SDK with context manager (required for proper auth)
        async with OpenRouter(api_key=self._api_key) as client:
            try:
                response = await client.chat.send_async(
                    messages=messages,
                    model=self._model,
                    stream=False,
                )

                # Extract response text
                if hasattr(response, "choices") and response.choices:
                    return response.choices[0].message.content or ""
                else:
                    # Response object doesn't have expected structure
                    raise RuntimeError(f"Unexpected response format: {response}")

            except Exception as e:
                # Check if it's a validation error with error details
                error_str = str(e)
                if (
                    "validation errors" in error_str.lower()
                    or "error" in error_str.lower()
                ):
                    # This might be an API error response that the SDK couldn't parse
                    # Re-raise with more context
                    raise RuntimeError(
                        f"OpenRouter API error (possibly Cloudflare or rate limit): {error_str}"
                    ) from e
                raise


class HuggingFaceLocalVisionModel(VisionModel):
    """Vision model loaded directly from HuggingFace for local inference.

    Runs entirely on-device — no API keys, no network calls after initial
    model download. Designed for one-shot batch jobs (e.g. figure verification
    during index build) where you don't want to maintain a long-running
    VLM endpoint.

    Recommended models (larger = better figure/table understanding):
    - Qwen/Qwen2.5-VL-72B-Instruct (~145 GB bf16, ~40 GB 4-bit) — best quality
    - Qwen/Qwen2.5-VL-32B-Instruct (~65 GB bf16, ~20 GB 4-bit)  — strong, less VRAM
    - Qwen/Qwen2.5-VL-7B-Instruct  (~16 GB bf16, ~8 GB 4-bit)   — lightweight fallback

    Usage:
        vlm = HuggingFaceLocalVisionModel(model="Qwen/Qwen2.5-VL-72B-Instruct")
        caption = await vlm.caption(image_bytes)
    """

    # Read-only PVC path where model weights live
    _PVC_HF_CACHE = "/models/.cache/huggingface"
    # Writable overlay for HF metadata (tokenizer caches, refs, etc.)
    _WRITABLE_HF_HOME = "/tmp/hf_home_vision"

    DEFAULT_CAPTION_PROMPT = (
        "This is a figure/chart/diagram from a scientific research paper. "
        "Describe what this image represents in detail, focusing on the actual content and data. "
        "If it's a plot/chart: provide specific metric values, axis labels, trends, and comparisons shown. "
        "If it's a diagram/architecture: describe the components, connections, and system structure. "
        "If it's a table: extract key data points and comparisons. "
        "Be specific and detailed (3-5 sentences), as this will be used for information retrieval."
    )

    @staticmethod
    def _setup_cache_overlay():
        """Create a writable HF cache overlay pointing to the read-only shared PVC.

        Same pattern used by embedding_server.py — symlinks model weight dirs
        from the PVC into a writable /tmp location so HuggingFace can write
        metadata files (tokenizer caches, refs, .no_exist) without needing
        write access to the PVC or network access (HF_HUB_OFFLINE=1 is fine).
        """
        import shutil

        pvc = HuggingFaceLocalVisionModel._PVC_HF_CACHE
        writable = HuggingFaceLocalVisionModel._WRITABLE_HF_HOME

        if not os.path.isdir(pvc):
            return  # No PVC mounted, use whatever HF_HOME is already set

        os.makedirs(writable, exist_ok=True)

        for entry in os.listdir(pvc):
            src = os.path.join(pvc, entry)
            if entry.startswith("models--") and os.path.isdir(src):
                model_dir = os.path.join(writable, entry)
                # Create writable dirs for metadata HF wants to write
                os.makedirs(os.path.join(model_dir, "snapshots"), exist_ok=True)
                os.makedirs(os.path.join(model_dir, "refs"), exist_ok=True)
                os.makedirs(os.path.join(model_dir, ".no_exist"), exist_ok=True)

                # Symlink each snapshot hash dir (actual model weights)
                snap_src = os.path.join(src, "snapshots")
                if os.path.isdir(snap_src):
                    for snap in os.listdir(snap_src):
                        snap_dst = os.path.join(model_dir, "snapshots", snap)
                        if not os.path.exists(snap_dst):
                            os.symlink(os.path.join(snap_src, snap), snap_dst)

                # Copy refs (tiny text files) so HF can overwrite them
                refs_src = os.path.join(src, "refs")
                if os.path.isdir(refs_src):
                    for ref in os.listdir(refs_src):
                        ref_dst = os.path.join(model_dir, "refs", ref)
                        if not os.path.exists(ref_dst):
                            shutil.copy2(os.path.join(refs_src, ref), ref_dst)

                # Create writable .locks dir
                locks_dst = os.path.join(model_dir, ".locks")
                os.makedirs(locks_dst, exist_ok=True)

            elif entry == ".locks":
                os.makedirs(os.path.join(writable, entry), exist_ok=True)
            elif not os.path.exists(os.path.join(writable, entry)):
                os.symlink(src, os.path.join(writable, entry))

        os.environ["HF_HOME"] = writable
        os.environ["HF_HUB_CACHE"] = writable
        os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf_modules_vision")
        print(f"[vision] Writable HF cache overlay at {writable}", flush=True)

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        dtype: str = "4bit",
        max_concurrent: int = 1,
    ) -> None:
        """Initialize local HuggingFace vision model.

        Args:
            model: HuggingFace model identifier or local path.
                   Must be a vision-language model with AutoProcessor support.
            dtype: Torch dtype — "bf16", "fp16", "4bit", or "auto"
            max_concurrent: Maximum concurrent inference calls (default 1 for
                          GPU memory safety; increase only if you have headroom)
        """
        if not HF_VISION_AVAILABLE:
            raise ImportError(
                "transformers, torch, and Pillow are required for "
                "HuggingFaceLocalVisionModel. Install with:\n"
                "  pip install torch transformers accelerate Pillow qwen-vl-utils"
            )

        # Set up writable HF cache overlay if models are on a read-only PVC.
        # This lets HF write metadata (tokenizer caches, config parsing) to /tmp
        # while reading actual model weights from the shared PVC via symlinks.
        self._setup_cache_overlay()

        self._model_id = model
        self._semaphore = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

        print(f"[vision] Loading processor for {model}...", flush=True)
        self._processor = AutoProcessor.from_pretrained(self._model_id)

        # Resolve dtype and quantization
        load_kwargs: dict = {"device_map": "auto"}
        if dtype == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "4-bit quantization requires bitsandbytes. "
                    "Install with: pip install bitsandbytes"
                )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif dtype == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif dtype == "fp16":
            load_kwargs["torch_dtype"] = torch.float16

        print(f"[vision] Loading model weights ({dtype})...", flush=True)
        self._model = _VisionAutoModel.from_pretrained(
            self._model_id,
            **load_kwargs,
        )
        print(f"[vision] Model loaded on {self._model.device}", flush=True)

    def _sync_caption(
        self, image_data: bytes, prompt: str, max_tokens: int
    ) -> str:
        """Synchronous caption generation (runs on GPU)."""
        # Load image
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Build chat-style messages with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text_input],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        # Decode only the generated tokens (skip input tokens)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        result = self._processor.decode(generated, skip_special_tokens=True)
        return result.strip()

    async def caption(
        self,
        image_data: bytes,
        *,
        prompt: str | None = None,
        max_tokens: int = 300,
    ) -> str:
        """Generate a caption for an image using local vision model.

        Args:
            image_data: Raw image bytes (JPEG, PNG, WebP, etc.)
            prompt: Custom captioning prompt (uses DEFAULT_CAPTION_PROMPT if None)
            max_tokens: Maximum tokens in response

        Returns:
            Generated caption text
        """
        if not image_data:
            raise ValueError("image_data cannot be empty")

        caption_prompt = prompt or self.DEFAULT_CAPTION_PROMPT

        if self._semaphore is not None:
            async with self._semaphore:
                return await asyncio.to_thread(
                    self._sync_caption, image_data, caption_prompt, max_tokens,
                )
        return await asyncio.to_thread(
            self._sync_caption, image_data, caption_prompt, max_tokens,
        )
