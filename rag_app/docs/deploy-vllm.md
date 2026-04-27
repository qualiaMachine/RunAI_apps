# Deploy the vLLM Server

Uses the official `vllm/vllm-openai` image — no pip-installing vLLM at
runtime.

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

## Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project (e.g. `jupyter-endemann01`) |
| **Inference type** | **Custom** |
| **Inference name** | `wattbot-chat` |

## Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `vllm/vllm-openai:latest` |
| **Image pull** | Pull the image only if it's not already present on the host (recommended) |
| **Image pull secret** | *(leave empty — public Docker Hub image)* |

## Serving endpoint

| Field | Value |
|-------|-------|
| **Protocol** | HTTP |
| **Container port** | `8000` |

## Runtime settings

The `vllm/vllm-openai` image has a built-in entrypoint that launches the
API server — you only need to pass the model as a positional argument. No
command is required.

| Field | Value |
|-------|-------|
| **Command** | *(leave empty — image default launches the API server)* |
| **Arguments** | `Qwen/Qwen3-30B-A3B-GPTQ-Int4 --quantization gptq_marlin --dtype half` |
| **Working directory** | *(leave empty)* |

**Environment variables:**

| Name | Value |
|------|-------|
| `HF_HOME` | `/models/.cache/huggingface` |
| `HF_HUB_CACHE` | `/models/.cache/huggingface` |
| `HF_HUB_OFFLINE` | `1` |

> **Note:** The image defaults to `--host 0.0.0.0` and uses the
> container port from the serving endpoint config. `HF_HUB_OFFLINE=1`
> prevents downloads at runtime — the model must be pre-cached on the
> shared PVC.

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled — set to `80%` of device |
| **CPU request** | *(leave default)* |
| **CPU memory request** | *(leave default)* |
| **Replica autoscaling** | Min `1`, Max `1` (no autoscaling) |

## Data & storage

Under **Data & storage**, select the `shared-models` data volume and
set the container path. (In Custom inference type, data volumes appear
directly in the initial setup form — no need for Advanced setup.)

| Data volume name | Container path |
|------------------|----------------|
| `shared-models` | `/models` |

## General

| Field | Value |
|-------|-------|
| **Priority** | `very-high` (or as appropriate) |

## Expected startup time

First deploy takes **5-10 minutes**:
- **Image pull** (~2-5 min): The `vllm/vllm-openai` image is ~15 GB.
  Subsequent deploys skip this if the image is cached on the node.
- **Model loading** (~1-2 min): vLLM loads OpenScholar 8B weights
  from the shared PVC into GPU memory (bitsandbytes quantized).
- **Engine warmup** (~30s): vLLM compiles CUDA kernels and initializes
  the KV cache.

You'll see `Initializing` in the RunAI UI during this time — this is
normal. The job transitions to `Running` once the HTTP health check
passes. Subsequent restarts (same node, cached image) take ~2-3 minutes.

## How it works

OpenScholar 8B weights are already pre-cached on the shared PVC at
`/models/.cache/huggingface/` — vLLM loads them directly on startup
(no download needed, `HF_HUB_OFFLINE=1`). The Data Volume is read-only,
so vLLM can't accidentally modify or delete weights.

> **Note:** vLLM exposes an **OpenAI-compatible** API (`/v1/chat/completions`),
> but it runs **entirely on your local GPU** — no OpenAI account or API
> charges. The `openai` Python package is just used as a client library
> to talk to your local vLLM server.

**Verify (from any other pod's terminal):**
```bash
curl http://wattbot-chat:8000/v1/models
```

## Switching to OpenScholar 8B

OpenScholar (`OpenSciLM/Llama-3.1_OpenScholar-8B`) is a Llama 3.1 8B
fine-tune trained for scientific literature synthesis. It's a drop-in
replacement for Qwen 7B — same VRAM footprint (~16 GB bf16, ~6 GB 4-bit).

1. Download the model to the shared PVC (see [Managing Models](managing-models.md)):
   ```bash
   python /models/provision_shared_models.py download OpenSciLM/Llama-3.1_OpenScholar-8B
   ```
2. Change the vLLM job's **Arguments** to:
   `--model OpenSciLM/Llama-3.1_OpenScholar-8B`
3. Change the Streamlit job's `VLLM_MODEL` env var to:
   `OpenSciLM/Llama-3.1_OpenScholar-8B`
4. Restart both jobs

No other changes needed — the embedding model, vector DB, and retrieval
pipeline are unchanged.

## GPU VRAM and quantization

The current deployment uses **bitsandbytes** quantization, which reduces
memory enough to fit an 8B model in 80% of a GPU. This is the recommended
setup for fractional GPU allocation.

| GPU VRAM | Flag | Notes |
|----------|------|-------|
| 80 GB (A100) | `--quantization bitsandbytes --load-format bitsandbytes` | Current setup, 80% fraction |
| 40 GB (A6000) | `--quantization bitsandbytes --load-format bitsandbytes` | |
| 24 GB (L4/4090) | `--quantization bitsandbytes --load-format bitsandbytes --max-model-len 4096` | |

## Model arguments reference

Copy-paste the **Arguments** field for each model. Remember to also update
the Streamlit job's `VLLM_MODEL` env var to match, and download the model
to the shared PVC first (see [Managing Models](managing-models.md)).

### Qwen3-30B-A3B (GPTQ 4-bit — official pre-quantized, MoE)

```
Qwen/Qwen3-30B-A3B-GPTQ-Int4 --quantization gptq_marlin --dtype half
```

> Mixture-of-Experts: 30B total params, ~3B active per token. Faster
> inference than the dense 32B at similar VRAM cost (~18 GB at 4-bit).
> Uses `gptq_marlin` for faster inference (vLLM warns the plain `gptq`
> kernel is buggy).

### Qwen 2.5 72B (AWQ 4-bit — official pre-quantized)

```
Qwen/Qwen2.5-72B-Instruct-AWQ --quantization awq_marlin --dtype half
```

> Largest dense model we run. Needs a full GPU (~39 GB VRAM at 4-bit).
> No fractional GPU sharing — requires 100% of an 80 GB A100.
> Uses `awq_marlin` for faster inference.

### Qwen3-32B (AWQ 4-bit — official pre-quantized)

```
Qwen/Qwen3-32B-AWQ --quantization awq_marlin --dtype half
```

> Dense 32B model. Needs a full GPU (~20 GB VRAM at 4-bit). Strong
> reasoning; rivals Llama 3.1 70B on many benchmarks at half the memory.
> Uses `awq_marlin` for faster inference.

### OpenScholar 8B (BitsAndBytes — no pre-quantized version available)

```
OpenSciLM/Llama-3.1_OpenScholar-8B --quantization bitsandbytes --load-format bitsandbytes --dtype half
```

> OpenScholar has no official AWQ/GPTQ release, so we fall back to
> BitsAndBytes for 4-bit quantization. This requires `--load-format
> bitsandbytes` in addition to `--quantization`.

### Qwen 2.5 7B (BitsAndBytes)

```
Qwen/Qwen2.5-7B-Instruct --quantization bitsandbytes --load-format bitsandbytes --dtype auto
```

> Original default model. Smallest footprint (~6 GB at 4-bit), fits
> easily in a fractional GPU allocation.

---

## CLI equivalent

If you prefer the CLI over the UI:

```bash
runai submit wattbot-chat \
  --type inference \
  --image vllm/vllm-openai:latest \
  --gpu 0.80 \
  --cpu 4 \
  --memory 16Gi \
  --pvc shared-models:/models \
  --env HF_HOME=/models/.cache/huggingface \
  --env HF_HUB_CACHE=/models/.cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --port 8000 \
  -- Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
    --quantization gptq_marlin \
    --dtype half
```
