# Deploy the vLLM Server (`qwen3--vl--32b--instruct-awq`)

> **Step 3** in the [deployment guide](README.md). Deploy this when
> you're ready for a persistent inference endpoint (after iterating
> in the setup workspace).

Uses the official `vllm/vllm-openai` image. Serves Qwen3-VL-32B-Instruct
(AWQ 4-bit quantized, ~20 GB) for the notebook chunk-based pipeline.

In the RunAI UI: **Workloads** > **New Workload** > **Inference**

---

## Page 1: Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | `shared-models` (shared endpoint for all users) |
| **Template** | Start from scratch |
| **Inference type** | **Hugging Face** |
| **Inference name** | `qwen3--vl--32b--instruct-awq` |

Click **Continue**.

---

## Page 2: Advanced configuration

### Model

| Field | Value |
|-------|-------|
| **Model** | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` |

### Environment

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `vllm/vllm-openai:latest` |

### Serving endpoint

Leave defaults (HTTP, container port auto-detected).

### Runtime settings

| Field | Value |
|-------|-------|
| **Command** | *(leave empty)* |
| **Arguments** | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ --quantization awq_marlin --dtype half --max-model-len 16384` |

**Environment variables** (click **+ Environment Variable** for each):

| Name | Source | Value |
|------|--------|-------|
| `HF_HOME` | Custom | `/models/.cache/huggingface` |
| `HF_HUB_CACHE` | Custom | `/models/.cache/huggingface` |
| `HF_HUB_OFFLINE` | Custom | `1` |

**Working directory:** *(leave empty)*

> **About the arguments:**
> - `--quantization awq_marlin` uses the optimized Marlin kernel for
>   AWQ weights (~6-10x faster generation than plain `awq`)
> - `--dtype half` = fp16 compute (required for AWQ on most GPUs)
> - `--max-model-len 16384` caps KV cache allocation to 16K tokens.
>   The model supports 262K natively but that would need 64+ GB just
>   for KV cache. 16K covers a default notebook chunk (~20 page images
>   at 2x render, well under 16K input tokens) plus the JSON response.
>   If you bump `MAX_PAGES_PER_CHUNK` in the notebook and start seeing
>   `finish_reason == "length"`, raise this.
> - `HF_HUB_OFFLINE=1` prevents any model downloads — loads from the
>   shared PVC only

### Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled |
| **GPU memory** | `% of device` — Request: `75` |

> **Memory breakdown (AWQ 32B, 96 GB GPU at 75%):**
> - Model weights: ~19.5 GB
> - KV cache (16K context): ~8 GB
> - CUDA graphs: ~1.3 GB
> - Overhead: ~1-2 GB
> - **Total: ~30 GB / 72 GB allocated** (ample headroom for load spikes and concurrent requests)
>
> Lower fractions (e.g. 35-40%) are tight — AWQ weight-load peaks and
> activation spikes under concurrency can push past the cap and cause
> the pod to fail readiness. Stay at 75% unless GPU is contended. For
> the full bf16 model (`Qwen/Qwen3-VL-32B-Instruct --dtype auto`), use
> 85%.

### Data & storage

Click **+ Data Volume**:

| Field | Value |
|-------|-------|
| **Data volume name** | `shared-models` |
| **Container path** | `/models` |

### Autoscaling

Configured to handle multiple concurrent users:

| Field | Value |
|-------|-------|
| **Min replicas** | `1` |
| **Max replicas** | `2` |
| **Scale-up variable** | `Throughput (Requests/sec)` |
| **Operator** | `>` |
| **Value** | `1` |

> vLLM's continuous batching handles concurrency within a single replica
> efficiently. Scale-up at 1 req/sec covers the case of 5+ simultaneous
> users or one very heavy batch job.

### General

| Field | Value |
|-------|-------|
| **Priority** | `very-high` |
| **Preemptibility** | `non-preemptible` |

Click **Create Inference**.

---

## Verify

Wait for the job to reach `Running` state (2-5 min), then test from any
workspace on the cluster:

```bash
curl http://qwen3--vl--32b--instruct-awq.runai-shared-models.svc.cluster.local/v1/models
```

Expected:
```json
{"data": [{"id": "QuantTrio/Qwen3-VL-32B-Instruct-AWQ", ...}]}
```

> **FQDN required.** Use `workload-name.runai-project.svc.cluster.local`
> on port 80 (no port number). Knative envoy requires this. The notebook
> defaults to `runai-shared-models` — users in any project can reach it
> via this FQDN without any config changes.

If it doesn't respond:
- Check the job status in RunAI UI — is it still `Initializing`?
- Click the job → **Logs** tab for errors
- See [Troubleshooting](troubleshooting.md)

---

## GPU sizing reference

| GPU | Arguments | GPU fraction |
|-----|-----------|-------------|
| **96 GB (AWQ, default)** | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ --quantization awq_marlin --dtype half --max-model-len 16384` | **75%** |
| A100 80GB (AWQ) | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ --quantization awq_marlin --dtype half --max-model-len 16384` | 75% |
| 96 GB (bf16, full) | `Qwen/Qwen3-VL-32B-Instruct --dtype auto --max-model-len 16384` | 85% |
| A100 80GB (bf16) | `Qwen/Qwen3-VL-32B-Instruct --dtype auto --max-model-len 16384` | 85% |
| L4/RTX 4090 24GB | `Qwen/Qwen3-VL-8B-Instruct --dtype auto --max-model-len 16384` | 100% |
