# Deploy the vLLM Server (`qwen3--vl--32b--instruct-awq`)

> **Step 3** in the [deployment guide](README.md). Deploy this when
> you're ready for a persistent inference endpoint (after iterating
> in the setup workspace).

Uses the official `vllm/vllm-openai` image. Serves Qwen3-VL-32B-Instruct
(AWQ 4-bit quantized, ~20 GB) for both text parsing (digital PDFs) and
VLM OCR (scans/TIFFs) — one model handles both paths.

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
| **Arguments** | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ --quantization awq --dtype half` |

**Environment variables** (click **+ Environment Variable** for each):

| Name | Source | Value |
|------|--------|-------|
| `HF_HOME` | Custom | `/models/.cache/huggingface` |
| `HF_HUB_CACHE` | Custom | `/models/.cache/huggingface` |
| `HF_HUB_OFFLINE` | Custom | `1` |

**Working directory:** *(leave empty)*

### Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `1` |
| **GPU fractioning** | Enabled |
| **GPU memory** | `% of device` — Request: `25` |

> **Note:** The AWQ 4-bit model needs ~20 GB VRAM. 25% of a 96 GB GPU
> (~24 GB) is sufficient for sliding window extraction (3 images per
> call). Increase to 35% if you need longer context or larger batches. For the full bf16 model
> (`Qwen/Qwen3-VL-32B-Instruct --dtype auto`), use 75-85%.

### Data & storage

Click **+ Data Volume**:

| Field | Value |
|-------|-------|
| **Data volume name** | `shared-models` |
| **Container path** | `/models` |

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
curl http://qwen3--vl--32b--instruct-awq.runai-<project>.svc.cluster.local/v1/models
```

Expected:
```json
{"data": [{"id": "QuantTrio/Qwen3-VL-32B-Instruct-AWQ", ...}]}
```

> **FQDN required.** Use `workload-name.runai-project.svc.cluster.local`
> on port 80 (no port number). Knative envoy requires this.

If it doesn't respond:
- Check the job status in RunAI UI — is it still `Initializing`?
- Click the job → **Logs** tab for errors
- See [Troubleshooting](troubleshooting.md)

---

## GPU sizing

If you hit OOM errors, adjust the **Arguments** and **GPU fraction**:

| GPU | Arguments | GPU fraction |
|-----|-----------|-------------|
| 96 GB (AWQ, default) | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ --quantization awq --dtype half` | 25% |
| A100 80GB (AWQ) | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ --quantization awq --dtype half` | 30% |
| 96 GB (bf16, full) | `Qwen/Qwen3-VL-32B-Instruct --dtype auto` | 75% |
| A100 80GB (bf16) | `Qwen/Qwen3-VL-32B-Instruct --dtype auto` | 85% |
| L4/RTX 4090 24GB | `Qwen/Qwen3-VL-8B-Instruct --dtype auto` | 100% |
