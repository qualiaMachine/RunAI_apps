# Batch Processing Workspace (`ocr-batch`)

> **Step 4** in the [deployment guide](README.md). Comes after
> [Deploy vLLM Server](deploy-vllm.md) (Step 3).

Production workspace for processing large document collections. Mounts
the document PVC and runs the batch script against vLLM.

> **Prerequisite:** You've already verified the pipeline works on sample
> docs in the setup workspace (Step 2). If you haven't, do that first.

---

In the RunAI UI: **Workloads** > **New Workload** > **Workspace**

## Basic settings

| Field | Value |
|-------|-------|
| **Cluster** | `doit-ai-cluster` |
| **Project** | Your project |
| **Workspace name** | `ocr-batch` |

## Environment image

| Field | Value |
|-------|-------|
| **Image** | Custom image |
| **Image URL** | `nvcr.io/nvidia/pytorch:25.02-py3` |
| **Image pull** | Pull the image only if it's not already present on the host |

## Tools

| Field | Value |
|-------|-------|
| **Tool type** | Jupyter |
| **Port** | `8888` |

## Runtime settings

| Field | Value |
|-------|-------|
| **Command** | `bash` |
| **Arguments** | See below |
| **Working directory** | *(leave empty)* |

### Arguments (copy-paste)

```
-c "pip install uv && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && curl -sL https://github.com/qualiaMachine/RunAI_apps/archive/refs/heads/main.tar.gz | tar xz -C /tmp && mv /tmp/RunAI_apps-main /tmp/RunAI_apps && cd /tmp/RunAI_apps && uv pip install --system httpx pymupdf Pillow && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
```

**Environment variables:**

| Name | Value |
|------|-------|
| `LLM_BASE_URL` | `http://qwen3--vl--32b--instruct.runai-<project>.svc.cluster.local/v1` |
| `VLM_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` |

## Compute resources

| Field | Value |
|-------|-------|
| **GPU devices** | `0` (none — all GPU work is in qwen3--vl--32b--instruct) |

## Data & storage

| Data volume name | Container path |
|------------------|----------------|
| `ocr-documents` | `/data/documents` |
| `ocr-extracted` | `/data/extracted` |

---

## Running the batch script

Open a terminal in the Jupyter workspace:

```bash
cd /tmp/RunAI_apps

# Process all PDFs — award notices
python ocr_app/scripts/batch_extract.py \
    --input-dir /data/documents \
    --output-dir /data/extracted \
    --format award \
    --concurrency 4
```

### Other examples

```bash
# Process only TIFFs
python ocr_app/scripts/batch_extract.py \
    --input-dir /data/documents \
    --output-dir /data/extracted \
    --format key_values \
    --extensions .tiff .tif \
    --concurrency 8

# Resume after failure or interruption
python ocr_app/scripts/batch_extract.py \
    --input-dir /data/documents \
    --output-dir /data/extracted \
    --format award \
    --resume

# Process a specific subdirectory
python ocr_app/scripts/batch_extract.py \
    --input-dir /data/documents/2024 \
    --output-dir /data/extracted/2024 \
    --format award \
    --concurrency 4
```

---

## Understanding the output

### Progress logging

```
[batch] Found 45000 files, 0 already completed, 45000 to process
[batch] LLM: Qwen/Qwen3-VL-32B-Instruct at http://qwen3--vl--32b--instruct.../v1
[1/45000] OK award_notice_2019.pdf (3p, 3d/0s, 2.1s) -> award_notice_2019.json
[2/45000] OK budget_fy2020.pdf (5p, 5d/0s, 3.4s) -> budget_fy2020.json
[3/45000] OK scanned_agreement.tiff (1p, 0d/1s, 8.2s) -> scanned_agreement.json
```

- **`3p`** = 3 pages total
- **`3d/0s`** = 3 digital pages, 0 scanned (text extraction used)
- **`0d/1s`** = 0 digital, 1 scanned (VLM OCR used)
- Digital pages are ~10x faster than scanned pages

### Output files

One JSON per input document, preserving subdirectory structure:

```
/data/extracted/
├── subdir_a/
│   ├── doc1.json
│   └── doc2.json
└── subdir_b/
    └── doc3.json
```

### Resume state

The batch script tracks completed files in `<output-dir>/.batch_state`.
When you re-run with `--resume`, it skips files already listed there.
This means:

- Safe to interrupt and resume (Ctrl+C, workspace restart, etc.)
- Safe to re-run against the same input directory — only new files get
  processed
- To start completely fresh, delete `.batch_state`:
  `rm /data/extracted/.batch_state`

---

## Tuning concurrency

| `--concurrency` | Best for |
|-----------------|----------|
| `2-4` | Default. Safe for single vLLM instance on any GPU. |
| `8-16` | If vLLM is on a large GPU (A100 80GB) with headroom. |
| `1` | Debugging. Sequential, easy to read logs. |

vLLM handles batching internally via continuous batching, so higher
concurrency doesn't always mean more throughput — it depends on GPU
memory and model size. Start at 4 and increase if vLLM's GPU utilization
is below 80%.

> **How to check GPU utilization:** From any workspace with the shared
> models PVC mounted, run `nvidia-smi`. Or check the RunAI UI's GPU
> utilization metrics for the `qwen3--vl--32b--instruct` job.

---

## Ongoing ingestion (~10K docs/month)

If your input PVC is a direct mount to the source storage (NFS), new
documents appear automatically. Re-run the batch script with `--resume`:

```bash
python ocr_app/scripts/batch_extract.py \
    --input-dir /data/documents \
    --output-dir /data/extracted \
    --format award \
    --concurrency 4 \
    --resume
```

It skips all previously processed files and only extracts the new ones.
You can run this on a schedule (monthly, weekly, daily) depending on
your intake volume.
