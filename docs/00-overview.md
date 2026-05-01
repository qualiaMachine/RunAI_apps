# 00 — Overview

> **Step 0** in the [New User Guide](../README.md#new-user-guide). Read this first to
> decide whether the cluster is the right tool, and to learn the three
> concepts the rest of the guide assumes.

## What the cluster is for

The RunAI cluster (`doit-ai-cluster`) is a small DoIT pilot — two
NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (96 GB GDDR7 each,
connected by PCIe Gen5 — no NVLink), big enough to host models up
to ~150B parameters when quantized and split across both cards. RunAI
is the scheduling layer on top; if you've used a major cloud's
notebook + endpoint UI, the experience is similar (workspaces,
fractional GPU allocation, autoscaling endpoints). See
[Hardware](#hardware) below for what the two-GPU layout does and
doesn't buy you.

This hardware+RunAI may help fill the following niches:

- **Long-term LLM/VLM apps the institution wants to support.** When a
  service is going to live for years, renting cloud capacity
  permanently doesn't make sense.
- **Long-running training or fine-tuning jobs.** These get expensive
  fast in the cloud. Sometimes doable in CHTC, but requires batching and smaller LLMs.
- **Sensitive / PHI / Institutional data workflows.** Pending the relevant
  cybersecurity reviews.
- **Sharing model endpoints across many users.** Large models live in one place; and many users can easily access them for their work. Graceful GPU scheduling and scaling means that GPUs can be used efficiently.


## Hardware

Two NVIDIA RTX PRO 6000 Blackwell **Server Edition** GPUs in a single
Dell PowerEdge chassis.

| | |
|---|---|
| Per-GPU memory | 96 GB GDDR7, ~1.6 TB/s memory bandwidth |
| Per-GPU compute | Blackwell GB202, 5th-gen Tensor cores (FP4 / FP6 / FP8 / BF16) |
| Per-GPU TDP | 600 W, passive 2-slot (server cooling) |
| MIG | Supported — up to 4 partitions per GPU (8 across the box) |
| GPU ↔ GPU link | **PCIe Gen5 x16 only — no NVLink, no NVSwitch** (~64 GB/s per direction; PCIe peer-to-peer enabled) |

The PCIe-only interconnect is the headline trade-off versus an
NVLink-class server (H100 SXM has roughly 14× the inter-GPU
bandwidth). It shapes what workloads fit well.

**Sweet spot for this cluster:**

- **Two independent single-GPU workloads** running side by side — e.g.
  one ~70B-class model hosted per card, or one big model plus a stack
  of fractional services. No inter-GPU traffic, no PCIe tax.
- **Fractional GPU sharing** (RunAI's default model) and **MIG
  partitioning** for many concurrent small workloads or users — all
  per-card, never crosses the link.
- **Pipeline parallelism** across the two GPUs — only communicates at
  layer boundaries, fine over PCIe.
- **Tensor-parallel-2 inference** of large quantized models with vLLM
  (`--tensor-parallel-size 2`) — each transformer layer all-reduces
  over PCIe, so expect roughly **20–40% lower per-GPU throughput** than
  the same setup would deliver on NVLink. Still typically a net win
  over single-GPU because you get 2× VRAM and 2× compute. Reasonable
  targets: Llama-3.1-70B / Qwen2.5-72B at FP8 or INT4 with comfortable
  KV-cache headroom, Qwen3-VL-72B at AWQ, ~150B-class models at 4-bit.

**Don't plan around it:**

- **Multi-GPU pretraining or full-parameter fine-tuning of 70B+
  models.** Gradient all-reduce saturates PCIe; this is the wrong tool unless future investment involves NVLinked GPUs.


## The three concepts you actually need

Every later doc assumes you've internalized these. They map directly to
RunAI UI sections.

### 1. Workload — your running thing

Three flavors:

| Type | What it is | Typical use |
|------|------------|-------------|
| **Workspace** | An interactive pod with a UI (Jupyter, VS Code, custom URL). Persists when you close the browser; you Stop/Start it manually. | Notebooks, terminals, dev work, batch scripts you babysit |
| **Training** | A pod that runs a job to completion and then exits. | Fine-tuning, large batch jobs, scheduled pipelines |
| **Inference** | An autoscaling pod (or pods) behind a network endpoint. | Hosting a model that other workloads call over HTTP |

Almost everything in this repo's tutorials starts with a **Workspace**.

### 2. Data Source — storage you mount into a workload

A pointer to "data lives here, mount it at this path." Types include
**PVC** — *Persistent Volume Claim*, Kubernetes' name for a chunk of
cluster-managed disk that survives pod restarts — plus **NFS**
(network share), **S3**, **Git**, and a few others. Per-project.
You'll create one per dataset, model, or output location your workload
reads or writes.

> **PVC is the term you'll see most.** The rest of these docs
> (storage, the per-app deploy guides) say "PVC" rather than spelling
> it out — when you see things like "shared PVC", "writable PVC", or
> "read-only PVC mount", that's the same Persistent Volume Claim
> concept introduced here.

> **You already get one for free — if you load the right template.**
> Every project on this cluster has a `user-workspace` workload
> template with a pre-attached ~30 GB persistent PVC. Choose
> **Load from existing** > **`user-workspace`** when creating a
> workspace and notebooks survive across stop/start cycles and
> workspace deletes without you setting anything up. The
> [Storage doc](04-storage.md) Step A shows the flow; 02's
> "Start from scratch" path opts out of this on purpose to keep
> the moving parts visible.

### 3. Data Volume — a shared, read-only wrapper around a populated PVC

Lets one project's PVC be mounted by **other projects** read-only. The
origin project keeps RW. Used for things like a 20 GB model cache that
every research group on the cluster wants to use without each
re-downloading it.

> **Don't conflate Data Source and Data Volume.** Almost everything you
> mount in your own project is a **Data Source**. Data Volume is the
> "share what I've built across the whole cluster" tier. The
> [Storage doc](04-storage.md) walks through the difference with a
> hands-on example.



## Next

Head to [01 Access](01-access.md) to get logged into the cluster.

