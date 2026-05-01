# 00b — Hardware

> Companion to [00 Overview](00-overview.md). Read this if you're sizing
> a workload, deciding between single-GPU and tensor-parallel
> deployment, or comparing this cluster to NVLink-class hardware
> elsewhere.

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

## Sweet spot for this cluster

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

## Don't plan around it

- **Multi-GPU pretraining or full-parameter fine-tuning of 70B+
  models.** Gradient all-reduce saturates PCIe; this is the wrong tool.
  Use H100/H200 NVLink hardware (CHTC, cloud) for that.
- **Tensor-parallel scaling beyond 2 GPUs.** Not applicable here, but
  worth flagging if anyone extrapolates from "TP-2 works" to "TP-N will
  scale linearly" — it won't, and there's no third GPU anyway.

When in doubt, prefer one model per GPU over splitting one model
across both.

## Next

Continue to [01 Access](01-access.md) to get logged into the cluster.
