# Shared Model Catalogue — Draft Plan

> **Status: draft / not yet stood up.** This doc proposes which models
> the pilot hosts as *standing shared endpoints* on the cluster's two
> GPUs, how we track whether they're still the right models, and what
> triggers a swap. It assumes the concepts from
> [00 Overview](00-overview.md) (hardware, workload types) and the
> serving patterns from [03 Share as endpoint](03-share-as-endpoint.md).
> Constraints from the [Usage Policy](usage-policy.md) apply — public
> data only during the pilot.

## Why a catalogue at all

Individual labs standing up their own copies of the same 7B model is
the failure mode this cluster exists to avoid. A small, curated set of
always-on endpoints — one URL per capability, OpenAI-compatible —
means researchers integrate against a stable API instead of learning
RunAI, and the two GPUs serve many groups instead of two.

The catalogue is also the unit of collaboration: if NRP or peer
universities stand up their own "BadgerBrain-like" deployments, a
published catalogue (models + endpoints + measured performance) is
what makes federation possible later. See
[Growth path](#growth-path-nrp-and-peer-campuses).

---

## The core trade-off: variety vs. replicas

With 2× 96 GB and a campus-wide audience, we can either host **more
distinct capabilities** or **more copies of fewer models**.

**Recommendation: optimize for variety. Do not budget standing
replicas at this scale.**

Rationale:

1. **vLLM already multiplexes users.** Continuous batching means one
   replica of a ~30B quantized model serves tens of concurrent
   requests; the real ceiling is KV-cache memory, not replica count.
   On a 96 GB card, a ~20–35 GB (AWQ/FP8) model leaves 40–60 GB of
   KV cache — that *is* the multi-user capacity.
2. **A replica costs a whole capability.** Duplicating the generalist
   across both cards buys roughly 2× throughput but evicts the
   embedding/reranker/vision stack — the things that make the cluster
   useful for RAG and document pipelines, which is where campus demand
   actually is.
3. **There are cheaper levers than replicas** when an endpoint
   saturates, in order: tighten quantization (the repo's
   [8-bit vs 4-bit notebook](../8bit-vs-4bit-latency_32B.ipynb)
   measured AWQ at ~1.7× the tokens/s of 8-bit for Qwen3-VL-32B),
   lower `--max-model-len`, cap `max_tokens` per request, add rate
   limits per project. Only after those are exhausted does a replica
   enter the conversation — as a *measured* response to the
   [saturation trigger](#triggers-for-changing-the-catalogue), not a
   default.
4. **Availability is the honest cost of no replicas.** A restart or
   node drain means that endpoint is down. During the pilot that is
   acceptable (and already what the Usage Policy promises); the
   small-fast model on the other card doubles as a degraded-mode
   fallback for the generalist.

Revisit this recommendation if the GPU count grows (NRP allocation,
future purchase) — at 4+ cards, one replica of the highest-traffic
endpoint becomes the right call.

---

## Draft catalogue v0 — what to stand up

Five standing endpoints covering the requested baseline (embedding,
reranker, general-purpose LLM) plus the two capabilities this repo's
apps already prove out (vision/OCR, small-fast).

> **On "Qwen 27B":** there is no 27B Qwen; the 27B you're thinking of
> is **Gemma 3 27B**. The closest Qwen is **Qwen3-32B** (dense). This
> draft picks Qwen3-32B as the generalist — same VRAM class, stronger
> benchmarks, first-class vLLM support, not license-gated, and it
> keeps the whole catalogue in one model family the cluster already
> caches. Gemma 3 27B is a fine alternate if a use case prefers its
> style; note it's HF-gated (license click-through + token).

### GPU 0 — generalist card

| Endpoint | Model | Quant | Weights | GPU fraction | Serves |
|----------|-------|-------|---------|--------------|--------|
| `general` | `Qwen/Qwen3-32B` | FP8 | ~33 GB | 0.80 (~77 GB) | Chat, RAG generation, code assistance, agent backends. ~40 GB KV cache → long context and many concurrent users |
| *(headroom)* | — | — | — | 0.20 | Burst scratch: ML Marathon experiments, short-lived workspaces, benchmark runs |

### GPU 1 — retrieval + vision card

| Endpoint | Model | Quant | Weights | GPU fraction | Serves |
|----------|-------|-------|---------|--------------|--------|
| `vision` | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` | AWQ 4-bit | ~20 GB | 0.45 | `ocr_app` document extraction, image QA, chart/figure reading |
| `embedding` | `Qwen/Qwen3-Embedding-4B` *(or keep `jinaai/jina-embeddings-v4`, already on the PVC and proven in `rag_app`)* | BF16 | ~8 GB | 0.15 | Vector search for every RAG project on campus |
| `reranker` | `Qwen/Qwen3-Reranker-4B` | BF16 | ~8 GB | 0.10 | Cross-encoder reranking behind any retrieval pipeline |
| `small-fast` | `Qwen/Qwen3-8B` | FP8 | ~9 GB | 0.30 | High-volume/batch jobs, classification sweeps, ML Marathon traffic, fallback when `general` is down |

Totals on GPU 1: ~45 GB weights, ~50 GB left for KV caches across the
four services — the same fractional-GPU pattern `rag_app` already runs
(vLLM 0.80 / embedding 0.10 / reranker 0.10 on one card).

**Consolidation option worth testing:** Qwen3-VL-32B's text-only
quality is close to the dense 32B. If benchmarks confirm it's "good
enough" as a generalist, one VL model could serve both `general` and
`vision`, freeing an entire card for more variety (a long-context
specialist, Whisper for transcription, or a code model). Run both for
a review cycle and let the metrics table decide.

**Explicitly not a standing service:** 70B+ models tensor-parallel
across both cards (Qwen2.5-72B, Qwen3-VL-72B, ~150B at 4-bit — all
feasible per [00 Overview](00-overview.md#hardware)). One TP-2 model
consumes the whole cluster. Offer these **on-request / scheduled**
(e.g. a lab books a week for evaluation), never as catalogue rows.

### ML Marathon mode

Marathon events are bursty: many students, small requests, short
window. Plan: keep the catalogue up, point marathon traffic at
`small-fast` and `general` with per-team rate limits, and repurpose
GPU 0's 0.20 headroom for event scratch. If a marathon project needs a
model outside the catalogue, that's the on-request path above — not a
permanent row.

---

## Catalogue tracking table

This is the living record — one row per endpoint. The **benchmark
columns are role-specific** (an embedding model and a chat model
should never be compared on the same number), and the **measured
columns come from our hardware**, not leaderboards, using
[`scripts/hardware_metrics.py`](../scripts/hardware_metrics.py) and
the latency notebook pattern.

| Column | Meaning |
|--------|---------|
| Endpoint / Model / Quant / GPU | Identity, as in the tables above |
| Stood up / Last reviewed / Owner | Accountability + staleness signal |
| **Benchmark (role-specific)** | Generalist: MMLU-Pro, GPQA-Diamond, LiveCodeBench. Embedding: MTEB v2 retrieval NDCG@10. Reranker: MTEB reranking / BEIR. VLM: OCRBench, DocVQA. Record the hosted model's published score. |
| **Gap-to-frontier** | Points behind the best *vLLM-servable open model in the same VRAM class* on that same benchmark. This is the number the update trigger watches. |
| **TTFT p50 / p95** | Time-to-first-token at realistic concurrency, measured on-cluster |
| **Tok/s @ N users** | Per-request decode speed at N concurrent requests (record N) |
| **Max concurrency** | Requests before queueing starts (KV-cache ceiling) |
| **Wh / 1k tokens** | Energy per 1k output tokens (`hardware_metrics.py`) — the WattBot angle; also the number peer campuses will ask for first |
| **Req/wk / Active projects** | Usage — drives the idle and saturation triggers |

Starter rows (fill measured columns during stand-up):

| Endpoint | Model | Quant | GPU | Benchmark (score) | Gap-to-frontier | TTFT p95 | Tok/s @ N | Max conc. | Wh/1k tok | Req/wk | Last reviewed |
|----------|-------|-------|-----|-------------------|-----------------|----------|-----------|-----------|-----------|--------|---------------|
| `general` | Qwen3-32B | FP8 | 0 (0.80) | MMLU-Pro: — | — | — | — | — | — | — | — |
| `vision` | Qwen3-VL-32B-Instruct-AWQ | AWQ | 1 (0.45) | OCRBench: — | — | — | — | — | — | — | — |
| `embedding` | Qwen3-Embedding-4B | BF16 | 1 (0.15) | MTEB ret.: — | — | — | — | — | — | — | — |
| `reranker` | Qwen3-Reranker-4B | BF16 | 1 (0.10) | MTEB rerank: — | — | — | — | — | — | — | — |
| `small-fast` | Qwen3-8B | FP8 | 1 (0.30) | MMLU-Pro: — | — | — | — | — | — | — | — |

Keep the canonical copy of this table in this file for now; if/when
other campuses want to consume it, export the same columns as
CSV/JSON (see [Growth path](#growth-path-nrp-and-peer-campuses)).

---

## Triggers for changing the catalogue

A model gets *reviewed* on any trigger below; it gets *replaced* only
after the [swap process](#swap-process) passes.

| # | Trigger | Threshold | Likely action |
|---|---------|-----------|---------------|
| 1 | **Frontier gap** | A vLLM-supported open model in the same VRAM class leads the hosted model by **≥5 points** on the role's tracked benchmark | Candidate swap |
| 2 | **Saturation** | TTFT p95 > 10 s or sustained queueing during **2+ consecutive normal weeks** | Tighter quant → lower max context → rate limits → *then* discuss a replica |
| 3 | **Unmet demand** | **≥3 distinct groups** request a capability the catalogue lacks (long-context, audio, code-specialist, …) | Slot review — what gets evicted or shrunk to make room? |
| 4 | **Idle** | An endpoint below ~20 req/wk for a full quarter | Demote to on-request; free the fraction |
| 5 | **Upstream event** | Model deprecated, license change, vLLM drops support, security advisory in the serving stack | Immediate review |
| 6 | **Cadence** | **Quarterly** review of every row, regardless — refresh Last reviewed, re-check gap-to-frontier | Keeps the table honest between events |

### Swap process

1. Open a GitHub issue in this repo: proposed model, role, published
   benchmark scores, VRAM/quant plan, what it replaces.
2. Provision weights to the shared PVC
   ([managing-models.md](../rag_app/docs/managing-models.md) — check
   vLLM compatibility first).
3. Stand up on GPU 0's headroom fraction (or off-hours) and measure
   the same columns as the incumbent: benchmarks, TTFT, tok/s,
   Wh/1k tokens.
4. It wins on the numbers → announce with **2 weeks' notice**, swap
   the endpoint, update the table, keep the old weights on the PVC
   for one review cycle for rollback.
5. Update this doc and any per-app docs whose model defaults changed
   (see the repo's `CLAUDE.md` doc-tracking table).

---

## Growth path: NRP and peer campuses

Design choices above that keep the federation door open:

- **OpenAI-compatible everywhere.** Every endpoint speaks `/v1` (vLLM
  default), so a client — or a gateway — can route across sites
  without code changes.
- **Portable catalogue schema.** The tracking table's columns are
  site-agnostic; exporting them as CSV/JSON lets each
  "BadgerBrain-like" deployment publish its own catalogue, and a
  LiteLLM-style gateway can eventually federate them (route to
  whichever site hosts the requested model, with local overflow to
  NRP-hosted endpoints for burst or 70B+ requests).
- **Energy column as a shared benchmark.** Wh/1k tokens measured the
  same way at each site is a rare apples-to-apples number across
  heterogeneous hardware — and a natural first joint publication with
  NRP/peer institutions.
- **Division of labor.** Local GPUs are the long-term home for
  workloads that will eventually touch institutional/sensitive data
  (pending security review); NRP is the natural overflow for large
  on-request models and burst capacity on public data.

Concrete next step when a collaboration starts: exchange catalogue
exports + measurement methodology before exchanging any traffic.

---

## Stand-up checklist (v0)

- [ ] Provision `Qwen/Qwen3-32B` (FP8), `Qwen/Qwen3-8B` (FP8),
      `Qwen/Qwen3-Embedding-4B`, `Qwen/Qwen3-Reranker-4B` to the
      shared PVC (`Qwen3-VL-32B-AWQ` and Jina V4 are already cached)
- [ ] Deploy the five endpoints per [03 Share as endpoint](03-share-as-endpoint.md),
      fractions as tabled above
- [ ] Run the benchmark + `hardware_metrics.py` pass; fill the
      measured columns
- [ ] Decide Jina V4 vs. Qwen3-Embedding-4B on MTEB + measured latency
      (Jina wins ties: already proven in `rag_app` and multimodal)
- [ ] Test the consolidation option (VL-32B as generalist) during one
      review cycle
- [ ] Publish endpoint URLs + this catalogue to pilot users; set the
      first quarterly review date
